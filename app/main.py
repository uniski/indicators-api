import json, os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
import ccxt
import pandas as pd

# ta indicators
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, ADXIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

# ---------- Config ----------
BASIC_SMA_WINDOWS = [10, 20, 50, 200]
BASIC_RSI_LENGTH = 14

SUPPORTED_EXCHANGES = {"binance", "coinbase", "kraken"}
SUPPORTED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

# x402 config (off by default; Render env overrides these)
X402_ENABLED = os.getenv("X402_ENABLED", "false").lower() == "true"
X402_CHAIN = os.getenv("X402_CHAIN", "base-sepolia")
X402_ASSET = os.getenv("X402_ASSET", "USDC")
X402_RECEIVER = os.getenv("X402_RECEIVER", "")
X402_PRICE_BASIC = os.getenv("X402_PRICE_BASIC", "0.005")
X402_PRICE_PRO = os.getenv("X402_PRICE_PRO", "0.010")
X402_PRICE_CANDLES = os.getenv("X402_PRICE_CANDLES", X402_PRICE_BASIC)

app = FastAPI(title="Trading Indicators API")

# ---------- Helpers ----------
def _ensure_inputs(symbol: str, exchange: str, interval: str):
    if exchange not in SUPPORTED_EXCHANGES:
        raise HTTPException(400, f"exchange must be one of {sorted(SUPPORTED_EXCHANGES)}")
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, f"interval must be one of {sorted(SUPPORTED_INTERVALS)}")
    if not isinstance(symbol, str) or ("/" not in symbol and "-" not in symbol):
        # basic sanity; exact formats validated by the exchange
        pass

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV to a higher timeframe using OHLCV aggregation."""
    if df.empty:
        return df
    dfi = df.set_index("ts")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = dfi.resample(rule, label="right", closed="right").agg(agg).dropna()
    return out.reset_index()

def fetch_ohlcv(symbol: str, interval: str, limit: int, exchange_name: str) -> pd.DataFrame:
    """
    Fetch OHLCV candles (timestamp, open, high, low, close, volume)
    from supported exchanges. Auto-resamples Coinbase 4h.
    """
    _ensure_inputs(symbol, exchange_name, interval)

    if not hasattr(ccxt, exchange_name):
        raise HTTPException(400, f"Unknown exchange '{exchange_name}'")

    ex = getattr(ccxt, exchange_name)()
    if not ex.has.get("fetchOHLCV", False):
        raise HTTPException(400, f"Exchange '{exchange_name}' does not support fetchOHLCV")

    fetch_interval = interval
    post_resample = None

    # Coinbase doesn’t support 4h — fetch 1h, then resample to 4H
    if exchange_name == "coinbase" and interval == "4h":
        fetch_interval, post_resample = "1h", "4H"

    # If resampling, grab extra candles so the last bar can form completely
    fetch_limit = limit * (4 if post_resample else 1)

    try:
        candles = ex.fetch_ohlcv(symbol, timeframe=fetch_interval, limit=fetch_limit)
    except Exception as e:
        raise HTTPException(400, f"CCXT error: {e}")

    if not candles:
        raise HTTPException(404, "No candles returned")

    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    if post_resample:
        df = _resample_ohlcv(df, post_resample)
        if len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)

    return df

def _serialize_candles(df: pd.DataFrame):
    """Convert OHLCV dataframe into a list of dicts for JSON output."""
    return [
        {
            "ts": row.ts.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in df.itertuples(index=False)
    ]

# ---- Indicator Calculations ----
def compute_basic(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    close = df["close"]
    out["sma"] = {str(n): float(SMAIndicator(close=close, window=n).sma_indicator().iloc[-1])
                  for n in BASIC_SMA_WINDOWS}
    rsi_val = float(RSIIndicator(close=close, window=BASIC_RSI_LENGTH).rsi().iloc[-1])
    out["rsi"] = {str(BASIC_RSI_LENGTH): rsi_val}
    return out

def compute_pro(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # Bollinger Bands (20, 2)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    basis = float(bb.bollinger_mavg().iloc[-1])
    upper = float(bb.bollinger_hband().iloc[-1])
    lower = float(bb.bollinger_lband().iloc[-1])
    width = upper - lower
    price = float(close.iloc[-1])
    out["bb"] = {
        "basis": basis,
        "upper": upper,
        "lower": lower,
        "bandwidth": float((width / basis) if basis else 0.0),
        "percentB": float((price - lower) / width) if width else 0.0,
    }

    # MACD (12,26,9)
    macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = {
        "macd": float(macd_obj.macd().iloc[-1]),
        "signal": float(macd_obj.macd_signal().iloc[-1]),
        "hist": float(macd_obj.macd_diff().iloc[-1]),
    }

    # Stochastic RSI
    st = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
    out["stoch_rsi"] = {
        "k": float(st.stochrsi_k().iloc[-1]),
        "d": float(st.stochrsi_d().iloc[-1]),
    }

    # ATR (14)
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]
    out["atr"] = {"14": float(atr)}

    # ADX (+/-DI)
    adx_obj = ADXIndicator(high=high, low=low, close=close, window=14)
    out["adx"] = {
        "adx": float(adx_obj.adx().iloc[-1]),
        "+di": float(adx_obj.adx_pos().iloc[-1]),
        "-di": float(adx_obj.adx_neg().iloc[-1]),
    }

    # VWAP (works best intraday)
    try:
        vwap = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=vol, window=14
        ).volume_weighted_average_price().iloc[-1]
        out["vwap"] = float(vwap)
    except Exception:
        pass

    return out

# ---- Meta + Payment ----
def meta(df: pd.DataFrame, symbol: str, exchange: str, interval: str):
    return {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "candles": int(len(df)),
        "last_candle_iso": df["ts"].iloc[-1].isoformat() if len(df) else None,
    }

def x402_requirements(price: str):
    if not (X402_RECEIVER and X402_ASSET and X402_CHAIN):
        raise HTTPException(500, "x402 not configured: set X402_RECEIVER, X402_ASSET, X402_CHAIN")
    return {
        "version": "1.0",
        "payment": {
            "chain": X402_CHAIN,
            "asset": X402_ASSET,
            "receiver": X402_RECEIVER,
            "amount": price,
            "currency": "USDC",
            "expires_in": 300,
        },
    }

def maybe_require_payment(kind: str):
    if not X402_ENABLED:
        return (None, None)
    if kind == "basic":
        price = X402_PRICE_BASIC
    elif kind == "pro":
        price = X402_PRICE_PRO
    elif kind == "candles":
        price = X402_PRICE_CANDLES
    else:
        price = X402_PRICE_BASIC
    return (402, x402_requirements(price))

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/v1/indicators/basic")
def indicators_basic(
    request: Request,
    symbol: str = Query(..., description="Binance: BTC/USDT, Coinbase: BTC-USD, Kraken: BTC/USDT"),
    exchange: str = Query("binance"),
    interval: str = Query("1h", description="1m,5m,15m,1h,4h,1d"),
    limit: int = Query(500, ge=100, le=2000),
):
    code, pay = maybe_require_payment("basic")
    if code:
        return JSONResponse(status_code=code, content=pay)
    df = fetch_ohlcv(symbol, interval, limit, exchange)
    return {"meta": meta(df, symbol, exchange, interval), "latest": compute_basic(df)}

@app.get("/v1/indicators/pro")
def indicators_pro(
    request: Request,
    symbol: str = Query(...),
    exchange: str = Query("binance"),
    interval: str = Query("1h"),
    limit: int = Query(500, ge=100, le=2000),
):
    code, pay = maybe_require_payment("pro")
    if code:
        return JSONResponse(status_code=code, content=pay)
    df = fetch_ohlcv(symbol, interval, limit, exchange)
    return {"meta": meta(df, symbol, exchange, interval), "latest": compute_pro(df)}

@app.get("/v1/candles")
def get_candles(
    request: Request,
    symbol: str = Query(..., description="Binance: BTC/USDT, Coinbase: BTC-USD, Kraken: BTC/USDT"),
    exchange: str = Query("binance"),
    interval: str = Query("1h", description="1m,5m,15m,1h,4h,1d"),
    limit: int = Query(500, ge=1, le=2000),
    resample: str | None = Query(None, description="Optional: 1m,5m,15m,1h,4h,1d to aggregate server-side"),
):
    code, pay = maybe_require_payment("candles")
    if code:
        return JSONResponse(status_code=code, content=pay)

    df = fetch_ohlcv(symbol, interval, limit, exchange)

    if resample:
        rule_map = {"1m": "1T", "5m": "5T", "15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
        rule = rule_map.get(resample.lower())
        if not rule:
            raise HTTPException(400, "resample must be one of 1m,5m,15m,1h,4h,1d")
        df = _resample_ohlcv(df, rule)
        df = df.iloc[-limit:].reset_index(drop=True)

    return {
        "meta": {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "returned_interval": resample or interval,
            "candles": int(len(df)),
            "last_candle_iso": df["ts"].iloc[-1].isoformat() if len(df) else None,
        },
        "candles": _serialize_candles(df),
    }
