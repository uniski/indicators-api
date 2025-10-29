# app/main.py

import os, json
from typing import Dict, Any
from decimal import Decimal, InvalidOperation

import ccxt
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

# ---- TA indicators
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, ADXIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

# ---------- Config ----------
BASIC_SMA_WINDOWS = [10, 20, 50, 200]
BASIC_RSI_LENGTH = 14

SUPPORTED_EXCHANGES = {"binance", "coinbase", "kraken"}
SUPPORTED_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

# x402 config (Render env overrides these)
X402_ENABLED = os.getenv("X402_ENABLED", "false").lower() == "true"
X402_CHAIN = os.getenv("X402_CHAIN", "base")     # base mainnet by default
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
        # basic sanity; exact symbol format is validated by the exchange
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
    from supported exchanges. Auto-resamples Coinbase 4h from 1h.
    """
    _ensure_inputs(symbol, exchange_name, interval)

    if not hasattr(ccxt, exchange_name):
        raise HTTPException(400, f"Unknown exchange '{exchange_name}'")

    ex = getattr(ccxt, exchange_name)()
    if not ex.has.get("fetchOHLCV", False):
        raise HTTPException(400, f"Exchange '{exchange_name}' does not support fetchOHLCV")

    fetch_interval, post_resample = interval, None
    if exchange_name == "coinbase" and interval == "4h":
        fetch_interval, post_resample = "1h", "4H"

    fetch_limit = limit * (4 if post_resample else 1)

    try:
        candles = ex.fetch_ohlcv(symbol, timeframe=fetch_interval, limit=fetch_limit)
    except Exception as e:
        msg = str(e)
        if "451" in msg or "restricted location" in msg.lower():
            raise HTTPException(403, "Exchange blocked from this server location. Try exchange=kraken or coinbase.")
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

    macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = {
        "macd": float(macd_obj.macd().iloc[-1]),
        "signal": float(macd_obj.macd_signal().iloc[-1]),
        "hist": float(macd_obj.macd_diff().iloc[-1]),
    }

    st = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
    out["stoch_rsi"] = {"k": float(st.stochrsi_k().iloc[-1]), "d": float(st.stochrsi_d().iloc[-1])}

    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]
    out["atr"] = {"14": float(atr)}

    adx_obj = ADXIndicator(high=high, low=low, close=close, window=14)
    out["adx"] = {
        "adx": float(adx_obj.adx().iloc[-1]),
        "+di": float(adx_obj.adx_pos().iloc[-1]),
        "-di": float(adx_obj.adx_neg().iloc[-1]),
    }

    try:
        vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=vol, window=14
        ).volume_weighted_average_price().iloc[-1]
        out["vwap"] = float(vwap)
    except Exception:
        pass

    return out

# ---- Meta ----
def meta(df: pd.DataFrame, symbol: str, exchange: str, interval: str):
    return {
        "symbol": symbol,
        "exchange": exchange,
        "interval": interval,
        "candles": int(len(df)),
        "last_candle_iso": df["ts"].iloc[-1].isoformat() if len(df) else None,
    }

# ---------- x402 helpers ----------
def _normalize_amount(price: str) -> str:
    """
    Sanitize env-provided price into a strict decimal STRING for x402scan:
    - strips stray quotes/spaces
    - parses as Decimal
    - must be > 0
    - formats as 6-decimal string (no exponent)
    """
    s = str(price).strip()
    if s.startswith(("'", '"')) and s.endswith(("'", '"')) and len(s) >= 2:
        s = s[1:-1].strip()
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        raise HTTPException(500, "Invalid x402 price format")
    if d <= 0:
        raise HTTPException(500, "x402 price must be > 0")
    return f"{d:.6f}"

def _x402_accepts(kind: str, request: Request, price: str) -> Dict[str, Any]:
    resource_url = str(request.url)
    description_by_kind = {
        "basic": "Basic indicators: SMA(10/20/50/200), RSI(14).",
        "pro": "Pro indicators: Bollinger Bands, MACD, StochRSI, ATR, ADX, VWAP.",
        "candles": "Raw OHLCV candles for charting.",
    }

    amount_str = _normalize_amount(price)

    query_schema = {
        "symbol": {"type": "string", "required": True, "description": "Trading pair"},
        "exchange": {"type": "string", "required": False, "enum": sorted(list(SUPPORTED_EXCHANGES))},
        "interval": {"type": "string", "required": False, "enum": sorted(list(SUPPORTED_INTERVALS))},
        "limit": {"type": "number", "required": False, "description": "1–2000"},
    }

    if kind == "basic":
        output_schema = {
            "meta": {"symbol": "string", "exchange": "string", "interval": "string", "candles": "number", "last_candle_iso": "string|null"},
            "latest": {"sma": {"10": "number", "20": "number", "50": "number", "200": "number"}, "rsi": {"14": "number"}},
        }
    elif kind == "pro":
        output_schema = {
            "meta": {"symbol": "string", "exchange": "string", "interval": "string", "candles": "number", "last_candle_iso": "string|null"},
            "latest": {
                "bb": {"basis": "number", "upper": "number", "lower": "number", "bandwidth": "number", "percentB": "number"},
                "macd": {"macd": "number", "signal": "number", "hist": "number"},
                "stoch_rsi": {"k": "number", "d": "number"},
                "atr": {"14": "number"},
                "adx": {"adx": "number", "+di": "number", "-di": "number"},
                "vwap": "number",
            },
        }
    else:  # candles
        output_schema = {
            "meta": {"symbol": "string", "exchange": "string", "interval": "string", "returned_interval": "string", "candles": "number", "last_candle_iso": "string|null"},
            "candles": [{"ts": "string", "open": "number", "high": "number", "low": "number", "close": "number", "volume": "number"}],
        }
        # allow optional resample in input for /v1/candles
        query_schema["resample"] = {"type": "string", "required": False, "enum": sorted(list(SUPPORTED_INTERVALS))}

    return {
        "scheme": "exact",
        "network": "base",
        "maxAmountRequired": amount_str,  # STRING per x402scan
        "resource": resource_url,
        "description": description_by_kind.get(kind, ""),
        "mimeType": "application/json",
        "payTo": X402_RECEIVER,
        "maxTimeoutSeconds": 300,
        "asset": X402_ASSET,
        "outputSchema": {"input": {"type": "http", "method": "GET", "queryParams": query_schema}, "output": output_schema},
        "extra": {"tier": kind},
    }

def _x402_response(kind: str, request: Request, price: str) -> Dict[str, Any]:
    if not X402_RECEIVER or not X402_ASSET or not X402_CHAIN:
        return {"x402Version": 1, "error": "x402 not configured: set X402_RECEIVER, X402_ASSET, X402_CHAIN"}
    return {"x402Version": 1, "accepts": [_x402_accepts(kind, request, price)]}

def maybe_require_payment(kind: str, request: Request):
    if not X402_ENABLED:
        return (None, None)
    price = {"basic": X402_PRICE_BASIC, "pro": X402_PRICE_PRO, "candles": X402_PRICE_CANDLES}.get(kind, X402_PRICE_BASIC)
    return (402, _x402_response(kind, request, price))

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/v1/indicators/basic")
def indicators_basic(
    request: Request,
    symbol: str = Query(..., description="Binance: BTC/USDT, Coinbase: BTC-USD, Kraken: BTC/USDT"),
    exchange: str = Query("binance"),
    interval: str = Query("1h"),
    limit: int = Query(500, ge=1, le=2000),
):
    code, pay = maybe_require_payment("basic", request)
    if code:
        return Response(status_code=code, content=json.dumps(pay), media_type="application/json")
    df = fetch_ohlcv(symbol, interval, limit, exchange)
    return {"meta": meta(df, symbol, exchange, interval), "latest": compute_basic(df)}

@app.get("/v1/indicators/pro")
def indicators_pro(
    request: Request,
    symbol: str = Query(...),
    exchange: str = Query("binance"),
    interval: str = Query("1h"),
    limit: int = Query(500, ge=1, le=2000),
):
    code, pay = maybe_require_payment("pro", request)
    if code:
        return Response(status_code=code, content=json.dumps(pay), media_type="application/json")
    df = fetch_ohlcv(symbol, interval, limit, exchange)
    return {"meta": meta(df, symbol, exchange, interval), "latest": compute_pro(df)}

@app.get("/v1/candles")
def get_candles(
    request: Request,
    symbol: str = Query(..., description="Binance: BTC/USDT, Coinbase: BTC-USD, Kraken: BTC/USDT"),
    exchange: str = Query("binance"),
    interval: str = Query("1h"),
    limit: int = Query(500, ge=1, le=2000),
    resample: str | None = Query(None, description="Optional: 1m,5m,15m,1h,4h,1d to aggregate server-side"),
):
    code, pay = maybe_require_payment("candles", request)
    if code:
        return Response(status_code=code, content=json.dumps(pay), media_type="application/json")
    df = fetch_ohlcv(symbol, interval, limit, exchange)
    if resample:
        rule_map = {"1m": "1T", "5m": "5T", "15m": "15T", "1h": "1H", "4h": "4H", "1d": "1D"}
        rule = rule_map.get(resample.lower())
        if not rule:
            raise HTTPException(400, "resample must be one of 1m,5m,15m,1h,4h,1d")
        df = _resample_ohlcv(df, rule)
        df = df.iloc[-limit:].reset_index(drop=True)
    return {"meta": meta(df, symbol, exchange, interval), "candles": _serialize_candles(df)}
