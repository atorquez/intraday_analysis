# ==============================================================================
# MASTER INTRADAY RANKER — BUYZONE + PCA + VMAS + MASTER SCANNER UNIFIED ENGINE
# STREAMLIT-FIXED VERSION — Uses st.cache_resource for cross-run persistence
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# STREAMLIT-COMPATIBLE COMPANY NAME CACHE
# ---------------------------------------------------------
# NOTE: For Streamlit, use st.cache_resource instead of module-level dict.
# If not using Streamlit, the fallback module-level cache still works.
try:
    import streamlit as st
    _USING_STREAMLIT = True
except ImportError:
    _USING_STREAMLIT = False

# Module-level fallback cache (for non-Streamlit usage)
_module_name_cache = {}

def _get_cache_dict():
    """Returns a cache dict that persists across Streamlit runs."""
    if _USING_STREAMLIT:
        # st.cache_resource persists across script re-runs
        @st.cache_resource
        def _cached_names():
            return {}
        return _cached_names()
    return _module_name_cache

# ---------------------------------------------------------
# INTRADAY BUY ZONE (Daily Value Zone for Page 1)
# ---------------------------------------------------------
def intraday_buy_zone(df, lookback=10, percentile=0.15):
    if df is None or df.empty:
        return None

    closes = df["Close"].tail(lookback).values
    if len(closes) < lookback:
        return None

    return float(np.percentile(closes, percentile * 100))

# ---------------------------------------------------------
# COMPANY NAME LOOKUP (FIXED: Streamlit-persistent cache)
# ---------------------------------------------------------
def get_company_name(ticker):
    cache = _get_cache_dict()
    if ticker in cache:
        return cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName", ticker)
        cache[ticker] = name
        return name
    except Exception as e:
        logger.warning(f"Failed to get company name for {ticker}: {e}")
        cache[ticker] = ticker
        return ticker

# ---------------------------------------------------------
# FETCH DAILY DATA
# ---------------------------------------------------------
def fetch_daily(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df

# ---------------------------------------------------------
# FETCH INTRADAY DATA (1-minute)
# ---------------------------------------------------------
def fetch_intraday(ticker):
    try:
        df = yf.download(
            tickers=ticker,
            period="1d",
            interval="1m",
            progress=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df

    except Exception as e:
        logger.warning(f"Failed to fetch intraday data for {ticker}: {e}")
        return None

# ---------------------------------------------------------
# EXTRA INDICATORS FOR PCA (With Timeframe Guardrails)
# ---------------------------------------------------------
def compute_extra_indicators(df, is_intraday=True):
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()

    df["BB_Width"] = np.where(
        df["SMA20"] != 0,
        (df["STD20"] * 2) / df["SMA20"],
        np.nan
    )

    df["ROC"] = df["Close"].pct_change(10)

    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()

    stoch_range = high_max - low_min
    df["StochK"] = np.where(
        stoch_range != 0,
        (df["Close"] - low_min) / stoch_range,
        0.5
    )
    df["StochK"] = df["StochK"].clip(0, 1)

    df["EMA_Curve"] = df["Close"].ewm(span=9).mean() - df["Close"].ewm(span=20).mean()

    df["VolDelta"] = df["Volume"].diff()

    if is_intraday:
        df["VWAP"] = (df["Volume"] * df["Close"]).cumsum() / df["Volume"].cumsum()
    else:
        df["VWAP"] = (df["Close"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

    df["VWAP_Dist"] = df["Close"] - df["VWAP"]

    return df.dropna()

# ---------------------------------------------------------
# PCA COMPONENTS (FIXED: Proper Index Alignment)
# ---------------------------------------------------------
def append_pca_components(df, is_intraday=True):
    processed_df = compute_extra_indicators(df, is_intraday=is_intraday)

    pca_features = processed_df[[
        "RSI",
        "BB_Width",
        "ROC",
        "StochK",
        "EMA_Curve",
        "VolDelta",
        "VWAP_Dist"
    ]].dropna()

    if len(pca_features) < 10:
        processed_df["PCA1"] = np.nan
        processed_df["PCA2"] = np.nan
        processed_df["PCA3"] = np.nan
        return processed_df

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pca_features)

    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(
        components,
        index=pca_features.index,
        columns=["PCA1", "PCA2", "PCA3"]
    )

    processed_df["PCA1"] = pca_df["PCA1"].reindex(processed_df.index)
    processed_df["PCA2"] = pca_df["PCA2"].reindex(processed_df.index)
    processed_df["PCA3"] = pca_df["PCA3"].reindex(processed_df.index)

    return processed_df

# ---------------------------------------------------------
# INDICATORS (DAILY - FIXED TO READ PCA MATRIX)
# ---------------------------------------------------------
def calculate_indicators(df):
    df = append_pca_components(df.copy(), is_intraday=False)
    if df.empty:
        return None

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    df["ATR%"] = np.where(
        df["Close"] != 0,
        (df["ATR"] / df["Close"]) * 100,
        np.nan
    )

    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Gap%"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100

    last = df.iloc[-1]

    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        trend = "UP"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        trend = "DOWN"
    else:
        trend = "FLAT"

    ema9 = last["EMA9"]
    ema20 = last["EMA20"]
    ema9_slope = df["EMA9"].iloc[-1] - df["EMA9"].iloc[-5]
    ema20_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]

    pca1 = last["PCA1"] if not pd.isna(last["PCA1"]) else None

    if ema9 > ema20 and ema9_slope > 0 and ema20_slope > 0 and trend == "UP" and (pca1 is None or pca1 > 0):
        daily_execution = "Ready"
    elif ema9 > ema20 and (ema9_slope < 0 or ema20_slope < 0):
        daily_execution = "False Ready"
    elif abs((ema9 - ema20) / ema20) < 0.003:
        daily_execution = "Crossing Soon"
    else:
        daily_execution = "Setup Only"

    return {
        "Close": round(last["Close"], 2),
        "ATR%": round(last["ATR%"], 2) if not pd.isna(last["ATR%"]) else None,
        "RVOL": round(last["RVOL"], 2) if not pd.isna(last["RVOL"]) else None,
        "Gap%": round(last["Gap%"], 2) if not pd.isna(last["Gap%"]) else None,
        "EMA9": ema9,
        "EMA20": ema20,
        "EMA50": last["EMA50"],
        "Trend": trend,
        "Execution_Status": daily_execution,
        "PCA1": round(pca1, 4) if pca1 is not None else None
    }

# ---------------------------------------------------------
# INTRADAY EXECUTION ENGINE
# ---------------------------------------------------------
def calculate_intraday_execution(df_intraday):
    df = append_pca_components(df_intraday, is_intraday=True)
    if df.empty:
        return None

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    ema9_slope = df["EMA9"].iloc[-1] - df["EMA9"].iloc[-3]
    ema20_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-3]

    df["ROC"] = df["Close"].pct_change() * 100
    intraday_momentum = df["ROC"].iloc[-1]

    if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1]:
        intraday_trend = "UP"
    else:
        intraday_trend = "DOWN"

    return {
        "intraday_ema9": df["EMA9"].iloc[-1],
        "intraday_ema20": df["EMA20"].iloc[-1],
        "intraday_ema9_slope": ema9_slope,
        "intraday_ema20_slope": ema20_slope,
        "intraday_momentum": intraday_momentum,
        "intraday_trend": intraday_trend,
        "intraday_pca1": df["PCA1"].iloc[-1] if not pd.isna(df["PCA1"].iloc[-1]) else None
    }

# ---------------------------------------------------------
# SCORING ENGINE
# ---------------------------------------------------------
def score_stock(ind):
    score = 5  # Baseline

    if ind["EMA9"] > ind["EMA20"]:
        score += 10
    if ind["EMA20"] > ind["EMA50"]:
        score += 10

    if ind["RVOL"] > 2:
        score += 20

    if 2 <= ind["Gap%"] <= 5:
        score += 10

    if ind["ATR%"] > 5:
        score += 15

    return score

# ---------------------------------------------------------
# VMAS
# ---------------------------------------------------------
def compute_vmas(price, buyzone10, pca1):
    if price is None or buyzone10 is None or pca1 is None:
        return None

    if abs(buyzone10) < 0.01:
        return None

    dist = (price - buyzone10) / buyzone10
    return (1 - dist) * pca1

# ---------------------------------------------------------
# BUYZONE HEATMAP
# ---------------------------------------------------------
def buyzone_heatmap(price, bz10, bz5, dist10):
    if price is None or bz10 is None or bz5 is None or dist10 is None:
        return "UNKNOWN"

    if price <= bz10:
        return "DEEP_VALUE_ZONE"
    elif price <= bz5:
        return "MID_VALUE_ZONE"
    elif dist10 < 0.02:
        return "NEAR_VALUE_ZONE"
    else:
        return "EXTENDED_ZONE"

# ---------------------------------------------------------
# MASTER ORCHESTRATOR: RANK UNIVERSE
# ---------------------------------------------------------
def rank_universe(tickers, buy_zone_percentile=0.15):
    if not tickers:
        return pd.DataFrame()

    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = [t.strip().upper() for t in tickers if t and isinstance(t, str)]

    if not tickers:
        return pd.DataFrame()

    ranked_results = []

    try:
        bulk_data = yf.download(
            tickers=tickers, 
            period="3mo", 
            interval="1d", 
            progress=False, 
            group_by="ticker"
        )
    except Exception as e:
        logger.error(f"Bulk data request failed: {e}")
        return pd.DataFrame()

    is_multi = isinstance(bulk_data.columns, pd.MultiIndex)

    for ticker in tickers:
        try:
            if is_multi:
                if ticker in bulk_data.columns.get_level_values(0):
                    daily_df = bulk_data[ticker].dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                else:
                    continue
            else:
                if len(tickers) == 1 and ticker == tickers[0]:
                    daily_df = bulk_data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
                else:
                    continue

            if daily_df.empty or len(daily_df) < 50:
                continue

            metrics = calculate_indicators(daily_df)
            if not metrics:
                continue

            bz10 = intraday_buy_zone(daily_df, lookback=10, percentile=buy_zone_percentile)
            bz5 = intraday_buy_zone(daily_df, lookback=5, percentile=buy_zone_percentile)

            company_name = get_company_name(ticker)
            score = score_stock(metrics)

            current_price = metrics["Close"]

            if bz10 and abs(bz10) > 0.01:
                dist_10 = (current_price - bz10) / bz10
            else:
                dist_10 = 0

            heatmap_status = buyzone_heatmap(current_price, bz10, bz5, dist_10)

            vmas_score = None
            if metrics["PCA1"] is not None and bz10 is not None and abs(bz10) > 0.01:
                vmas_score = compute_vmas(current_price, bz10, metrics["PCA1"])

            stock_data = {
                "Ticker": ticker,
                "Company": company_name,
                "Score": score,
                "Execution": metrics["Execution_Status"],
                "Trend": metrics["Trend"],
                "Close": current_price,
                "RVOL": metrics["RVOL"],
                "Gap%": metrics["Gap%"],
                "ATR%": metrics["ATR%"],
                "PCA1": metrics["PCA1"],
                "VMAS": round(vmas_score, 4) if vmas_score is not None else None,
                "BuyZone_Heatmap": heatmap_status
            }

            ranked_results.append(stock_data)

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            continue

    if not ranked_results:
        return pd.DataFrame()

    master_df = pd.DataFrame(ranked_results)
    master_df = master_df.sort_values(by=["Score", "RVOL"], ascending=[False, False])

    return master_df.reset_index(drop=True)