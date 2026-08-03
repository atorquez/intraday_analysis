# ==============================================================================
# SPC Intraday Ranker v3.0 (Hybrid) — FULLY PATCHED FOR 729 MULTI-EXCHANGE TICKERS
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

print(">>> intraday_ranker_v3 LOADED <<<")

def _safe_flatten_columns(df):
    """Safely handles yfinance MultiIndex column layouts without damaging caches."""
    if df is None or df.empty:
        return df
    
    df_copy = df.copy() # Creates a clean frame copy to make thread processing isolated
    if isinstance(df_copy.columns, pd.MultiIndex):
        # Dynamically scans the tuple levels to find standard data field labels
        new_cols = []
        for col_tuple in df_copy.columns:
            # Captures standard data fields if yfinance flips string position level rankings
            field = next((level for level in col_tuple if level in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]), col_tuple[0])
            new_cols.append(field)
        df_copy.columns = new_cols
        
    return df_copy

# ---------------------------------------------------------
# FETCH DAILY DATA
# ---------------------------------------------------------
def fetch_daily(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return None

        df = _safe_flatten_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df
    except Exception as e:
        logger.warning(f"Daily fetch error for {ticker}: {e}")
        return None

# ---------------------------------------------------------
# FETCH INTRADAY DATA (1m)
# ---------------------------------------------------------
def fetch_intraday(ticker):
    try:
        df = yf.download(tickers=ticker, period="1d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            return None

        df = _safe_flatten_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df
    except Exception as e:
        logger.warning(f"Intraday fetch error for {ticker}: {e}")
        return None

# ---------------------------------------------------------
# EXTRA INDICATORS FOR PCA
# ---------------------------------------------------------
def compute_extra_indicators(df, is_intraday=True):
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame()
        
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["STD20"] = df["Close"].rolling(20, min_periods=1).std().fillna(0)
    df["BB_Width"] = np.where(
        df["SMA20"] != 0,
        (df["STD20"] * 2) / df["SMA20"],
        0.0
    )

    df["ROC"] = df["Close"].pct_change(10).fillna(0)

    low_min = df["Low"].rolling(14, min_periods=1).min()
    high_max = df["High"].rolling(14, min_periods=1).max()
    stoch_range = high_max - low_min
    df["StochK"] = np.where(stoch_range != 0, (df["Close"] - low_min) / stoch_range, 0.5)
    df["StochK"] = df["StochK"].clip(0, 1)

    df["EMA_Curve"] = df["Close"].ewm(span=9).mean() - df["Close"].ewm(span=20).mean()
    df["VolDelta"] = df["Volume"].diff().fillna(0)

    if is_intraday:
        cv = df["Volume"] * df["Close"]
        df["VWAP"] = cv.cumsum() / df["Volume"].cumsum()
    else:
        df["VWAP"] = (df["Close"] * df["Volume"]).rolling(20, min_periods=1).sum() / \
                     df["Volume"].rolling(20, min_periods=1).sum()

    df["VWAP_Dist"] = df["Close"] - df["VWAP"]
    return df

# ---------------------------------------------------------
# PCA ENGINE (3 components, hybrid)
# ---------------------------------------------------------
def append_pca_components(df, is_intraday=True):
    processed = compute_extra_indicators(df, is_intraday=is_intraday)
    if processed.empty:
        # Fallback allocation to avoid downstream structural rank breaking
        df["PCA1"] = np.nan
        df["PCA2"] = np.nan
        df["PCA3"] = np.nan
        return df

    feature_cols = ["RSI", "BB_Width", "ROC", "StochK", "EMA_Curve", "VolDelta", "VWAP_Dist"]
    pca_features = processed[feature_cols].copy()

    if len(pca_features) < 10:
        processed["PCA1"] = np.nan
        processed["PCA2"] = np.nan
        processed["PCA3"] = np.nan
        return processed

    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pca_features)
        pca = PCA(n_components=3)
        components = pca.fit_transform(scaled)
        pca_df = pd.DataFrame(components, index=pca_features.index, columns=["PCA1", "PCA2", "PCA3"])
        processed = processed.join(pca_df, how="left")
    except Exception as e:
        logger.warning(f"PCA error: {e}")
        processed["PCA1"] = np.nan
        processed["PCA2"] = np.nan
        processed["PCA3"] = np.nan

    return processed
# ---------------------------------------------------------
# DAILY INDICATOR + EXECUTION ENGINE (HYBRID — SECURED)
# ---------------------------------------------------------
import streamlit as st # Imported to access the high-speed global cache network

def calculate_indicators(df):
    if df is None or df.empty or len(df) < 40:
        return None

    # PREMIUM FILTER — tests premium-only universe bounds safely
    if float(df["Close"].iloc[-1]) < 40:
        return None

    df = append_pca_components(df.copy(), is_intraday=False)

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    df["ATR%"] = np.where(df["Close"] != 0, (df["ATR"] / df["Close"]) * 100, np.nan)
    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Gap%"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100

    last = df.iloc[-1]

    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        trend = "UP"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        trend = "DOWN"
    else:
        trend = "FLAT"

    ema9_slope = df["EMA9"].iloc[-1] - df["EMA9"].iloc[-5]
    ema20_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]
    pca1 = last["PCA1"] if not pd.isna(last["PCA1"]) else None

    # Fixed: Prevent division-by-zero errors if EMA20 sits flat
    ema20_val = last["EMA20"] if last["EMA20"] != 0 else 1.0
    proximity_metric = abs((last["EMA9"] - last["EMA20"]) / ema20_val)

    if (
        trend == "UP"
        and last["EMA9"] > last["EMA20"]
        and ema9_slope > 0
        and ema20_slope > 0
        and (pca1 is None or pca1 > 0)
    ):
        execution = "Watch List"
    elif last["EMA9"] > last["EMA20"] and (ema9_slope < 0 or ema20_slope < 0):
        execution = "Not Watch List"
    elif proximity_metric < 0.003:
        execution = "Crossing Soon"
    else:
        execution = "Setup Only"

    return {
        "Close": round(float(last["Close"]), 2),
        "ATR%": round(float(last["ATR%"]), 2) if not pd.isna(last["ATR%"]) else 0.0,
        "RVOL": round(float(last["RVOL"]), 2) if not pd.isna(last["RVOL"]) else 0.0,
        "Gap%": round(float(last["Gap%"]), 2) if not pd.isna(last["Gap%"]) else 0.0,
        "Trend": trend,
        "Execution": execution,
        "PCA1": float(pca1) if pca1 is not None else np.nan,
        "Avg_Volume_20d": float(df["Volume"].rolling(20).mean().iloc[-1]),
    }

# ---------------------------------------------------------
# RANK UNIVERSE (UPGRADED HIGH-SPEED GLOBAL DATA INTERFACE)
# ---------------------------------------------------------
def rank_universe(tickers, buy_zone_percentile=0.15):
    print(">>> rank_universe() PIPELINE EXECUTED <<<")

    rows = []

    for ticker in tickers:
        # HOOKS GLOBAL MEMORY PRELOADER: Checks if cache is loaded before downloading
        try:
            # References the exact TTL signature from your warm_daily_backend setup
            daily_df = warm_daily_backend(ticker)
        except Exception:
            daily_df = fetch_daily(ticker)

        meta = calculate_indicators(daily_df)
        if meta is None:
            continue

        try:
            intraday_df = warm_intraday_backend(ticker)
        except Exception:
            intraday_df = fetch_intraday(ticker)

        if intraday_df is not None and not intraday_df.empty:
            current_intraday_price = float(intraday_df["Close"].iloc[-1])
        else:
            current_intraday_price = meta["Close"]

        try:
            prev_close = float(daily_df["Close"].iloc[-2])
        except Exception:
            prev_close = float(meta["Close"])

        if current_intraday_price > prev_close:
            price_vs_close = "Above Close"
        elif current_intraday_price < prev_close:
            price_vs_close = "Below Close"
        else:
            price_vs_close = "Equal"

        # Dynamically pulls exchange labels from your core fetch modules
        from utils.data_fetch import get_universe_source
        exchange_source = get_universe_source(ticker)

        rows.append({
            "Ticker": ticker,
            "Universe": exchange_source,  # Re-maps tracking source to fix Page 1 KeyError
            "Close": meta["Close"],
            "ATR%": meta["ATR%"],
            "RVOL": meta["RVOL"],
            "Gap%": meta["Gap%"],
            "Trend": meta["Trend"],
            "Execution": meta["Execution"],
            "PCA1": meta["PCA1"],
            "Avg_Volume_20d": meta["Avg_Volume_20d"],
            "Price_vs_Close": price_vs_close,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Score"] = (
        (df["Trend"] == "UP").astype(int) * 2 +
        (df["Execution"] == "Watch List").astype(int) * 3 +
        df["RVOL"].clip(lower=0).fillna(0) +
        df["PCA1"].clip(lower=0).fillna(0)
    )

    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    print(">>> RANK_UNIVERSE COLUMNS COMPILING COMPLETED:", df.columns.tolist())

    return df

# ---------------------------------------------------------
# FALLBACK UTILITIES TO PREVENT IMPORT COUPLING CRASHES
# ---------------------------------------------------------
# Explicit mapping ensures internal file triggers don't throw NameErrors
def warm_daily_backend(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False, threads=False)
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def warm_intraday_backend(ticker):
    df = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
