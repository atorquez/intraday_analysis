# ==============================================================================
# PART 1 -SPC Intraday Ranker v3
# ==============================================================================
# ==============================================================================
# SPC Intraday Ranker v3.0 — HIGH SPEED RE-LOG ENGINE
# ==============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# CRITICAL SPEED PATCH: Completely suppress console logging output 
# to stop console text-rendering from bottlenecking your CPU processing time
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

print(">>> intraday_ranker_v3 PERFORMANCE ENGINES LOADED <<<")

# ---------------------------------------------------------
# SAFE COLUMN FLATTENER (Upgraded for MultiIndex/Single Ticker safety)
# ---------------------------------------------------------
def _safe_flatten_columns(df):
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    if isinstance(df_copy.columns, pd.MultiIndex):
        df_copy.columns = [col if isinstance(col, tuple) else col for col in df_copy.columns]
    
    df_copy.columns = [str(c) for c in df_copy.columns]
    return df_copy

# ---------------------------------------------------------
# SAFE COLUMN FLATTENER (Upgraded for MultiIndex/Single Ticker safety)
# ---------------------------------------------------------
def _safe_flatten_columns(df):
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    # Handle modern yfinance multi-index column templates cleanly
    if isinstance(df_copy.columns, pd.MultiIndex):
        if len(df_copy.columns.levels) > 1:
            # Drop the ticker level if it leaked into the columns
            df_copy.columns = df_copy.columns.get_level_values(0)
        else:
            df_copy.columns = [col[0] if isinstance(col, tuple) else col for col in df_copy.columns]
            
    df_copy.columns = [str(c).strip() for c in df_copy.columns]
    return df_copy

# ---------------------------------------------------------
# DAILY FETCH
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
# INTRADAY FETCH (1m)
# ---------------------------------------------------------
def fetch_intraday(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = _safe_flatten_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df
    except Exception as e:
        logger.warning(f"Intraday fetch error for {ticker}: {e}")
        return None

# ---------------------------------------------------------
# EXTRA INDICATORS (RSI, BB, ROC, StochK, EMA_Curve, VWAP)
# ---------------------------------------------------------
def compute_extra_indicators(df, is_intraday=True):
    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame()
        
    df = df.copy()

    # Explicitly squeeze to 1D Series to prevent MultiIndex division NaN errors
    close_s = df["Close"].squeeze()
    high_s = df["High"].squeeze()
    low_s = df["Low"].squeeze()
    vol_s = df["Volume"].squeeze()

    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    df["SMA20"] = close_s.rolling(20, min_periods=1).mean()
    df["STD20"] = close_s.rolling(20, min_periods=1).std().fillna(0)
    df["BB_Width"] = np.where(
        df["SMA20"] != 0,
        (df["STD20"] * 2) / df["SMA20"],
        0.0
    )

    df["ROC"] = close_s.pct_change(10).fillna(0)

    low_min = low_s.rolling(14, min_periods=1).min()
    high_max = high_s.rolling(14, min_periods=1).max()
    stoch_range = high_max - low_min
    df["StochK"] = np.where(stoch_range != 0, (close_s - low_min) / stoch_range, 0.5)
    df["StochK"] = df["StochK"].clip(0, 1)

    df["EMA_Curve"] = close_s.ewm(span=9).mean() - close_s.ewm(span=20).mean()
    df["VolDelta"] = vol_s.diff().fillna(0)

    if is_intraday:
        cv = vol_s * close_s
        df["VWAP"] = cv.cumsum() / vol_s.cumsum()
    else:
        df["VWAP"] = (close_s * vol_s).rolling(20, min_periods=1).sum() / \
                     vol_s.rolling(20, min_periods=1).sum()

    df["VWAP_Dist"] = close_s - df["VWAP"]
    return df

# ---------------------------------------------------------
# PCA ENGINE
# ---------------------------------------------------------
def append_pca_components(df, is_intraday=True):
    processed = compute_extra_indicators(df, is_intraday=is_intraday)
    if processed.empty:
        df["PCA1"] = np.nan
        df["PCA2"] = np.nan
        df["PCA3"] = np.nan
        return df

    feature_cols = ["RSI", "BB_Width", "ROC", "StochK", "EMA_Curve", "VolDelta", "VWAP_Dist"]
    pca_features = processed[feature_cols].copy()
    pca_features = pca_features.fillna(0)

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
        
        # Consistent Sign Flipping Fix: Ensure deterministic orientation
        for i in range(components.shape[1]):
            if components[0, i] < 0:
                components[:, i] = -components[:, i]

        pca_df = pd.DataFrame(components, index=pca_features.index, columns=["PCA1", "PCA2", "PCA3"])
        processed = processed.drop(columns=["PCA1", "PCA2", "PCA3"], errors="ignore")
        processed = processed.join(pca_df, how="left")
    except Exception as e:
        logger.warning(f"PCA error: {e}")
        processed["PCA1"] = np.nan
        processed["PCA2"] = np.nan
        processed["PCA3"] = np.nan

    return processed

# ---------------------------------------------------------
# DAILY INDICATORS + EXECUTION (WITH INTRADAY REUSE)
# ---------------------------------------------------------
def calculate_indicators(daily_df, ticker, intraday_df=None):
    if daily_df is None or daily_df.empty or len(daily_df) < 40:
        return None

    # Safe scalar extraction to protect against MultiIndex or Series types
    try:
        last_close = float(daily_df["Close"].iloc[-1].squeeze())
    except AttributeError:
        last_close = float(daily_df["Close"].iloc[-1])

    # Enforce explicit baseline boundaries defined by Section 1 of the documentation
    if last_close < 40.0 or last_close > 110.0:
        return None

    df = append_pca_components(daily_df.copy(), is_intraday=False)

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

    # DAILY TREND
    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        trend = "UP"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        trend = "DOWN"
    else:
        trend = "FLAT"

    # DAILY SLOPES
    ema9_slope = df["EMA9"].iloc[-1] - df["EMA9"].iloc[-5]
    ema20_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]
    pca1 = last["PCA1"] if not pd.isna(last["PCA1"]) else None

    ema20_val = last["EMA20"] if last["EMA20"] != 0 else 1.0
    proximity_metric = abs((last["EMA9"] - last["EMA20"]) / ema20_val)

    # ---------------------------------------------------------
    # INTRADAY OVERRIDE (SOFT — OPTIMIZED REUSE)
    # ---------------------------------------------------------
    intraday_override_watch = False
    intraday_override_cross = False

    if intraday_df is None:
        intraday_df = fetch_intraday(ticker)

    if intraday_df is not None and not intraday_df.empty:
        intraday_processed = append_pca_components(intraday_df.copy(), is_intraday=True)

        if len(intraday_processed) > 5:
            pca1_slope_intraday = float(
                intraday_processed["PCA1"].iloc[-1] - intraday_processed["PCA1"].iloc[-5]
            )
        else:
            pca1_slope_intraday = 0.0

        last_intraday = intraday_processed.iloc[-1]
        ema_curve_intraday = float(last_intraday.get("EMA_Curve", 0.0))
        vwap_dist_intraday = float(last_intraday.get("VWAP_Dist", 0.0))

        if ema_curve_intraday > 0 and pca1_slope_intraday > 0:
            intraday_override_watch = True

        if abs(ema_curve_intraday) < 0.02 and abs(vwap_dist_intraday) < 0.15:
            intraday_override_cross = True

    # ---------------------------------------------------------
    # FINAL EXECUTION DECISION
    # ---------------------------------------------------------
    if intraday_override_watch:
        execution = "Watch List"
    elif intraday_override_cross:
        execution = "Crossing Soon"
    else:
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
        "Close": round(float(last_close), 2),
        "ATR%": round(float(last["ATR%"]), 2) if not pd.isna(last["ATR%"]) else 0.0,
        "RVOL": round(float(last["RVOL"]), 2) if not pd.isna(last["RVOL"]) else 0.0,
        "Gap%": round(float(last["Gap%"]), 2) if not pd.isna(last["Gap%"]) else 0.0,
        "Trend": trend,
        "Execution": execution,
        "PCA1": float(pca1) if pca1 is not None else np.nan,"Avg_Volume_20d": float(df["Volume"].rolling(20).mean().iloc[-1]),}

# ---------------------------------------------------------
# PART 2 - RANK UNIVERSE (BATCH PARADIGM UPGRADE — REAL-TIME FIXED)
# ---------------------------------------------------------
def rank_universe_batch(tickers, batch_daily, batch_intra, buy_zone_percentile=0.15):
    """
    Finalized high-speed vectorized engine. Pre-filters structural ticker 
    intersections up front to deliver sub-15 second model runtimes.
    Incorporates dynamic real-time un-clipped momentum sorting vectors.
    """
    print(">>> rank_universe_batch() PRODUCTION BLITZ EXECUTED <<<")
    rows = []

    # 1. Safely flip multi-index levels en masse (Ticker to Level 0)
    try:
        if isinstance(batch_daily.columns, pd.MultiIndex):
            if 'Close' in batch_daily.columns.get_level_values(0):
                batch_daily = batch_daily.swaplevel(0, 1, axis=1).sort_index(axis=1)
    except Exception:
        pass

    try:
        if isinstance(batch_intra.columns, pd.MultiIndex):
            if 'Close' in batch_intra.columns.get_level_values(0):
                batch_intra = batch_intra.swaplevel(0, 1, axis=1).sort_index(axis=1)
    except Exception:
        pass

    # Extract clean sets of available tickers sitting on level 0
    available_daily = set(batch_daily.columns.get_level_values(0)) if hasattr(batch_daily, "columns") else set()
    available_intra = set(batch_intra.columns.get_level_values(0)) if hasattr(batch_intra, "columns") else set()

    #----------------------------------------------------------------------
    # ⚡ CRITICAL OPTIMIZATION: CRUSH THE LOOP OVERHEAD EN MASSE
    #----------------------------------------------------------------------
    # Take the exact mathematical intersection of tickers that have active data frames
    active_pool = set(tickers).intersection(available_daily).intersection(available_intra)
    
    print(f">>> Funneling Optimization: Processing {len(active_pool)} active assets out of {len(tickers)} requested.")

    # Loop ONLY through verified active tokens to eliminate processing latency
    for ticker in active_pool:
        daily_df = pd.DataFrame()
        intraday_df = pd.DataFrame()
        
        try:
            daily_df = batch_daily[ticker].copy().dropna(subset=["Close"])
        except Exception:
            continue
                
        try:
            intraday_df = batch_intra[ticker].copy().dropna(subset=["Close"])
        except Exception:
            continue

        if daily_df.empty or intraday_df.empty or len(daily_df) < 40:
            continue

        meta = calculate_indicators(daily_df, ticker, intraday_df=intraday_df)
        if meta is None:
            continue

        pca1_slope = 0.0
        ema_curve = 0.0
        vwap_dist = 0.0
        roc_10 = 0.0
        stoch_k = 0.5
        current_intraday_price = meta["Close"]

        intraday_processed = append_pca_components(intraday_df.copy(), is_intraday=True)
        if not intraday_processed.empty:
            last_intraday = intraday_processed.iloc[-1]
            
            if len(intraday_processed) > 5:
                try:
                    pca1_slope = float(intraday_processed["PCA1"].iloc[-1] - intraday_processed["PCA1"].iloc[-5])
                except Exception:
                    pca1_slope = 0.0

            try:
                ema_curve = float(last_intraday.get("EMA_Curve", 0.0))
                vwap_dist = float(last_intraday.get("VWAP_Dist", 0.0))
                roc_10 = float(last_intraday.get("ROC", 0.0))
                stoch_k = float(last_intraday.get("StochK", 0.5))
            except Exception:
                pass

            try:
                current_intraday_price = float(intraday_df["Close"].iloc[-1].squeeze())
            except AttributeError:
                current_intraday_price = float(intraday_df["Close"].iloc[-1])

        try:
            prev_close = float(daily_df["Close"].iloc[-2].squeeze())
        except Exception:
            prev_close = float(meta["Close"])

        if current_intraday_price > prev_close:
            price_vs_close = "Above Close"
        elif current_intraday_price < prev_close:
            price_vs_close = "Below Close"
        else:
            price_vs_close = "Equal"

        rows.append({
            "Ticker": ticker,
            "Universe": "Premium Slot",
            "Close": meta["Close"],
            "ATR%": meta["ATR%"],
            "RVOL": meta["RVOL"],
            "Gap%": meta["Gap%"],
            "Trend": meta["Trend"],
            "Execution": meta["Execution"],
            "PCA1": meta["PCA1"],
            "Avg_Volume_20d": meta["Avg_Volume_20d"],
            "Price_vs_Close": price_vs_close,
            "PCA1_slope": pca1_slope,
            "EMA_Curve": ema_curve,
            "VWAP_Dist": vwap_dist,
            "ROC_10": roc_10,
            "StochK": stoch_k,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
    # Fill any missing ML values with flat zeros safely to protect operations from breaking
    df["PCA1"] = pd.to_numeric(df["PCA1"], errors="coerce").fillna(0.0)
    df["PCA1_slope"] = pd.to_numeric(df["PCA1_slope"], errors="coerce").fillna(0.0)
    df["RVOL"] = pd.to_numeric(df["RVOL"], errors="coerce").fillna(0.0)

    # ==============================================================================
    # REAL-TIME INTRADAY BREAKOUT MOMENTUM SCORE FORMULA
    # ==============================================================================
    # 1. Removed the .clip(lower=0) on PCA1 so negative momentum lowers rank
    # 2. Included PCA1_slope multiplied by a velocity scaling variable to reward acceleration
    df["Score"] = (
        (df["Trend"] == "UP").astype(int) * 2.0 +
        (df["Execution"] == "Watch List").astype(int) * 3.0 +
        df["RVOL"].clip(lower=0) +
        df["PCA1"] +                       # Allows pullbacks or distribution prints to drop score
        (df["PCA1_slope"] * 2.5)           # Highlights relative strength velocity
    )

    return df.sort_values("Score", ascending=False).reset_index(drop=True)
