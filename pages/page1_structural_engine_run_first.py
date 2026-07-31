# ==============================================================================
# 📈 STRUCTURAL ENGINE PAGE 1 — COMPLETE UNIFIED MASTER CODE
# SPECIFICATION: PREMIUM UNIVERSE STRUCTURAL TRACKING SYSTEM (>$50 TICKERS)
# INCLUDES: HIGH-SPEED PROXIMITY, SPREADS, AND adaptive INDEX REGIME PLUGINS
# ==============================================================================

import streamlit as st

# Secure Layout Initialization Layer (Must be the absolute first execution point)
st.set_page_config(layout="wide", page_title="Structural Engine Page1")
st.caption("Version: 2026-07-23")
st.title("📈 Structural Engine Page 1")

import importlib
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# Secure Dynamic Reloading Components from backend file
import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)
rank_universe = v3.rank_universe

from utils.data_fetch import load_universe, get_universe_source

# ---------------------------------------------------------
# CORRELATED BACKEND-ALIGNED PRELOAD CORES
# ---------------------------------------------------------
def _flatten_columns(df):
    """Safely flatten MultiIndex columns from yfinance downloads."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=28800)
def warm_daily_backend(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    return _flatten_columns(df)

@st.cache_data(ttl=300)
def warm_intraday_backend(ticker):
    df = yf.download(ticker, period="1d", interval="1m", progress=False)
    return _flatten_columns(df)

@st.cache_data(ttl=300)
def get_intraday_5m(ticker):
    df = yf.download(ticker, period="1d", interval="5m", progress=False)
    return _flatten_columns(df)

def warm_up_cache(tickers):
    progress = st.progress(0.0, text="Initializing high-speed local data cache preloader...")
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            warm_daily_backend(ticker)
            warm_intraday_backend(ticker)
            get_intraday_5m(ticker)
        except Exception:
            pass
        # Rate-limit protection: brief pause every 20 tickers
        if i > 0 and i % 20 == 0:
            time.sleep(0.5)
        if i % 20 == 0 or (i + 1) == total:
            progress.progress((i + 1) / total, text=f"Hydrating Local Caches: [{ticker}] ({i+1}/{total})")
    st.success("High-speed data pre-loading complete! Structural scans will execute smoothly now.")

if st.button("🔥 Preload and Warm Up Local System Memory"):
    warm_up_cache(load_universe())

# ---------------------------------------------------------
# SCALAR EXTRACTION & MATHEMATICAL HELPERS
# ---------------------------------------------------------
def to_scalar(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, (pd.Series, np.ndarray)):
            return float(x[-1])
        return float(x)
    except Exception:
        try:
            return float(x.iloc[-1])
        except Exception:
            return float("nan")

def compute_drop_pct(prev_close, current_price):
    prev_close = to_scalar(prev_close)
    current_price = to_scalar(current_price)
    return (prev_close - current_price) / prev_close if prev_close > 0 else 0

def compute_recovery_probability(df_daily):
    if df_daily.empty or len(df_daily) < 30:
        return 0.0
    
    df_window = df_daily.tail(30)
    recoveries = 0
    total = len(df_window)

    for i in range(total):
        low = to_scalar(df_window["Low"].iloc[i])
        close = to_scalar(df_window["Close"].iloc[i])
        high = to_scalar(df_window["High"].iloc[i])

        dip = high - low
        recovered = close - low

        if dip > 0 and recovered >= 0.5 * dip:
            recoveries += 1

    return recoveries / total

def ema9_cross_ema20_intraday(ticker):
    df = get_intraday_5m(ticker)
    if df.empty or len(df) < 20:
        return False

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    ema9 = to_scalar(df["EMA9"].iloc[-1])
    ema20 = to_scalar(df["EMA20"].iloc[-1])
    return ema9 > ema20

# ---------------------------------------------------------
# FIXED HIGH-PROXIMITY & SPREAD COMPUTATION MECHANICS
# ---------------------------------------------------------
def compute_high_proximity(ticker):
    """Calculates how close the current price is to the absolute daily high."""
    try:
        df = warm_intraday_backend(ticker)
        if df.empty:
            return None
            
        daily_high = float(df["High"].max())
        current_price = float(df["Close"].iloc[-1])
        
        if current_price <= 0:
            return 0.0
        return (daily_high - current_price) / current_price
    except Exception:
        return None

def compute_bidask_spread(ticker):
    """Generates a stable institutional spread proxy relative to the asset close price."""
    try:
        df = warm_intraday_backend(ticker)
        if df.empty:
            return None
            
        current_price = float(df["Close"].iloc[-1])
        if current_price <= 0:
            return 0.0
            
        # Calculate trailing 5-minute High-to-Low minute range window spread
        recent_minute_spread = (df["High"] - df["Low"]).tail(5).mean()
        # Scale down standard range metrics to match an ultra-tight institutional liquid bid-ask profile
        implied_spread = recent_minute_spread * 0.15 
        
        return implied_spread / current_price
    except Exception:
        return None

# ---------------------------------------------------------
# MARKET REGIME VECTOR CLASSIFIERS (MULTIINDEX PATCHED)
# ---------------------------------------------------------
def get_index_trend(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty:
            return "Unknown"
            
        df = _flatten_columns(df)

        df["EMA9"] = df["Close"].ewm(span=9).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()

        ema9 = float(df["EMA9"].iloc[-1])
        ema20 = float(df["EMA20"].iloc[-1])
        ema50 = float(df["EMA50"].iloc[-1])
        slope20 = float(df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5])

        if ema9 > ema20 > ema50 and slope20 > 0:
            return "Bullish"
        if ema9 < ema20 < ema50 and slope20 < 0:
            return "Bearish"
        return "Choppy"
    except Exception:
        return "Unknown"

def classify_regime(sp500_trend, nasdaq_trend):
    if sp500_trend == "Bullish" and nasdaq_trend == "Bullish":
        return "Trending"
    if sp500_trend == "Bearish" and nasdaq_trend == "Bearish":
        return "Bearish"
    if sp500_trend != nasdaq_trend:
        return "Mixed"
    return "Choppy"

@st.cache_data(ttl=300, show_spinner=False)
def cached_rank_universe(tickers_tuple, buy_zone_percentile=0.15):
    return rank_universe(list(tickers_tuple), buy_zone_percentile)

def color_execution_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    if "Execution" in df.columns:
        style_df["Execution"] = [
            "background-color: #4CAF50; color: white;" if v == "Watch List"
            else "background-color: #FFC107; color: black;" if v == "Crossing Soon"
            else "background-color: #FF9800; color: white;" if v == "Not Watch List"
            else "background-color: #9E9E9E; color: white;"
            for v in df["Execution"]
        ]
    return style_df

# ---------------------------------------------------------
# INTERACTIVE USER INTERFACE FILTERS PANEL
# ---------------------------------------------------------
st.markdown("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Asset Close Gate Price", value=50.0, key="intraday_min_price")
max_price = st.number_input("Maximum Asset Close Gate Price", value=100, key="intraday_max_price")

st.markdown("### 🎛️ Anomaly Multi-Factor Filters")
pca_filter = st.slider("Minimum PCA1 Vector Strength (Institutional Filter)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

execution_filter = st.multiselect(
    "Filter by Real-Time Structural Execution Status",
    ["Watch List", "Not Watch List", "Crossing Soon", "Setup Only"],
    default=["Watch List", "Crossing Soon"],
    key="intraday_execution_filter"
)

# ---------------------------------------------------------
# RESULT DISPLAY HELPER
# ---------------------------------------------------------
def render_results(filtered, ranking, regime, sp500_trend, nasdaq_trend):
    st.metric(label="📊 Active Market Regime Identified", value=f"{regime} (SP500: {sp500_trend} | NASDAQ: {nasdaq_trend})")
    st.markdown(f"Structural Tier-1 Premium Universe: {len(ranking)} premium tokens verified.")

    if filtered.empty:
        st.info("No tickers matched your interactive filter constraints.")
    else:
        display_df = filtered.copy()
        display_df["Ticker"] = display_df.apply(lambda r: f"{r['Ticker']} ({r['Universe']})", axis=1)
        display_df = display_df.drop(columns=["Universe"], errors="ignore")

        # Standardize format scaling metrics safely
        if "High_Proximity" in display_df.columns:
            display_df["High_Proximity"] = (display_df["High_Proximity"] * 100).round(2)

        if "BidAsk_Spread" in display_df.columns:
            display_df["BidAsk_Spread"] = (display_df["BidAsk_Spread"] * 100).round(3)

        st.subheader(f"🚀 Actionable Structural Matrix Results — {len(display_df)} Tickers")
        st.dataframe(display_df.style.apply(color_execution_column, axis=None), hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# RUN INTER-SESSION MODEL ENGINE — PARALLELIZED & COMPILED
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model Scan", key="intraday_run_button")

if run_model:
    try:
        start_time = time.time()
        eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now().astimezone(eastern)

        st.markdown(f"⏱️ Scan Execution Time Stamp: **{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**")

        progress_bar = st.progress(0, text="Loading baseline configuration maps...")

        base_universe = load_universe()
        if not base_universe:
            st.error("The source stock universe list returned empty.")
            st.stop()

        progress_bar.progress(0.1, text=f"Scanning {len(base_universe)} tickers...")
        ranking = cached_rank_universe(tuple(base_universe))

        progress_bar.progress(0.4, text="Calculating index trend environment matrix...")
        sp500_trend = get_index_trend("^GSPC")
        nasdaq_trend = get_index_trend("^IXIC")
        regime = classify_regime(sp500_trend, nasdaq_trend)

        # ==============================================================================
        # PARALLELIZED PER-TICKER PROCESSING ENGINE
        # ==============================================================================
        progress_bar.progress(0.5, text="Processing High Proximity, Spreads, and SRC Candidates...")

        prime_time = (now_est.hour == 10) or (now_est.hour == 11 and now_est.minute <= 30)
        
        high_prox_list = [np.nan] * len(ranking)
        spread_list = [np.nan] * len(ranking)
        src_flags = [""] * len(ranking)
        
        if ranking is not None and not ranking.empty:
            ranking["SP500_Trend"] = sp500_trend
            ranking["NASDAQ_Trend"] = nasdaq_trend
            ranking["Market_Regime"] = regime
            
            # Worker: isolated per-ticker logic so threads don't interfere
            def _process_single_ticker(args):
                i, row, prime_time_flag, regime_flag = args
                ticker = row["Ticker"]
                current_price = to_scalar(row["Close"])

                # Proximity & Spread (2nd call hits the in-thread cache)
                high_prox = compute_high_proximity(ticker)
                spread = compute_bidask_spread(ticker)

                src_flag = ""
                if prime_time_flag:
                    df_daily = warm_daily_backend(ticker)
                    if not df_daily.empty and len(df_daily) >= 5:
                        df_daily = _flatten_columns(df_daily)
                        required_cols = {"Close", "High", "Low"}
                        if required_cols.issubset(df_daily.columns):
                            prev_close = to_scalar(df_daily["Close"].iloc[-2])
                            drop_pct = compute_drop_pct(prev_close, current_price)
                            recovery_prob = compute_recovery_probability(df_daily)
                            ema_cross = ema9_cross_ema20_intraday(ticker)

                            SRC = (drop_pct >= 0.03 and recovery_prob >= 0.60 and ema_cross and regime_flag in ["Bearish", "Choppy"])
                            src_flag = "YES" if SRC else ""

                return i, high_prox, spread, src_flag

            # Build task list with stable integer indices
            tasks = [(i, row, prime_time, regime) for i, (_, row) in enumerate(ranking.iterrows())]
            
            completed = 0
            total = len(tasks)
            max_workers = 6  # Sweet spot for yfinance concurrency without rate limits
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_single_ticker, task): task[0] for task in tasks}
                
                for future in as_completed(futures):
                    try:
                        i, high_prox, spread, src_flag = future.result()
                        high_prox_list[i] = high_prox if high_prox is not None else np.nan
                        spread_list[i] = spread if spread is not None else np.nan
                        src_flags[i] = src_flag
                    except Exception:
                        # Isolate failures so one bad ticker doesn't crash the batch
                        pass
                    
                    completed += 1
                    if completed % 20 == 0 or completed == total:
                        progress_bar.progress(
                            0.5 + 0.3 * (completed / total),
                            text=f"Processing candidates... ({completed}/{total})"
                        )
                
            ranking["High_Proximity"] = high_prox_list
            ranking["BidAsk_Spread"] = spread_list
            ranking["SRC"] = src_flags
        else:
            ranking = pd.DataFrame()

        # ==============================================================================
        # FIXED INDENTATION LAYER (Pushed outside the else block so it runs every time)
        # ==============================================================================
        progress_bar.progress(0.8, text="Applying layout filters...")
        
        if ranking.empty:
            st.warning("No stock configurations satisfied the technical requirements.")
        else:
            ranking["Universe"] = ranking["Ticker"].apply(get_universe_source)

            filtered = ranking.copy()
            filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]

            if "PCA1" in filtered.columns:
                filtered = filtered[filtered["PCA1"] >= pca_filter]

            if execution_filter and "Execution" in filtered.columns:
                filtered = filtered[filtered["Execution"].isin(execution_filter)]

            # Save clean states to Session Memory
            st.session_state["intraday_filtered_results"] = ranking
            st.session_state["intraday_visual_results"] = filtered
            st.session_state["intraday_regime"] = regime
            st.session_state["intraday_sp500"] = sp500_trend
            st.session_state["intraday_nasdaq"] = nasdaq_trend

            render_results(filtered, ranking, regime, sp500_trend, nasdaq_trend)

        progress_bar.empty()
        st.write(f"⏱️ End Time Trace: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"⚡ Total Runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        try:
            progress_bar.empty()
        except NameError:
            pass
        st.error(f"Model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# SESSION STATE HYDRATION — Display cached results on rerun
# ---------------------------------------------------------
elif "intraday_visual_results" in st.session_state:
    regime = st.session_state.get("intraday_regime", "Unknown")
    sp500_trend = st.session_state.get("intraday_sp500", "Unknown")
    nasdaq_trend = st.session_state.get("intraday_nasdaq", "Unknown")
    ranking = st.session_state["intraday_filtered_results"]
    filtered = st.session_state["intraday_visual_results"]
    render_results(filtered, ranking, regime, sp500_trend, nasdaq_trend)