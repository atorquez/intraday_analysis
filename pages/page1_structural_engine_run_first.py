# ==============================================================================
# 📈 STRUCTURAL ENGINE PAGE 1 — COMPLETE UNIFIED MASTER CODE (FIXED VERSION)
# ==============================================================================

import streamlit as st

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

import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)
rank_universe = v3.rank_universe

from utils.data_fetch import load_universe, get_universe_source

# ---------------------------------------------------------
# CACHE WARM-UP CORES
# ---------------------------------------------------------
@st.cache_data(ttl=28800)
def warm_daily_backend(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c for c in df.columns]
    return df

@st.cache_data(ttl=300)
def warm_intraday_backend(ticker):
    df = yf.download(ticker, period="1d", interval="1m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c for c in df.columns]
    return df

@st.cache_data(ttl=300)
def get_intraday_5m(ticker):
    df = yf.download(ticker, period="1d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c for c in df.columns]
    return df

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
        if i % 20 == 0 or (i + 1) == total:
            progress.progress((i + 1) / total, text=f"Hydrating Local Caches: [{ticker}] ({i+1}/{total})")
    st.success("High-speed data pre-loading complete! Structural scans will execute smoothly now.")

if st.button("🔥 Preload and Warm Up Local System Memory"):
    warm_up_cache(load_universe())

# ---------------------------------------------------------
# HELPERS
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
            return float(x)

def compute_drop_pct(prev_close, current_price):
    prev_close = to_scalar(prev_close)
    current_price = to_scalar(current_price)
    return (prev_close - current_price) / prev_close if prev_close > 0 else 0

def compute_recovery_probability(df_daily):
    if df_daily.empty or len(df_daily) < 10:
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

    return to_scalar(df["EMA9"].iloc[-1]) > to_scalar(df["EMA20"].iloc[-1])

# ---------------------------------------------------------
# ADDITIONAL STRUCTURAL SIGNALS
# ---------------------------------------------------------
def compute_high_proximity(ticker):
    """
    Returns how close the current price is to the intraday high.
    0.0 means at the high, higher values mean farther from high.
    """
    df = get_intraday_5m(ticker)
    if df.empty or "High" not in df.columns or "Close" not in df.columns:
        return None

    current_price = to_scalar(df["Close"].iloc[-1])
    intraday_high = to_scalar(df["High"].max())

    if intraday_high <= 0:
        return None

    return (intraday_high - current_price) / intraday_high


def compute_bidask_spread(ticker):
    """
    Returns bid–ask spread as % of current price.
    yfinance provides bid/ask only in the fast info API.
    """
    try:
        info = yf.Ticker(ticker).fast_info
        bid = info.get("bid")
        ask = info.get("ask")
        last = info.get("last_price")

        if bid is None or ask is None or last is None:
            return None

        if last <= 0:
            return None

        return (ask - bid) / last

    except Exception:
        return None

# ---------------------------------------------------------
# MARKET REGIME
# ---------------------------------------------------------
def get_index_trend(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty:
            return "Unknown"
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c for c in df.columns]

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
# UI FILTERS
# ---------------------------------------------------------
st.markdown("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Asset Close Gate Price", value=50.0)
max_price = st.number_input("Maximum Asset Close Gate Price", value=500.0)

st.markdown("### 🎛️ Anomaly Multi-Factor Filters")
pca_filter = st.slider("Minimum PCA1 Vector Strength", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)

execution_filter = st.multiselect(
    "Filter by Execution Status",
    ["Watch List", "Not Watch List", "Crossing Soon", "Setup Only"],
    default=["Watch List", "Crossing Soon"]
)

# ---------------------------------------------------------
# RUN MODEL
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model Scan")

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

        progress_bar.progress(0.5, text="Processing Structural Recovery (SRC) Candidates...")

        prime_time = (now_est.hour == 10) or (now_est.hour == 11 and now_est.minute <= 30)
        src_flags = []

        if ranking is not None and not ranking.empty:
            ranking["SP500_Trend"] = sp500_trend
            ranking["NASDAQ_Trend"] = nasdaq_trend
            ranking["Market_Regime"] = regime

            # ---------------------------------------------------------
            # ADD HIGH_PROXIMITY AND BIDASK_SPREAD (CORRECTED)
            # ---------------------------------------------------------
            high_prox_list = []
            spread_list = []

            for idx, row in ranking.iterrows():
                ticker = row["Ticker"]

                high_prox = compute_high_proximity(ticker)
                spread = compute_bidask_spread(ticker)

                high_prox_list.append(high_prox if high_prox is not None else np.nan)
                spread_list.append(spread if spread is not None else np.nan)

            # Assign AFTER loop — correct placement
            ranking["High_Proximity"] = high_prox_list
            ranking["BidAsk_Spread"] = spread_list

            # ---------------------------------------------------------
            # SRC BLOCK
            # ---------------------------------------------------------
            if not prime_time:
                src_flags = [""] * len(ranking)
            else:
                for idx, row in ranking.iterrows():
                    ticker = row["Ticker"]
                    current_price = to_scalar(row["Close"])

                    df_daily = warm_daily_backend(ticker)
                    if df_daily.empty or len(df_daily) < 5:
                        src_flags.append("")
                        continue

                    prev_close = to_scalar(df_daily["Close"].iloc[-2])
                    drop_pct = compute_drop_pct(prev_close, current_price)
                    recovery_prob = compute_recovery_probability(df_daily)
                    ema_cross = ema9_cross_ema20_intraday(ticker)

                    SRC = (
                        drop_pct >= 0.03 and
                        recovery_prob >= 0.60 and
                        ema_cross and
                        regime in ["Bearish", "Choppy"]
                    )
                    src_flags.append("YES" if SRC else "")

        ranking["SRC"] = src_flags

        progress_bar.progress(0.8, text="Applying layout filters...")

        ranking["Universe"] = ranking["Ticker"].apply(get_universe_source)

        filtered = ranking.copy()
        filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]

        if "PCA1" in filtered.columns:
            filtered = filtered[filtered["PCA1"] >= pca_filter]

        if execution_filter and "Execution" in filtered.columns:
            filtered = filtered[filtered["Execution"].isin(execution_filter)]

        st.session_state["intraday_filtered_results"] = ranking
        st.session_state["intraday_visual_results"] = filtered

        st.metric(label="📊 Active Market Regime Identified", value=f"{regime} (SP500: {sp500_trend} | NASDAQ: {nasdaq_trend})")
        st.markdown(f"Structural Tier-1 Premium Universe: {len(ranking)} premium tokens verified.")

        if filtered.empty:
            st.info("No tickers matched your interactive filter constraints.")
        else:
            display_df = filtered.copy()
            display_df["Ticker"] = display_df.apply(lambda r: f"{r['Ticker']} ({r['Universe']})", axis=1)
            display_df = display_df.drop(columns=["Universe"])

            # Format new structural columns
            if "High_Proximity" in display_df.columns:
                display_df["High_Proximity"] = (display_df["High_Proximity"] * 100).round(2)

            if "BidAsk_Spread" in display_df.columns:
                display_df["BidAsk_Spread"] = (display_df["BidAsk_Spread"] * 100).round(3)

            st.subheader(f"🚀 Actionable Structural Matrix Results — {len(display_df)} Tickers")
            st.dataframe(display_df.style.apply(color_execution_column, axis=None), hide_index=True, use_container_width=True)

        progress_bar.empty()
        st.write(f"⏱️ End Time Trace: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"⚡ Total Runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        progress_bar.empty()
        st.error(f"Model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# FALLBACK RENDER
# ---------------------------------------------------------
elif st.session_state.get("intraday_visual_results") is not None:
    filtered_results = st.session_state["intraday_visual_results"]
    structural_results = st.session_state.get("intraday_filtered_results")

    if structural_results is not None and not structural_results.empty:
        reg_val = structural_results["Market_Regime"].iloc[0] if "Market_Regime" in structural_results.columns else "Unknown"
        st.metric(label="📊 Stored Market Regime Profile", value=reg_val)
        st.markdown(f"Structural Tier-1 Premium Universe (Stored Baseline): {len(structural_results)} assets.")

    st.subheader(f"🚀 Stored Run State Results — {len(filtered_results)} Stocks")
    render_df = filtered_results.copy()

    if "Universe" in render_df.columns:
        render_df["Ticker"] = render_df.apply(lambda r: f"{r['Ticker']} ({r['Universe']})", axis=1)
        render_df = render_df.drop(columns=["Universe"])

    st.dataframe(render_df.style.apply(color_execution_column, axis=None), hide_index=True, use_container_width=True)
