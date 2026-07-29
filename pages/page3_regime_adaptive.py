# ==============================================================================
# MASTER REGIME-ADAPTIVE INTRADAY SCANNER (PAGE 3 — FULLY PATCHED)
# ==============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import pytz

from analysis.intraday_ranker_v3 import rank_universe, fetch_intraday
from analysis.intraday_ranker_regime_v3 import (
    automatic_market_regime_detector,
    part1_bullish_scanner,
    part1_bearish_scanner,
    process_regime_intraday_trigger,
    process_bulk_regime_triggers
)

from utils.data_fetch import load_universe

st.set_page_config(layout="wide")
st.caption("SPC Version: 2026-08-08 (Regime Adaptive Patched)")
st.title("🔄 Regime-Adaptive Intraday Model")

# ---------------------------------------------------------
# CACHED WRAPPERS
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_rank_universe(tickers_tuple, buy_zone_percentile=0.15):
    return rank_universe(list(tickers_tuple), buy_zone_percentile)

@st.cache_data(ttl=120, show_spinner=False)
def cached_spy_data(period, interval):
    return yf.download("SPY", period=period, interval=interval, progress=False)

@st.cache_data(ttl=120, show_spinner=False)
def cached_bulk_regime_triggers(watchlist_tuple, mode):
    spy_1min = yf.download("SPY", period="1d", interval="1m", progress=False)
    return process_bulk_regime_triggers(list(watchlist_tuple), spy_1min, mode=mode)

# ---------------------------------------------------------
# HELPER: Market Timestamp
# ---------------------------------------------------------
def get_market_timestamp():
    return datetime.datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S %Z")

# ---------------------------------------------------------
# STEP 1 — MARKET REGIME
# ---------------------------------------------------------
st.subheader("🌐 Global Market Context")

regime_col1, regime_col2 = st.columns([3, 1])
with regime_col2:
    refresh_regime = st.button("🔄 Refresh Regime", key="refresh_regime_btn")

if refresh_regime or "market_regime" not in st.session_state:
    with st.spinner("Checking market regime status using SPY index matrix..."):
        try:
            spy_daily = cached_spy_data("5d", "1d")
            spy_5min = cached_spy_data("1d", "5m")
            market_regime = automatic_market_regime_detector(spy_daily, spy_5min)
            st.session_state["market_regime"] = market_regime
        except Exception as e:
            st.error(f"Failed to fetch market context metrics: {str(e)}")
            market_regime = st.session_state.get("market_regime", "CHOPPY_MARKET")
else:
    market_regime = st.session_state["market_regime"]

if market_regime == "BULLISH_MARKET":
    st.success(f"🟢 **Current Market Regime: {market_regime}** — System is hunting for standard long momentum breakouts.")
elif market_regime == "BEARISH_MARKET":
    st.error(f"🔴 **Current Market Regime: {market_regime}** — System is hunting for short-side drops or relative strength leaders.")
else:
    st.warning(f"🟡 **Current Market Regime: {market_regime}** — High-risk choppy environment. Narrow targets are enforced.")

# ---------------------------------------------------------
# STEP 2 — ALWAYS-SHOW VALIDATION BANNER (NEW)
# ---------------------------------------------------------
base_tickers = load_universe()
page1_results = st.session_state.get("intraday_filtered_results", None)

st.markdown("### 🔎 Universe Validation Summary")

if page1_results is not None and not page1_results.empty:
    structural_count = len(page1_results)
    scan_limit = st.session_state.get("regime_max_tickers", 100)

    st.markdown(
        f"""
**Structural Universe from Page 1:** {structural_count} tickers  
**Regime Scan Limit:** {scan_limit}  
**Tickers Routed to Regime Engine:** {min(structural_count, scan_limit)}  
"""
    )
    source_tickers = page1_results["Ticker"].tolist()
else:
    structural_count = len(base_tickers)
    scan_limit = st.session_state.get("regime_max_tickers", 100)

    st.warning("Page 1 structural universe not found — using fallback raw universe.")

    st.markdown(
        f"""
**Fallback Universe Size:** {structural_count} tickers  
**Regime Scan Limit:** {scan_limit}  
"""
    )
    source_tickers = base_tickers

# ---------------------------------------------------------
# STEP 3 — RUN BUTTON
# ---------------------------------------------------------
run_adaptive_model = st.button("Run Regime-Adaptive Scan", key="regime_run_button")

if run_adaptive_model:
    progress_bar = st.progress(0, text="Initializing scan...")

    try:
        max_tickers = st.session_state.get("regime_max_tickers", 1500)
        test_tickers = source_tickers[:max_tickers]

        progress_bar.progress(0.15, text=f"Running technical analysis on {len(test_tickers)} stocks...")

        calculated_universe = cached_rank_universe(tuple(test_tickers))

        progress_bar.progress(0.5, text="Applying regime filters...")

        if calculated_universe is None or calculated_universe.empty:
            st.warning("The calculated stock technical matrix returned empty. Adjust timeframes or test during market hours.")
            st.stop()

        # ---------------------------------------------------------
        # REGIME ROUTING
        # ---------------------------------------------------------
        watchlist = []
        execution_mode = "LONG"

        if market_regime == "BULLISH_MARKET":
            watchlist = part1_bullish_scanner(calculated_universe)
            execution_mode = "LONG"
            st.info(f"Targeting {len(watchlist)} standard bullish breakouts based on market conditions.")

        elif market_regime == "BEARISH_MARKET":
            watchlist = part1_bearish_scanner(calculated_universe, mode="LONG")
            execution_mode = "LONG"
            st.info(f"Targeting {len(watchlist)} Relative Strength defensive leaders outperforming the index flush.")

        else:
            watchlist = part1_bullish_scanner(calculated_universe)
            execution_mode = "LONG"

        # ---------------------------------------------------------
        # STEP 4 — INTRADAY PCA TRIGGERS
        # ---------------------------------------------------------
        if not watchlist:
            st.info("No underlying assets passed the Part 1 regime selection filters.")
        else:
            st.subheader("⚡ Real-Time Intraday Signals (1-Min Matrix)")

            active_signals = []

            progress_bar.progress(0.6, text=f"Downloading 1-minute data for {len(watchlist)} tickers...")

            triggered_list = cached_bulk_regime_triggers(tuple(watchlist), execution_mode)

            progress_bar.progress(0.9, text="Compiling results...")

            for ticker in triggered_list:
                try:
                    meta = calculated_universe[calculated_universe["Ticker"] == ticker].iloc[0]
                    active_signals.append({
                        "Ticker": ticker,
                        "Company": meta["Company"],
                        "Macro Score": meta["Score"],
                        "Daily Close": meta["Close"],
                        "RVOL": meta["RVOL"],
                        "Gap%": meta["Gap%"],
                        "Zone Status": meta["BuyZone_Heatmap"],
                        "Signal Timestamp": get_market_timestamp()
                    })
                except Exception:
                    continue

            if active_signals:
                st.success(f"🔥 Found {len(active_signals)} verified entry opportunities!")
                df_signals = pd.DataFrame(active_signals)
                st.session_state["regime_active_signals"] = df_signals
                st.dataframe(df_signals, hide_index=True, use_container_width=True)
            else:
                st.info("Watchlist is active, but no stocks have confirmed a 1-minute EMA 9/20 crossover or positive PCA acceleration right now.")

        progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        st.error(f"Model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# STORED SIGNALS
# ---------------------------------------------------------
elif st.session_state.get("regime_active_signals") is not None:
    stored_signals = st.session_state["regime_active_signals"]
    st.subheader(f"⚡ Stored Signals — {len(stored_signals)} Results")
    st.dataframe(stored_signals, hide_index=True, use_container_width=True)
    st.caption(f"Last updated: {get_market_timestamp()}")

# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Page 3 Settings")
    max_tickers_setting = st.number_input(
        "Max Tickers to Process",
        min_value=10,
        max_value=1500,
        value=st.session_state.get("regime_max_tickers", 1500),
        key="regime_max_tickers"
    )
    
