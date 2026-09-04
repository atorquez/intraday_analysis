# ==============================================================================
# 📈 PAGE 8 — INSTITUTIONAL EMA ALIGNMENT MODEL
# PURPOSE: Identify institutional-quality tickers showing early EMA alignment
# WITHOUT momentum, continuation, VWAP, proximity, or strict institutional gates.
# ==============================================================================

import streamlit as st

st.set_page_config(layout="wide", page_title="Institutional EMA Alignment Model")
st.caption("Version: 2026-09-04 — Institutional EMA Alignment")
st.title("📈 Institutional EMA Alignment Model")

import importlib
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

# Load institutional universe
try:
    from utils.data_fetch import load_universe, get_universe_source
except (ImportError, ModuleNotFoundError):
    def load_universe(): return []
    def get_universe_source(ticker): return "Premium Slot"

# ---------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------
def _flatten_columns(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if isinstance(df_copy.columns, pd.MultiIndex):
        df_copy.columns = df_copy.columns.get_level_values(0)
    df_copy.columns = [str(c).strip() for c in df_copy.columns]
    return df_copy

@st.cache_data(ttl=120, show_spinner=False)
def fetch_clean_market_batch(tickers_tuple):
    ticker_list = list(tickers_tuple)
    if not ticker_list:
        return pd.DataFrame(), pd.DataFrame()
    try:
        raw_daily = yf.download(
            ticker_list, period="3mo", interval="1d",
            group_by="ticker", progress=False, threads=True
        )
        raw_intra = yf.download(
            ticker_list, period="1d", interval="1m",
            group_by="ticker", progress=False, threads=True
        )
        return raw_daily, raw_intra
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def to_scalar(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, (pd.Series, np.ndarray)):
            val = x.squeeze()
            return float(val)
        return float(x)
    except Exception:
        return float("nan")

# ---------------------------------------------------------
# PAGE 8 — EMA ALIGNMENT ENGINE
# ---------------------------------------------------------
def ema_alignment_engine(tickers, batch_daily, batch_intra, min_price, max_price):
    rows = []
    if batch_daily is None or batch_daily.empty or batch_intra is None or batch_intra.empty:
        return pd.DataFrame()

    available_daily = set(batch_daily.columns.get_level_values(0))
    available_intra = set(batch_intra.columns.get_level_values(0))
    active_pool = list(set(tickers).intersection(available_daily).intersection(available_intra))

    for ticker in active_pool:
        try:
            daily_df = batch_daily[ticker].copy().dropna(subset=["Close"])
            intraday_df = batch_intra[ticker].copy().dropna(subset=["Close"])

            if daily_df.empty or intraday_df.empty or len(daily_df) < 40:
                continue

            # Institutional backbone: liquidity
            vol_d = daily_df["Volume"].values
            avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
            if avg_volume_20d < 250000:
                continue

            # Institutional backbone: price band
            current_price = float(daily_df["Close"].iloc[-1])
            if current_price < min_price or current_price > max_price:
                continue

            # Intraday arrays
            close_i = intraday_df["Close"].values
            if len(close_i) < 5:
                continue

            current_intraday_price = float(close_i[-1])

            # EMA9 / EMA20 intraday
            ema9_i = intraday_df["Close"].ewm(span=9).mean().values
            ema20_i = intraday_df["Close"].ewm(span=20).mean().values

            ema9_slope = float(ema9_i[-1] - ema9_i[-5]) if len(ema9_i) >= 5 else 0.0
            ema20_slope = float(ema20_i[-1] - ema20_i[-5]) if len(ema20_i) >= 5 else 0.0

            # EMA alignment conditions
            cond_price_above_ema9 = current_intraday_price > ema9_i[-1]
            cond_price_above_ema20 = current_intraday_price > ema20_i[-1]
            cond_ema9_slope_pos = ema9_slope > 0

            # Score (Option B: no RVOL, no execution)
            score = 0
            if cond_price_above_ema9: score += 2
            if cond_price_above_ema20: score += 2
            if cond_ema9_slope_pos: score += 1
            if ema20_slope > 0: score += 1

            # Only show tickers with Score ≥ 3 (Option B)
            if score < 3:
                continue

            rows.append({
                "Ticker": ticker,
                "Universe": get_universe_source(ticker),
                "Close": round(current_price, 2),
                "EMA9": round(float(ema9_i[-1]), 2),
                "EMA20": round(float(ema20_i[-1]), 2),
                "EMA9_slope": round(ema9_slope, 4),
                "EMA20_slope": round(ema20_slope, 4),
                "Score": score
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["EMA9_slope", "Score"], ascending=[False, False])

    return df

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.markdown("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Asset Close Gate Price ($)", value=40.0, min_value=40.0, max_value=120.0)
max_price = st.number_input("Maximum Asset Close Gate Price ($)", value=120.0, min_value=40.0, max_value=120.0)

# ---------------------------------------------------------
# RUN MODEL
# ---------------------------------------------------------
run_model = st.button("Run EMA Alignment Scan")

if run_model:
    st.cache_data.clear()
    start_time = time.time()
    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now().astimezone(eastern)

    st.markdown(f"⏱️ Scan Execution Time Stamp: **{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**")

    universe_list = load_universe()
    raw_daily, raw_intra = fetch_clean_market_batch(tuple(universe_list))

    ranking = ema_alignment_engine(universe_list, raw_daily, raw_intra, min_price, max_price)

    if ranking is not None and not ranking.empty:
        st.session_state["ema_alignment_ranking"] = ranking

    st.write(f"⚡ Total Model Runtime: {time.time() - start_time:.2f} seconds")

# ---------------------------------------------------------
# RENDER RESULTS
# ---------------------------------------------------------
if "ema_alignment_ranking" in st.session_state:
    ranking = st.session_state["ema_alignment_ranking"]

    if ranking is not None and not ranking.empty:
        st.subheader(f"🚀 EMA Alignment Results — {len(ranking)} Tickers")
        st.dataframe(ranking, hide_index=True, use_container_width=True)
    else:
        st.info("No tickers matched EMA alignment conditions.")
