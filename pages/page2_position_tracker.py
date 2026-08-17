import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

st.set_page_config(layout="wide", page_title="Position Tracker")
st.caption("Version: 2026-08-01")
st.title("📊 Position Tracker — Live Exhaustion Monitor")


# ============================================================
# DATA FETCH
# ============================================================
def fetch_position_data(ticker):
    try:
        daily = yf.download(ticker, period="3mo", interval="1d", progress=False, threads=False)
        intra = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)

        if daily is None or daily.empty or intra is None or intra.empty:
            return None, None

        daily = daily.dropna(subset=["Close"])
        intra = intra.dropna(subset=["Close"])

        if len(daily) < 2 or len(intra) < 5:
            return None, None

        return daily, intra
    except Exception:
        return None, None


# ============================================================
# METRICS ENGINE
# ============================================================
def compute_position_metrics(ticker):
    daily_df, intra_df = fetch_position_data(ticker)
    if daily_df is None or intra_df is None:
        return None

    close_d = daily_df["Close"].values
    high_d = daily_df["High"].values
    vol_d = daily_df["Volume"].values

    close_intra = intra_df["Close"].values
    high_intra = intra_df["High"].values
    low_intra = intra_df["Low"].values
    vol_intra = intra_df["Volume"].values
    open_intra = intra_df["Open"].values

    current_price = float(close_intra[-1])
    prev_close = float(close_d[-2])

    # Current vs yesterday close
    current_vs_close_pct = ((current_price - prev_close) / prev_close) * 100

    # EMA9 intraday slope (last 10 bars)
    ema9_i = intra_df["Close"].ewm(span=9).mean().values
    if len(ema9_i) >= 10:
        ema9_slope_10 = float((ema9_i[-1] - ema9_i[-10]) / ema9_i[-10]) * 100
    else:
        ema9_slope_10 = 0.0

    # VWAP
    cv_slice = close_intra * vol_intra
    vwap_spot = cv_slice.sum() / vol_intra.sum() if vol_intra.sum() > 0 else current_price
    vwap_dist_pct = (current_price - vwap_spot) / vwap_spot * 100

    # RVOL
    avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
    rvol = float(vol_d[-1] / avg_volume_20d) if avg_volume_20d > 0 else 1.0

    # High proximity (intraday)
    day_high = float(high_intra.max())
    high_proximity_pct = (day_high - current_price) / current_price * 100

    # Velocity penalty (reuse your logic)
    if len(ema9_i) >= 20:
        intraday_velocity_slope = float((ema9_i[-1] - ema9_i[-20]) / ema9_i[-20]) * 100
    else:
        intraday_velocity_slope = ema9_slope_10

    if abs(intraday_velocity_slope) < 0.10:
        velocity_penalty = -1.0
    elif intraday_velocity_slope > 0.20:
        velocity_penalty = +2.0
    elif intraday_velocity_slope < -0.20:
        velocity_penalty = -2.0
    else:
        velocity_penalty = 0.0

    # Exhaustion flag (first version)
    exhaustion = (
        (ema9_slope_10 < 0) or
        (current_vs_close_pct < 0) or
        (rvol < 0.8) or
        (vwap_dist_pct < 0) or
        (high_proximity_pct > 5.0)
    )

    exhaustion_flag = "EXIT" if exhaustion else "OK"

    return {
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "Current_vs_Close%": round(current_vs_close_pct, 2),
        "EMA9_Slope_10": round(ema9_slope_10, 3),
        "VWAP_Dist%": round(vwap_dist_pct, 2),
        "RVOL": round(rvol, 2),
        "High_Proximity%": round(high_proximity_pct, 2),
        "Velocity_Penalty": round(velocity_penalty, 2),
        "Exhaustion_Flag": exhaustion_flag,
    }


# ============================================================
# STREAMLIT UI
# ============================================================
tickers_input = st.text_input(
    "Active Positions (comma-separated tickers)",
    value="",
    help="Example: CCB, AAPL, MSFT"
)

run_tracker = st.button("Run Position Tracker")

if run_tracker and tickers_input.strip():
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    rows = []
    for t in tickers:
        metrics = compute_position_metrics(t)
        if metrics is not None:
            rows.append(metrics)

    if rows:
        df = pd.DataFrame(rows)
        st.subheader(f"🔍 Live Exhaustion Matrix — {len(df)} Positions")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No valid data returned for the provided tickers.")
