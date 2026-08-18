import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide", page_title="Position Tracker")
st.caption("Version: 2026-08-01")
st.title("📊 Position Tracker — Live Exhaustion Monitor (Intraday + Daily Fallback)")


# ============================================================
# FETCH INTRADAY + DAILY
# ============================================================
def fetch_intraday(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < 5:
            return None
        return df
    except Exception:
        return None


def fetch_daily(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        if len(df) < 5:
            return None
        return df
    except Exception:
        return None


# ============================================================
# INTRADAY METRICS
# ============================================================
def compute_intraday_metrics(ticker, intra_df, daily_df):
    close_i = intra_df["Close"].values
    high_i = intra_df["High"].values
    low_i = intra_df["Low"].values
    vol_i = intra_df["Volume"].values

    close_d = daily_df["Close"].values
    vol_d = daily_df["Volume"].values

    current_price = float(close_i[-1])
    prev_close = float(close_d[-2])

    # Current vs yesterday close
    current_vs_close_pct = ((current_price - prev_close) / prev_close) * 100

    # EMA9 slope (10 bars)
    ema9_i = intra_df["Close"].ewm(span=9).mean().values
    ema9_slope_10 = float((ema9_i[-1] - ema9_i[-10]) / ema9_i[-10]) * 100 if len(ema9_i) >= 10 else 0.0

    # VWAP
    cv_slice = close_i * vol_i
    vwap_spot = cv_slice.sum() / vol_i.sum() if vol_i.sum() > 0 else current_price
    vwap_dist_pct = (current_price - vwap_spot) / vwap_spot * 100

    # RVOL
    avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
    rvol = float(vol_i.sum() / avg_volume_20d) if avg_volume_20d > 0 else 1.0

    # High proximity
    day_high = float(high_i.max())
    high_proximity_pct = (day_high - current_price) / current_price * 100

    # Velocity penalty
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

    # Exhaustion
    exhaustion = (
        (ema9_slope_10 < 0) or
        (current_vs_close_pct < 0) or
        (rvol < 0.8) or
        (vwap_dist_pct < 0) or
        (high_proximity_pct > 5.0)
    )

    return {
        "Mode": "Intraday",
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "Current_vs_Close%": round(current_vs_close_pct, 2),
        "EMA9_Slope_10": round(ema9_slope_10, 3),
        "VWAP_Dist%": round(vwap_dist_pct, 2),
        "RVOL": round(rvol, 2),
        "High_Proximity%": round(high_proximity_pct, 2),
        "Velocity_Penalty": round(velocity_penalty, 2),
        "Exhaustion_Flag": "EXIT" if exhaustion else "OK",
    }


# ============================================================
# DAILY FALLBACK METRICS
# ============================================================
def compute_daily_metrics(ticker, daily_df):

    if daily_df is None or daily_df.empty:
        return {
            "Mode": "No Daily Data",
            "Ticker": ticker,
            "Current_Price": None,
            "Current_vs_Close%": None,
            "EMA9_Slope_10": None,
            "VWAP_Dist%": None,
            "RVOL": None,
            "High_Proximity%": None,
            "Velocity_Penalty": None,
            "Exhaustion_Flag": "N/A",
        }

    close_d = daily_df["Close"].values
    high_d = daily_df["High"].values
    vol_d = daily_df["Volume"].values

    current_price = float(close_d[-1])
    prev_close = float(close_d[-2])

    current_vs_close_pct = ((current_price - prev_close) / prev_close) * 100

    ema9_d = daily_df["Close"].ewm(span=9).mean().values
    ema9_slope_3d = float((ema9_d[-1] - ema9_d[-4]) / ema9_d[-4]) * 100 if len(ema9_d) >= 4 else 0.0

    cv_slice = close_d[-20:] * vol_d[-20:]
    vwap_proxy = cv_slice.sum() / vol_d[-20:].sum() if vol_d[-20:].sum() > 0 else current_price
    vwap_dist_pct = (current_price - vwap_proxy) / vwap_proxy * 100

    avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
    rvol = float(vol_d[-1] / avg_volume_20d) if avg_volume_20d > 0 else 1.0

    day_high = float(high_d[-1])
    high_proximity_pct = (day_high - current_price) / current_price * 100

    exhaustion = (
        (ema9_slope_3d < 0) or
        (current_vs_close_pct < 0) or
        (rvol < 0.8) or
        (vwap_dist_pct < 0) or
        (high_proximity_pct > 5.0)
    )

    return {
        "Mode": "Daily Fallback",
        "Ticker": ticker,
        "Current_Price": round(current_price, 2),
        "Current_vs_Close%": round(current_vs_close_pct, 2),
        "EMA9_Slope_10": round(ema9_slope_3d, 3),
        "VWAP_Dist%": round(vwap_dist_pct, 2),
        "RVOL": round(rvol, 2),
        "High_Proximity%": round(high_proximity_pct, 2),
        "Velocity_Penalty": 0.0,
        "Exhaustion_Flag": "EXIT" if exhaustion else "OK",
    }


# ============================================================
# STREAMLIT UI
# ============================================================
tickers_input = st.text_input(
    "Active Positions (comma-separated tickers)",
    value="",
    help="Example: AAPL, MSFT, SEI"
)

run_tracker = st.button("Run Position Tracker")

if run_tracker and tickers_input.strip():
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    rows = []

    for t in tickers:
        daily_df = fetch_daily(t)
        intra_df = fetch_intraday(t)

        if intra_df is not None:
            st.success(f"{t}: Intraday Mode Active")
            metrics = compute_intraday_metrics(t, intra_df, daily_df)
        else:
            st.warning(f"{t}: Intraday Missing → Daily Fallback Mode")
            metrics = compute_daily_metrics(t, daily_df)

        rows.append(metrics)

    df = pd.DataFrame(rows)
    st.subheader(f"🔍 Exhaustion Matrix — {len(df)} Positions")
    st.dataframe(df, use_container_width=True)
