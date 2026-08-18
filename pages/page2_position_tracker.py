# ======================================================================
# 📈 PAGE 2 — POSITION LIFECYCLE TRACKER (NO CHARTS VERSION)
# ======================================================================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(layout="wide", page_title="Position Lifecycle Tracker")
st.title("📈 Position Lifecycle Tracker — KEEP or EXIT")

# ---------------------------------------------------------
# INPUT — Ticker you are holding
# ---------------------------------------------------------
ticker = st.text_input("Enter the ticker you are holding:", value="")

run_button = st.button("Run Lifecycle Tracker")

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def compute_percentile(close_arr):
    low = np.min(close_arr)
    high = np.max(close_arr)
    current = close_arr[-1]
    if high == low:
        return 50.0
    return ((current - low) / (high - low)) * 100

def compute_slope(series, window=10):
    if len(series) < window + 1:
        return 0.0
    return ((series[-1] - series[-window]) / series[-window]) * 100

# ---------------------------------------------------------
# RUN (wrapped in a function so st.stop() works)
# ---------------------------------------------------------
def run_tracker():
    # Fetch intraday data FIRST
    df = yf.download(
        ticker.strip().upper(),
        period="1d",
        interval="1m",
        progress=False
    )

    # Intraday availability check
    required_cols = {"Open", "High", "Low", "Close", "Volume"}

    if df.empty or not required_cols.issubset(df.columns):
        st.warning("⚠️ No intraday data available. This tracker only works during regular market hours (09:30–16:00 EST).")
        return


    # SAFE: From here on, df['Close'] exists
    df = df.dropna(subset=["Close"])
    close_arr = df["Close"].values
    high_arr = df["High"].values
    low_arr = df["Low"].values
    vol_arr = df["Volume"].values

    # VWAP
    vwap = (vol_arr * close_arr).sum() / vol_arr.sum()

    # EMA9 and EMA20
    ema9 = df["Close"].ewm(span=9).mean().values
    ema20 = df["Close"].ewm(span=20).mean().values

    # Slopes
    ema9_slope = compute_slope(ema9, 10)
    ema20_slope = compute_slope(ema20, 10)
    trend_slope = compute_slope(close_arr, 10)

    # Percentile
    percentile = compute_percentile(close_arr)

    # Breakout levels
    premarket_high = df.head(10)["High"].max()
    first15_high = df.head(15)["High"].max()

    # Current price
    current_price = close_arr[-1]

    # Daily data for yesterday close
    daily = yf.download(ticker.strip().upper(), period="5d", interval="1d", progress=False)
    yesterday_close = daily["Close"].iloc[-2] if len(daily) >= 2 else current_price

    # BuyZone
    buyzone = "IN" if current_price > ema20[-1] else "OUT"

    # Decision logic
    exit_reasons = []

    if current_price < vwap:
        exit_reasons.append("VWAP lost")
    if ema9_slope < 0:
        exit_reasons.append("EMA9 slope negative")
    if ema20_slope < 0:
        exit_reasons.append("EMA20 slope negative")
    if trend_slope < 0:
        exit_reasons.append("Trend slope negative")
    if current_price < first15_high:
        exit_reasons.append("Below first 15-minute high")
    if current_price < premarket_high:
        exit_reasons.append("Below premarket high")
    if percentile < 50:
        exit_reasons.append("Percentile < 50")
    if buyzone == "OUT":
        exit_reasons.append("BuyZone = OUT")
    if current_price < yesterday_close:
        exit_reasons.append("Below yesterday close")

    # Output
    st.subheader(f"📌 Ticker: {ticker.upper()}")

    if exit_reasons:
        st.error("❌ EXIT")
        for r in exit_reasons:
            st.write(f"- {r}")
    else:
        st.success("✅ KEEP")
        st.write("Momentum structure intact.")

    # Details
    st.markdown("---")
    st.subheader("📊 Details")
    st.write(f"**Current Price:** {current_price:.2f}")
    st.write(f"**VWAP:** {vwap:.2f}")
    st.write(f"**EMA9 Slope:** {ema9_slope:.3f}%")
    st.write(f"**EMA20 Slope:** {ema20_slope:.3f}%")
    st.write(f"**Trend Slope:** {trend_slope:.3f}%")
    st.write(f"**Percentile:** {percentile:.2f}")
    st.write(f"**Premarket High:** {premarket_high:.2f}")
    st.write(f"**First 15-Min High:** {first15_high:.2f}")
    st.write(f"**Yesterday Close:** {yesterday_close:.2f}")
    st.write(f"**BuyZone:** {buyzone}")


# ---------------------------------------------------------
# BUTTON CALL
# ---------------------------------------------------------
if run_button and ticker.strip():
    run_tracker()
