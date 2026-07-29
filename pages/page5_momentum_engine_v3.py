# 📈 Page5 Momentum Engine v3 — Premium Outlier Scanner (Price > 50)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from utils.universe import load_sp500, load_nasdaq1000, load_nyse_top100
from analysis.choppy_engine import calculate_ema_compression, check_intraday_actionability
from analysis.intraday_ranker_v3 import fetch_intraday
from utils.or15 import calculate_or15

# ---------------------------------------------------------
# OR15 breakout scoring
# ---------------------------------------------------------
def compute_or15_score(df_intraday: pd.DataFrame) -> float:
    or15 = calculate_or15(df_intraday)
    if or15 is None:
        return 0.0

    last_price = float(df_intraday["Close"].iloc[-1])

    if last_price > float(or15["OR15_High"]):
        return 3.0
    if last_price > float(or15["OR15_Close"]):
        return 1.5
    if last_price < float(or15["OR15_Low"]):
        return -2.0

    return 0.0


# ---------------------------------------------------------
# Cached universe + daily data
# ---------------------------------------------------------
@st.cache_data(ttl=28800)
def generate_universe():
    sp500 = set(load_sp500())
    nasdaq1000 = set(load_nasdaq1000())
    nyse100 = set(load_nyse_top100())
    full = sorted(sp500 | nasdaq1000 | nyse100)
    return full, sp500, nasdaq1000, nyse100

@st.cache_data(ttl=28800)
def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="60d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


@st.cache_data(ttl=3600)
def load_index_daily(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="5d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


@st.cache_data(ttl=300)
def load_index_intraday(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# ---------------------------------------------------------
# Page header + controls
# ---------------------------------------------------------
st.set_page_config(page_title="Page5 Momentum Engine", layout="wide")
st.title("🚀 Page5 — Premium Momentum Engine (> $50)")
st.write("Ranks premium equities decoupling from weak, bearish, or choppy markets using multi-factor anomaly scoring.")

col1, col2 = st.columns(2)
with col1:
    price_floor = st.number_input("Minimum Price Gate", value=50.00, step=5.0)
with col2:
    run_scan = st.button("Execute Momentum Scan")

universe_list, sp500, nasdaq1000, nyse100 = generate_universe()

def classify_universe(ticker: str) -> str:
    if ticker in sp500:
        return "SP500"
    if ticker in nasdaq1000:
        return "NASDAQ1000"
    if ticker in nyse100:
        return "NYSE100"
    return "OTHER"


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------
if run_scan:
    spy_daily = load_index_daily("SPY")
    qqq_daily = load_index_daily("QQQ")
    spy_intraday = load_index_intraday("SPY")
    qqq_intraday = load_index_intraday("QQQ")

    if spy_daily.empty or qqq_daily.empty or spy_intraday.empty or qqq_intraday.empty:
        st.error("Index benchmarks unavailable. Check connections.")
        st.stop()

    spy_today_return = (
        float(spy_intraday["Close"].iloc[-1]) / float(spy_daily["Close"].iloc[-2]) - 1.0
    )
    qqq_today_return = (
        float(qqq_intraday["Close"].iloc[-1]) / float(qqq_daily["Close"].iloc[-2]) - 1.0

    )

    results = []
    progress = st.progress(0.0, text="Scanning premium universe...")

    total = len(universe_list)
    for idx, ticker in enumerate(universe_list):
        if idx % 50 == 0:
            progress.progress(idx / total, text=f"Analyzing {ticker}...")

        try:
            intraday_df = fetch_intraday(ticker)
            if intraday_df is None or intraday_df.empty:
                continue

            if not all(c in intraday_df.columns for c in ["Close", "Open", "Volume"]):
                continue

            current_price = float(intraday_df["Close"].iloc[-1])
            if current_price < price_floor:
                continue  # premium gate

            df_daily = load_daily(ticker)
            if df_daily.empty or len(df_daily) < 5:
                continue

            if "Close" not in df_daily.columns or "Volume" not in df_daily.columns:
                continue

            previous_close = float(df_daily["Close"].iloc[-2])
            asset_today_return = (current_price / previous_close) - 1.0

            benchmark_return = qqq_today_return if ticker in nasdaq1000 else spy_today_return
            rs_divergence = asset_today_return - benchmark_return

            avg_daily_vol = float(df_daily["Volume"].tail(20).mean())
            intraday_vol = float(intraday_df["Volume"].sum())
            rvol = intraday_vol / avg_daily_vol if avg_daily_vol > 0 else 0.0

            if len(intraday_df) > 10:
                roc = (float(intraday_df["Close"].iloc[-1]) /
                       float(intraday_df["Close"].iloc[-10])) - 1.0
            else:
                roc = 0.0

            if len(intraday_df) >= 20:
                avg_intraday_vol = float(intraday_df["Volume"].mean())
                last_vol = float(intraday_df["Volume"].iloc[-1])
                vol_intensity = last_vol / avg_intraday_vol if avg_intraday_vol > 0 else 0.0
            else:
                vol_intensity = 0.0

            or15_score = compute_or15_score(intraday_df)

            # Structural readiness from Page4 (intraday compression preferred)
            df_struct = intraday_df[["Close", "Volume"]].dropna().copy()
            df_struct = calculate_ema_compression(df_struct)
            structural_ready = bool(df_struct["Structural_Ready"].iloc[-1]) if "Structural_Ready" in df_struct.columns else False

            actionable = check_intraday_actionability(current_price, previous_close, intraday_df)

            universe_label = classify_universe(ticker)

            score = (
                (rs_divergence * 8.0) +
                (roc * 6.0) +
                (np.log1p(rvol) * 4.0) +
                (np.log1p(vol_intensity) * 3.0) +
                (or15_score * 2.0)
            )

            results.append({
                "Ticker": ticker,
                "Universe": universe_label,
                "Price": round(current_price, 2),
                "Alpha_Decoupling%": round(rs_divergence * 100, 2),
                "RVOL": round(rvol, 2),
                "ROC_10m%": round(roc * 100, 2),
                "Vol_Intensity": round(vol_intensity, 2),
                "OR15_Score": or15_score,
                "Structural_Ready": structural_ready,
                "Actionable": actionable,
                "Composite_Score": round(score, 2),
            })

        except Exception:
            continue

    progress.empty()

    if not results:
        st.warning("No premium momentum outliers satisfied requirements for this time window.")
    else:
        df_results = pd.DataFrame(results).sort_values("Composite_Score", ascending=False)

        st.subheader(f"🚀 Top Premium Momentum Outliers ({len(df_results)} Assets)")
        st.caption("Ranked by Composite Momentum Score — higher values indicate strong decoupling from market gravity.")

        st.dataframe(
            df_results.head(50),
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Alpha_Decoupling%": st.column_config.NumberColumn(format="%.2f%%"),
                "ROC_10m%": st.column_config.NumberColumn(format="%.2f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Active Universe Distribution")
        universe_counts = df_results["Universe"].value_counts().reset_index()
        universe_counts.columns = ["Universe Included", "Active Matches Count"]
        st.dataframe(universe_counts, use_container_width=True)
