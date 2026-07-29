import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------------
# Universe loaders (your actual file)
# ---------------------------------------------------------
from utils.universe import (
    load_sp500,
    load_nasdaq1000,
    load_nyse_top100,
)

# ---------------------------------------------------------
# Page4 engine (your actual file)
# ---------------------------------------------------------
from analysis.choppy_engine import (
    calculate_ema_compression,
    check_intraday_actionability,
)

# ---------------------------------------------------------
# Intraday loader + OR15
# ---------------------------------------------------------
from analysis.intraday_ranker_v3 import fetch_intraday
from utils.or15 import calculate_or15


# ---------------------------------------------------------
# OR15 breakout scoring
# ---------------------------------------------------------
def compute_or15_score(df):
    or15 = calculate_or15(df)
    if or15 is None:
        return 0.0

    last_price = df["Close"].iloc[-1]

    if last_price > or15["OR15_High"]:
        return 3.0
    if last_price > or15["OR15_Close"]:
        return 1.5
    if last_price < or15["OR15_Low"]:
        return -2.0

    return 0.0


# ---------------------------------------------------------
# Page Header
# ---------------------------------------------------------
st.set_page_config(page_title="Outlier Alerts — Page5", layout="wide")
st.title("🚨 Outlier Alerts — Page5")
st.write("Page5 runs in ALL market regimes. It finds strong tickers even when the market is weak or choppy.")


# ---------------------------------------------------------
# Load Universe
# ---------------------------------------------------------
sp500 = load_sp500()
nasdaq1000 = load_nasdaq1000()
nyse100 = load_nyse_top100()

universe = sorted(set(sp500) | set(nasdaq1000) | set(nyse100))


# ---------------------------------------------------------
# Universe Classification
# ---------------------------------------------------------
def classify_universe(ticker: str) -> str:
    if ticker in sp500:
        return "SP500"
    if ticker in nasdaq1000:
        return "NASDAQ1000"
    if ticker in nyse100:
        return "NYSE100"

    # IPO detection: < 30 days of history
    try:
        hist = yf.Ticker(ticker).history(period="60d")
        if len(hist) < 30:
            return "IPO"
    except Exception:
        pass

    return "OTHER"


# ---------------------------------------------------------
# Main Scan
# ---------------------------------------------------------
results = []

st.subheader("Full Universe Scan")
st.caption("Scanning SP500 / NASDAQ1000 / NYSE100 for intraday outliers, plus IPO/OTHER when they appear in data.")

for ticker in universe:
    try:
        intraday_df = fetch_intraday(ticker)
        if intraday_df is None or intraday_df.empty:
            continue

        if not all(col in intraday_df.columns for col in ['Close', 'Open', 'Volume']):
            continue

        # Daily data
        df_daily = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df_daily is None or df_daily.empty:
            continue

        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = [c[0] for c in df_daily.columns]

        if 'Close' not in df_daily.columns or 'Volume' not in df_daily.columns:
            continue

        previous_close = df_daily['Close'].iloc[-2]
        current_price = intraday_df['Close'].iloc[-1]

        # RS divergence
        rs_divergence = (current_price / previous_close) - (
            df_daily['Close'].iloc[-1] / df_daily['Close'].iloc[-5]
        )

        # RVOL
        avg_daily_vol = df_daily['Volume'].tail(20).mean()
        intraday_vol = intraday_df['Volume'].sum()
        rvol = intraday_vol / avg_daily_vol if avg_daily_vol > 0 else np.nan

        # ROC
        if len(intraday_df) > 10:
            roc = (intraday_df['Close'].iloc[-1] / intraday_df['Close'].iloc[-10]) - 1.0
        else:
            roc = np.nan

        # Volume intensity
        avg_intraday_vol = intraday_df['Volume'].mean()
        last_vol = intraday_df['Volume'].iloc[-1]
        vol_intensity = last_vol / avg_intraday_vol if avg_intraday_vol > 0 else np.nan

        # OR15 breakout score
        or15_score = compute_or15_score(intraday_df)

        # Structural compression (Page4 engine)
        df_struct = df_daily[['Close', 'Volume']].dropna().copy()
        df_struct = calculate_ema_compression(df_struct)
        structural_ready = bool(df_struct['Structural_Ready'].iloc[-1]) if 'Structural_Ready' in df_struct.columns else False

        # Intraday actionability (Page4 engine)
        actionable = check_intraday_actionability(current_price, previous_close, intraday_df)

        # Universe classification
        universe_label = classify_universe(ticker)

        # Composite score
        score = (
            (rs_divergence if not np.isnan(rs_divergence) else 0) +
            (roc if not np.isnan(roc) else 0) +
            (np.log1p(rvol) if not np.isnan(rvol) else 0) +
            (np.log1p(vol_intensity) if not np.isnan(vol_intensity) else 0) +
            (or15_score if not np.isnan(or15_score) else 0)
        )
        results.append({
            "Ticker": ticker,
            "Universe": universe_label,
            "Price": round(current_price, 2),
            "RS_Divergence": rs_divergence,
            "RVOL": rvol,
            "ROC": roc,
            "Vol_Intensity": vol_intensity,
            "OR15_Score": or15_score,
            "Structural_Ready": structural_ready,
            "Actionable": actionable,
            "Score": score,
       })


    except Exception:
        continue


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
if not results:
    st.warning("No outlier candidates found in the current universe.")
else:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Score", ascending=False)

    st.subheader("Top Outliers")
    st.caption("Higher Score = stronger intraday outlier (RS, RVOL, ROC, volume, OR15).")

    st.dataframe(df_results.head(50), use_container_width=True)

    st.subheader("Universe Breakdown")
    universe_counts = df_results['Universe'].value_counts().reset_index()
    universe_counts.columns = ['Universe', 'Count']
    st.dataframe(universe_counts, use_container_width=True)


