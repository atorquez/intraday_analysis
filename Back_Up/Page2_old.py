# =========================================================
# PAGE 2 — INTRADAY ENGINE ONLY (FINAL PATCHED VERSION)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
#st.sidebar.title("Navigation")
#page = st.sidebar.radio("Go to:", ["Page 1 — Full Model", "Page 2 — Intraday Engine"])
#page = "Page 2 — Intraday Engine"

# ---------------------------------------------------------
# Page Title
# ---------------------------------------------------------
st.title("📈 Page 2 — Intraday Engine Only")
st.markdown("Pure intraday scanner (no daily structure).")

# ---------------------------------------------------------
# FILTER PANEL (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

intraday_signal_filter = st.multiselect(
    "Filter by Intraday Buy Signal",
    ["Strong Intraday", "Intraday Buy", "Neutral", "Avoid"],
    default=["Strong Intraday"],
    key="intraday_signal_filter"
)

min_price = st.number_input("Minimum Price", value=5.0, key="intraday_min_price")
max_price = st.number_input("Maximum Price", value=100.0, key="intraday_max_price")

execution_filter = st.multiselect(
    "Filter by Execution Readiness",
    ["Ready", "Crossing Soon", "Intraday False Ready", "Setup Only", "UNKNOWN"],
    default=["Ready"],
    key="intraday_execution_filter"
)

run_intraday = st.button("Run Intraday Engine", key="intraday_run_button")

# ---------------------------------------------------------
# Universe Loader (IDENTICAL TO PAGE 1)
# ---------------------------------------------------------
@st.cache_data
def load_primary_universe():
    from utils.data_fetch import load_universe
    return load_universe()

tickers = load_primary_universe()

# ---------------------------------------------------------
# Intraday Data Fetcher (REAL WRAPPER)
# ---------------------------------------------------------
from analysis.intraday_ranker import fetch_intraday

def fetch_intraday_data(ticker):
    try:
        df = fetch_intraday(ticker)
        return df
    except Exception:
        return None

# ---------------------------------------------------------
# Intraday Indicator Calculations
# ---------------------------------------------------------
def compute_intraday_indicators(df):
    # Normalize column names (robust)
    df = df.rename(columns={
        "Close": "close", "close": "close", "CLOSE": "close",
        "Last": "close", "TradePrice": "close",

        "High": "high", "HIGH": "high", "high": "high",
        "Low": "low", "LOW": "low", "low": "low",

        "Open": "open", "OPEN": "open", "open": "open",

        "Volume": "volume", "VOLUME": "volume",
        "Vol": "volume", "vol": "volume", "volume": "volume",

        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume"
    })

    # EMA
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()

    # VWAP
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vp"] = (df["close"] * df["volume"]).cumsum()
    df["VWAP"] = df["cum_vp"] / df["cum_vol"]

    # ATR%
    df["H-L"] = df["high"] - df["low"]
    df["ATR"] = df["H-L"].rolling(14).mean()
    df["ATR%"] = df["ATR"] / df["close"] * 100

    # RVOL
    df["RVOL"] = df["volume"] / df["volume"].rolling(20).mean()

    return df

# ---------------------------------------------------------
# PCA Engine (Intraday Only)
# ---------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_intraday_pca(df):
    features = ["close", "EMA9", "EMA20", "VWAP", "ATR%", "RVOL"]

    X = df[features].ffill().dropna()

    if len(X) < 10:
        df["PCA1"] = np.nan
        df["PCA2"] = np.nan
        df["PCA3"] = np.nan
        df["PCA1_slope"] = np.nan
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs,
        index=X.index,
        columns=["PCA1", "PCA2", "PCA3"]
    )

    df.loc[pca_df.index, "PCA1"] = pca_df["PCA1"]
    df.loc[pca_df.index, "PCA2"] = pca_df["PCA2"]
    df.loc[pca_df.index, "PCA3"] = pca_df["PCA3"]

    df["PCA1_slope"] = df["PCA1"].diff()

    return df

# ---------------------------------------------------------
# Execution Readiness (Intraday Only)
# ---------------------------------------------------------
def classify_execution_readiness(df):
    latest = df.iloc[-1]

    ema_bull = latest["EMA9"] > latest["EMA20"]
    vwap_reclaim = latest["close"] > latest["VWAP"]
    momentum = latest["PCA1_slope"] > 0
    rvol = latest["RVOL"] > 2

    if ema_bull and vwap_reclaim and momentum and rvol:
        return "Ready"
    if ema_bull and momentum:
        return "Crossing Soon"
    if ema_bull and vwap_reclaim and not rvol:
        return "Intraday False Ready"
    if ema_bull:
        return "Setup Only"
    return "UNKNOWN"

# ---------------------------------------------------------
# Intraday Score (0–40)
# ---------------------------------------------------------
def compute_intraday_score(df):
    latest = df.iloc[-1]
    score = 0

    if latest["EMA9"] > latest["EMA20"]:
        score += 10
    if latest["RVOL"] > 2:
        score += 10
    if latest["ATR%"] > 5:
        score += 10
    if latest["close"] > latest["VWAP"]:
        score += 10

    return score

# ---------------------------------------------------------
# Risk Efficiency Score
# ---------------------------------------------------------
def compute_risk_efficiency(df):
    latest = df.iloc[-1]

    pca_slope = latest["PCA1_slope"]
    rvol = latest["RVOL"]
    atr = latest["ATR%"]

    if pd.isna(pca_slope) or pd.isna(rvol) or pd.isna(atr) or atr == 0:
        return np.nan

    return (pca_slope * rvol) / atr

# ---------------------------------------------------------
# Intraday Buy_Signal (Simplified)
# ---------------------------------------------------------
def classify_intraday_buy_signal(df):
    latest = df.iloc[-1]

    if latest["PCA1"] > 0 and latest["close"] > latest["VWAP"]:
        return "Strong Intraday"
    if latest["PCA1"] > 0:
        return "Intraday Buy"
    if latest["PCA1"] < 0:
        return "Avoid"
    return "Neutral"

# ---------------------------------------------------------
# Run Intraday Engine for All Tickers
# ---------------------------------------------------------
results = []

if run_intraday:
    for ticker in tickers:
        df = fetch_intraday_data(ticker)
        if df is None or df.empty:
            continue

        df = compute_intraday_indicators(df)
        df = compute_intraday_pca(df)

        readiness = classify_execution_readiness(df)
        score = compute_intraday_score(df)
        signal = classify_intraday_buy_signal(df)
        risk_eff = compute_risk_efficiency(df)

        results.append({
            "Ticker": ticker,
            "Price": df.iloc[-1]["close"],
            "EMA Alignment": "Bullish" if df.iloc[-1]["EMA9"] > df.iloc[-1]["EMA20"] else "Bearish",
            "VWAP Status": "Above" if df.iloc[-1]["close"] > df.iloc[-1]["VWAP"] else "Below",
            "RVOL": df.iloc[-1]["RVOL"],
            "ATR%": df.iloc[-1]["ATR%"],
            "PCA1": df.iloc[-1]["PCA1"],
            "PCA1 Slope": df.iloc[-1]["PCA1_slope"],
            "PCA2": df.iloc[-1]["PCA2"],
            "PCA3": df.iloc[-1]["PCA3"],
            "Execution Readiness": readiness,
            "Intraday Score": score,
            "Intraday Buy_Signal": signal,
            "Risk Efficiency Score": risk_eff
        })

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
if run_intraday:
    df_results = pd.DataFrame(results)

    if df_results.empty:
        st.warning("⚠ No intraday data available.")
        st.stop()

    # Price filter
    df_results = df_results[
        (df_results["Price"] >= min_price) &
        (df_results["Price"] <= max_price)
    ]

    # Intraday Buy Signal filter
    if intraday_signal_filter:
        df_results = df_results[
            df_results["Intraday Buy_Signal"].isin(intraday_signal_filter)
        ]

    # Execution Readiness filter
    if execution_filter:
        df_results = df_results[
            df_results["Execution Readiness"].isin(execution_filter)
        ]

    # Sort by Risk Efficiency Score (strongest risk-adjusted candidates first)
    df_results = df_results.sort_values(
        ["Risk Efficiency Score", "Intraday Score"],
        ascending=False
    )

    # FULL-WIDTH TABLE
    st.data_editor(
        df_results,
        use_container_width=True,
        hide_index=True
    )
