# ==============================================================================
# PAGE 2 — INTRADAY ENGINE ONLY (HARDENED PRODUCTION VERSION — FULL PATCH)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.caption("SPC Version: 2026-08-08 (Pure Intraday Hardened)")
st.title("📈 Page 2 — Intraday Engine Only")
st.markdown("Pure intraday scanner operating exclusively on 1-minute real-time structural movements.")

# ---------------------------------------------------------
# FILTER PANEL (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

col1, col2, col3 = st.columns(3)

with col1:
    intraday_signal_filter = st.multiselect(
        "Filter by Intraday Buy Signal",
        ["Strong Intraday", "Intraday Buy", "Neutral", "Avoid"],
        default=["Strong Intraday", "Intraday Buy", "Neutral", "Avoid"],
        key="intraday_signal_filter"
    )

with col2:
    min_price = st.number_input("Minimum Price ($)", value=1.0, key="intraday_min_price")
    max_price = st.number_input("Maximum Price ($)", value=2000.0, key="intraday_max_price")

with col3:
    execution_filter = st.multiselect(
        "Filter by Execution Readiness",
        ["Ready", "Crossing Soon", "Intraday False Ready", "Setup Only", "UNKNOWN"],
        default=["Ready", "Crossing Soon", "Intraday False Ready", "Setup Only", "UNKNOWN"],
        key="intraday_execution_filter"
    )

run_intraday = st.button("Run Intraday Engine", key="intraday_run_button")

# ---------------------------------------------------------
# Universe Loader
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_primary_universe():
    try:
        from utils.data_fetch import load_universe
        return load_universe()
    except Exception:
        return ["AAPL", "NVDA", "MSFT", "AMD", "TSLA", "META", "AMZN", "MDT"]

tickers = load_primary_universe()

# ---------------------------------------------------------
# Intraday Data Fetcher & Data Column Flattener
# ---------------------------------------------------------
from analysis.intraday_ranker import fetch_intraday

def fetch_intraday_data(ticker):
    try:
        df = fetch_intraday(ticker)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception:
        return None

# ---------------------------------------------------------
# Intraday Indicator Calculations
# ---------------------------------------------------------
def compute_intraday_indicators(df):
    df = df.copy()

    rename_dict = {
        "vol": "volume", "tradeprice": "close", "last": "close",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
    }
    df = df.rename(columns=rename_dict)

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            alt_col = col.capitalize()
            if alt_col in df.columns:
                df[col] = df[alt_col]
            else:
                return pd.DataFrame()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()

    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vp"] = (df["close"] * df["volume"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"]

    df["h-l"] = df["high"] - df["low"]
    df["atr"] = df["h-l"].rolling(14).min().ffill().bfill().fillna(0.01)
    df["atr%"] = (df["atr"] / df["close"]) * 100

    rolling_vol_mean = df["volume"].rolling(20).mean().ffill().bfill()
    df["rvol"] = df["volume"] / rolling_vol_mean.replace(0, 1)

    return df

# ---------------------------------------------------------
# PCA Engine
# ---------------------------------------------------------
def compute_intraday_pca(df):
    if df.empty or len(df) < 5:
        df["PCA1"] = 0.0
        df["PCA1_slope"] = 0.0
        return df

    df["spread_ema"] = df["ema9"] - df["ema20"]
    df["dist_vwap"] = df["close"] - df["vwap"]
    df["pct_return"] = df["close"].pct_change().fillna(0.0)

    pca_features = ["spread_ema", "dist_vwap", "pct_return", "atr%", "rvol"]
    X = df[pca_features].ffill().bfill().fillna(0.0)

    if len(X) < 3 or X.var().sum() == 0:
        df["PCA1"] = 0.0
        df["PCA1_slope"] = 0.0
        return df

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=1)
        pcs = pca.fit_transform(X_scaled)

        pca_series = pd.Series(pcs.flatten(), index=X.index)
        df.loc[pca_series.index, "PCA1"] = pca_series
        df["PCA1_slope"] = df["PCA1"].diff().fillna(0.0)
    except Exception:
        df["PCA1"] = 0.0
        df["PCA1_slope"] = 0.0

    return df

# ---------------------------------------------------------
# Scoring and Classification Modules
# ---------------------------------------------------------
def classify_execution_readiness(latest):
    ema_bull = latest["ema9"] > latest["ema20"]
    vwap_reclaim = latest["close"] > latest["vwap"]
    momentum = latest["PCA1_slope"] > 0
    rvol = latest["rvol"] > 2

    if ema_bull and vwap_reclaim and momentum and rvol:
        return "Ready"
    if ema_bull and momentum:
        return "Crossing Soon"
    if ema_bull and vwap_reclaim and not rvol:
        return "Intraday False Ready"
    if ema_bull:
        return "Setup Only"
    return "UNKNOWN"

def compute_intraday_score(latest):
    score = 0
    if latest["ema9"] > latest["ema20"]: score += 10
    if latest["rvol"] > 2: score += 10
    if latest["atr%"] > 5: score += 10
    if latest["close"] > latest["vwap"]: score += 10
    return score

def compute_risk_efficiency(latest):
    pca_slope = latest["PCA1_slope"]
    rvol = latest["rvol"]
    atr = latest["atr%"]
    if pd.isna(pca_slope) or pd.isna(rvol) or pd.isna(atr) or atr == 0:
        return 0.0
    return float((pca_slope * rvol) / atr)

def classify_intraday_buy_signal(latest):
    pca1 = latest["PCA1"]
    vwap_status = latest["close"] > latest["vwap"]

    if pca1 > 0 and vwap_status:
        return "Strong Intraday"
    if pca1 > 0:
        return "Intraday Buy"
    if pca1 < 0:
        return "Avoid"
    return "Neutral"

# ---------------------------------------------------------
# Core Iteration Execution Loop
# ---------------------------------------------------------
if run_intraday:
    results = []

    processing_universe = tickers[:40] if len(tickers) > 40 else tickers
    progress_bar = st.progress(0.0)

    for idx, ticker in enumerate(processing_universe):
        progress_bar.progress((idx + 1) / len(processing_universe))

        raw_df = fetch_intraday_data(ticker)
        if raw_df is None or raw_df.empty or len(raw_df) < 3:
            continue

        processed_df = compute_intraday_indicators(raw_df)
        if processed_df.empty:
            continue

        final_df = compute_intraday_pca(processed_df)
        latest_bar = final_df.iloc[-1].fillna(0.0)

        current_price = float(latest_bar["close"])
        if not (min_price <= current_price <= max_price):
            continue

        readiness = classify_execution_readiness(latest_bar)
        score = compute_intraday_score(latest_bar)
        signal = classify_intraday_buy_signal(latest_bar)
        risk_eff = compute_risk_efficiency(latest_bar)

        results.append({
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "EMA Alignment": "Bullish" if latest_bar["ema9"] > latest_bar["ema20"] else "Bearish",
            "VWAP Status": "Above" if latest_bar["close"] > latest_bar["vwap"] else "Below",
            "RVOL": round(latest_bar["rvol"], 2),
            "ATR%": round(latest_bar["atr%"], 2),
            "PCA1": round(latest_bar["PCA1"], 4),
            "PCA1 Slope": round(latest_bar["PCA1_slope"], 4),
            "Execution Readiness": readiness,
            "Intraday Score": score,
            "Intraday Buy_Signal": signal,
            "Risk Efficiency Score": round(risk_eff, 4)
        })

    progress_bar.empty()

    # ---------------------------------------------------------
    # DISPLAY ENGINE & RENDERER PANEL
    # ---------------------------------------------------------
    if not results:
        st.warning("No assets successfully bypassed background processing parameters. Check your console logs.")
    else:
        master_df = pd.DataFrame(results)

        if intraday_signal_filter:
            master_df = master_df[master_df["Intraday Buy_Signal"].isin(intraday_signal_filter)]

        if execution_filter:
            master_df = master_df[master_df["Execution Readiness"].isin(execution_filter)]

        if master_df.empty:
            st.info("Watchlist generated structural entries, but they were filtered out by user checkbox configurations.")
        else:
            st.subheader(f"🚀 Live Intraday Universe Matrix ({len(master_df)} Tickers)")

            master_df = master_df.sort_values(
                by=["Intraday Score", "Risk Efficiency Score"],
                ascending=[False, False]
            )

            def style_readiness(df):
                style_grid = pd.DataFrame('', index=df.index, columns=df.columns)
                if "Execution Readiness" in df.columns:
                    style_grid["Execution Readiness"] = [
                        "background-color: #2E7D32; color: white; font-weight: bold;" if v == "Ready"
                        else "background-color: #EF6C00; color: white;" if v == "Intraday False Ready"
                        else "background-color: #FBC02D; color: black;" if v == "Crossing Soon"
                        else "background-color: #757575; color: white;"
                        for v in df["Execution Readiness"]
                    ]
                return style_grid

            st.dataframe(
                master_df.style.apply(style_readiness, axis=None),
                hide_index=True,
                use_container_width=True
            )

# ---------------------------------------------------------
# EXECUTIVE SUMMARY, PROCESS, AND DEFINITIONS
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 📘 Pure Intraday Engine — Parameter Definitions")

st.markdown("""
# Executive Summary — Pure Intraday Momentum Framework

The **Intraday Engine Only** dashboard is a high-speed, purely tactical screening engine designed for zero-macro execution. Unlike Stage 1 systems that rely on 3-month daily trends or macro value floors, this engine views the market through a **1-minute localized microscope**. It treats each trading session as a clean slate, capturing rapid structural shifts, immediate volume spikes, and high-velocity momentum expansions occurring strictly within the current session's boundaries.

This engine is built specifically for **scalpers, high-frequency momentum traders, and index-velocity players** who require rapid identification of immediate chart breakouts.

---
## The Data Normalization and Flattening Protocol

Because intraday ticks from yFinance enter your system with varied casing formats and Multi-Index layers depending on data stream patches, Page 2 enforces a strict preprocessing pipeline before any indicator is calculated:
1. **Index Level Stripping:** Collapses multi-tiered column headers down to a single clean layer.
2. **Case Normalization:** Forces all parameters to standardized lowercase characters to completely eliminate string-matching runtime errors.
3. **Indicator Alias Mapping:** Automatically re-maps erratic platform inputs (e.g., `vol`, `last`, `tradeprice`) into predictable software columns (`volume`, `close`).
---

## Advanced Feature Scaling & Stationary PCA

Standard indicator panels break down when absolute prices shift dramatically over a few trading hours. To maintain mathematical validity, the Page 2 PCA engine transforms price levels into **Stationary Feature Relationships**:

- **EMA Spread:** Tracks short-term price velocity by calculating the literal distance between the `EMA9` and `EMA20` vectors (`ema9 - ema20`).
- **VWAP Distance:** Measures true immediate premium or discount valuation by mapping the price delta against the cumulative session baseline (`close - vwap`).
- **Percent Return:** Calculates log-returns on a bar-by-bar minute-interval to observe real-time rate changes.

These stationary inputs are scaled through a `StandardScaler` to remove variance skewing and fed to a single-component **Principal Component Analysis Engine (`PCA1`)**. This isolates the primary direction of structural variance on the 1-minute chart.
---

## Specialized Intraday Risk Analytics

### **1. Intraday Score Formula (0 to 40 Points)**
Assets are ranked using a focused, single-session point aggregation framework. A maximum score of 40 confirms that the stock is displaying total microstructural dominance:
- `EMA9 > EMA20` (Short-term trend acceleration) → **+10 Points**
- `RVOL > 2.0` (Institutional participation is expanding over double its average) → **+10 Points**
- `ATR% > 5.0%` (Asset displays sufficient localized range expansion) → **+10 Points**
- `Close > VWAP` (Buyers successfully control the session average cost basis) → **+10 Points**
---

### **2. Risk Efficiency Score Engine**
The Risk Efficiency metric is a highly custom analytical formula unique to this pure intraday page. It quantifies the amount of directional momentum an asset is generating relative to the amount of localized risk it forces you to assume:

Risk Efficiency Score = (PCA1_slope × RVOL) / ATR%

- **The Numerator (`PCA1_slope * RVOL`):** Multiplies short-term velocity change by institutional participation volume to compute raw momentum force.
- **The Denominator (`ATR%`):** Divides that force by the immediate percentage volatility of the stock.
- **The Trading Application:** This formula filters out erratic, choppy, wide-spread assets and elevates clean, smooth, high-volume directional trend runners directly to the top of your Streamlit table.
---

## 🚦 Dual UI Filter Dimensions

To keep visual inspections completely synchronized with code execution, the dashboard ranks assets across two core column mappings:

1. **Intraday Buy_Signal:** Evaluates value-momentum positioning relative to the daily anchor (`Strong Intraday`, `Intraday Buy`, `Neutral`, or `Avoid`).
2. **Execution Readiness:** Evaluates live microstructural actionability at the current minute bar.

### **Execution Readiness Status Meanings:**
- **Ready:** Short-term EMAs are stacked bullishly, the stock has claimed its daily VWAP line, volume expansion is high, and the PCA slope is moving positive. Fully actionable.
- **Crossing Soon:** Localized price velocity is expanding, and short-term moving average crossovers are actively forming.
- **Intraday False Ready:** Visual indicators look bullish, but the entry lacks vital relative volume (RVOL) or session volatility support. High trap probability.
- **Setup Only:** Structural baselines are organizing, but 1-minute tracking momentum is completely stagnant or recovering from a volume drop.
- **UNKNOWN:** Conflicting indicator properties or insufficient localized candle history.
""")
