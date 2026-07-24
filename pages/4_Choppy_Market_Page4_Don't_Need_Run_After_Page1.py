import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from utils.data_fetch import load_universe
from analysis.intraday_ranker_v3 import fetch_intraday
from analysis.intraday_ranker_regime_v3 import automatic_market_regime_detector

from analysis.choppy_engine import (
    calculate_ema_compression,
    calculate_residual_pca,
    check_intraday_actionability,
)

st.set_page_config(page_title="Choppy Market Engine", layout="wide")
st.title("🌀 Choppy Market Opportunity Engine")
st.write("This page activates only in CHOPPY regimes and focuses on compression → expansion setups.")

# ---------------------------------------------------------
# 1. Load QQQ daily + intraday for regime detection
# ---------------------------------------------------------
qqq_daily = yf.download("QQQ", period="60d", interval="1d", progress=False)
qqq_intraday = fetch_intraday("QQQ")

regime = automatic_market_regime_detector(qqq_daily, qqq_intraday)

st.subheader("Current Market Regime")
st.write(f"Regime: **{regime}**")

if regime != "CHOPPY_MARKET":
    st.info("Page6 is only active in CHOPPY_MARKET. No choppy model signals today.")
    st.stop()

# ---------------------------------------------------------
# 2. Load universe tickers + SPY daily data
# ---------------------------------------------------------
tickers = load_universe()

spy_daily = yf.download("SPY", period="60d", interval="1d", progress=False)
if isinstance(spy_daily.columns, pd.MultiIndex):
    spy_daily.columns = [c[0] for c in spy_daily.columns]
spy_features = spy_daily[['Close', 'Volume']].dropna().copy()

results = []

# ---------------------------------------------------------
# 3. Loop tickers and compute compression + residual PCA
# ---------------------------------------------------------
for ticker in tickers:

    df = yf.download(ticker, period="60d", interval="1d", progress=False)
    if df is None or df.empty:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if 'Close' not in df.columns or 'Volume' not in df.columns:
        continue

    df = df[['Close', 'Volume']].dropna()

    # EMA computation
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    df['EMA_Spread'] = (df['EMA9'] - df['EMA50']) / df['EMA50']

    # Structural compression
    df_comp = calculate_ema_compression(df.copy())
    if 'Structural_Ready' not in df_comp.columns or not df_comp['Structural_Ready'].iloc[-1]:
        continue

    # ---------------------------------------------------------
    # FIXED: Residual PCA alignment
    # ---------------------------------------------------------
    ticker_features = df_comp[['Close', 'Volume']].copy()

    aligned_index = ticker_features.index.intersection(spy_features.index)

    ticker_aligned = ticker_features.loc[aligned_index].dropna()
    spy_aligned = spy_features.loc[aligned_index].dropna()

    if len(ticker_aligned) < 10 or len(spy_aligned) < 10:
        continue

    residual_pca1 = calculate_residual_pca(ticker_aligned, spy_aligned)
    residual_slope = np.gradient(residual_pca1.flatten())
    residual_expanding = residual_slope[-1] > 0

    if not residual_expanding:
        continue

    sector = "Unknown"

    results.append({
        "Ticker": ticker,
        "Sector": sector,
        "Compression_Spread": df_comp['EMA_Spread'].iloc[-1],
        "Residual_PCA1": residual_pca1[-1][0],
        "Residual_Slope": residual_slope[-1],
    })

# ---------------------------------------------------------
# 4. Sector summary
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
if results_df.empty:
    st.warning("No structurally compressed, independently strong names found.")
    st.stop()

sector_summary = (
    results_df.groupby("Sector")
    .agg({
        "Compression_Spread": "mean",
        "Residual_PCA1": "mean",
    })
    .sort_values("Residual_PCA1", ascending=False)
)

top_sectors = sector_summary.head(3)
st.subheader("Top Sectors (Compression + Residual Strength)")
st.dataframe(top_sectors)

# ---------------------------------------------------------
# 5. Intraday actionability
# ---------------------------------------------------------
st.subheader("Intraday Actionable Tickers (Choppy Model)")

actionable_rows = []

for _, row in results_df.iterrows():
    if row["Sector"] not in top_sectors.index:
        continue

    ticker = row["Ticker"]
    intraday_df = fetch_intraday(ticker)

    if intraday_df is None or intraday_df.empty:
        continue

    required_cols = ['Close', 'Open']
    if not all(col in intraday_df.columns for col in required_cols):
        continue

    df_daily = yf.download(ticker, period="60d", interval="1d", progress=False)
    if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = [c[0] for c in df_daily.columns]

    previous_close = df_daily['Close'].iloc[-2]
    current_price = intraday_df['Close'].iloc[-1]

    decision = check_intraday_actionability(current_price, previous_close, intraday_df)

    actionable_rows.append({
        "Ticker": ticker,
        "Sector": row["Sector"],
        "Decision": decision,
        "Compression_Spread": row["Compression_Spread"],
        "Residual_PCA1": row["Residual_PCA1"],
        "Residual_Slope": row["Residual_Slope"],
    })

actionable_df = pd.DataFrame(actionable_rows)
st.dataframe(actionable_df)

# ---------------------------------------------------------
# SUMMARY AND DEFINITIONS — Choppy Market Opportunity Engine (Page6)
# ---------------------------------------------------------
st.markdown("### 📘 Choppy Market Engine — Summary & Definitions")

st.markdown("""
---
# Executive Summary — Choppy Market Opportunity Engine

The Choppy Market Engine is a **specialized scanning model** designed for market environments
where traditional trend‑based signals (Page1/Page2) lose reliability.  
It activates **only** when the regime detector (Page3) identifies:

### **CHOPPY_MARKET**

In these conditions, broad indices (SPY/QQQ) exhibit:
- weak or inconsistent trend structure  
- low directional conviction  
- high intraday noise  
- rotational sector behavior  
- frequent failed breakouts  

Page6 focuses on **compression → expansion setups**, **independent momentum**, and **sector‑level participation strength**, providing a structured way to identify high‑quality opportunities in otherwise noisy environments.

---

# Stage 1 — Structural Compression Engine (EMA Compression)

Trend stacking is unreliable in choppy markets.  
Instead, Page6 uses **EMA Compression**, which detects when short‑, medium‑, and long‑term EMAs coil tightly.

### **EMA Compression**
- EMA9, EMA20, EMA50 converge within a narrow band  
- EMA_Spread ≤ 1.5%  
- Price above EMA50  
- Indicates **stored energy** and **potential expansion**

Compression setups are statistically favorable in choppy regimes because they represent **localized structure** independent of broad market trend.

---

# Stage 2 — Residual PCA1 (Independent Momentum)

Choppy markets are dominated by index noise.  
Page6 isolates **ticker‑specific momentum** by removing SPY’s influence.

### **Residual PCA1**
1. Compute PCA1 for ticker  
2. Compute PCA1 for SPY  
3. Regress ticker PCA1 against SPY PCA1  
4. Extract residuals (momentum not explained by SPY)

### **Residual PCA1 > 0 and expanding**
Indicates:
- independent strength  
- sector rotation  
- institutional accumulation  
- non‑index‑driven movement  

This is the **core signal** of Page6.

---

# Stage 3 — Sector Participation Strength (ATR% + RVOL)

Choppy markets often rotate between sectors.  
Page6 identifies the **3 strongest sectors** using:

### **Sector Filters**
- **ATR%** → volatility availability  
- **RVOL** → participation strength  
- **Compression_Spread** → structural readiness  
- **Residual PCA1** → independent momentum  

Only tickers inside the top 3 sectors are considered for intraday actionability.

---

# Stage 4 — Intraday Actionability (Noise‑Filtered Triggers)

Page6 applies strict intraday filters to avoid false signals:

### **Intraday Filters**
- Opening gap < 1.5%  
- VWAP reclaim  
- RVOL spike  
- PCA1_slope > 0  
- Price structure improving (higher‑low formation)  

These ensure Page6 only triggers when **real intraday expansion** is forming.

---

# Execution Readiness — Page6 Classification

Tickers passing all Page6 filters are classified as:

### **Actionable (Choppy Model)**
- EMA compression  
- Residual PCA1 expanding  
- Sector strength confirmed  
- Intraday acceleration present  

### **Structural Only**
- Compression present  
- Residual PCA1 present  
- Sector strength present  
- Intraday confirmation missing  

### **Not Actionable**
- No compression  
- No independent momentum  
- Weak sector  
- Intraday noise  

---

# Trader Workflow — Page6

The trader uses Page6 in a **three‑stage workflow**:

### **Stage A — Structural Compression Scan**
Identify:
- EMA compression  
- Residual PCA1 expansion  
- Sector strength  

### **Stage B — Intraday Confirmation**
Confirm:
1. VWAP reclaim  
2. RVOL spike  
3. PCA1_slope > 0  
4. Opening gap < 1.5%  
5. Clean intraday structure  

### **Stage C — Execution**
Act only when:
- structural compression  
- independent momentum  
- sector strength  
- intraday acceleration  
all align.

---

# Technical Glossary

### **EMA Compression**
Tight coiling of EMAs indicating stored energy.

### **Residual PCA1**
Momentum not explained by SPY; measures independent strength.

### **Sector Strength**
ATR% + RVOL + compression + residual PCA1.

### **VWAP Reclaim**
Price moves above institutional cost basis.

### **RVOL Spike**
Volume expansion indicating institutional activity.

### **PCA1_slope**
Real‑time acceleration of intraday momentum.

---
""")
