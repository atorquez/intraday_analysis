import streamlit as st
from analysis.intraday_ranker import rank_universe, fetch_daily
from utils.data_fetch import load_universe

st.set_page_config(layout="wide")
st.caption("SPC Version: 2026-08-08")
st.title("📈 Intraday Analysis")

# ---------------------------------------------------------
# CACHE DAILY DATA
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_fetch_daily(ticker):
    return fetch_daily(ticker)

# ---------------------------------------------------------
# PRICE FILTERS (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🔍 Price Filter")
min_price = st.number_input("Minimum Price", value=5.0, key="intraday_min_price")
max_price = st.number_input("Maximum Price", value=50.0, key="intraday_max_price")

# ---------------------------------------------------------
# FILTER PANEL (BEFORE RUN)
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

buy_signal_filter = st.multiselect(
    "Filter by Buy Signal",
    ["Strong Buy Zone", "Buy Zone", "Neutral", "Avoid", "Extended"],
    default=["Strong Buy Zone"],
    key="intraday_buy_signal_filter"
)

execution_filter = st.multiselect(
    "Filter by Execution Status",
    ["Ready", "Intraday False Ready", "Crossing Soon", "Setup Only", "UNKNOWN"],
    default=["Ready"],
    key="intraday_execution_filter"
)

# ---------------------------------------------------------
# RUN RANKER (BUTTON BEFORE DEFINITIONS)
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model", key="intraday_run_button")

if run_model:

    base_universe = load_universe()

    filtered = []
    for ticker in base_universe:
        df = cached_fetch_daily(ticker)

        if df is None or len(df) < 1:
            continue

        last_close = df["Close"].iloc[-1]
        if min_price <= last_close <= max_price:
            filtered.append(ticker)

    final_universe = list(set(filtered))

    ranking = rank_universe(final_universe)

    # APPLY FILTERS
    ranking_display = ranking.copy()

    if buy_signal_filter:
        ranking_display = ranking_display[
            ranking_display["Buy_Signal"].isin(buy_signal_filter)
        ]

    if execution_filter:
        ranking_display = ranking_display[
            ranking_display["Execution_Status"].isin(execution_filter)
        ]

    # COLORING
    def color_execution_column(col):
        return [
            "background-color: #4CAF50; color: white" if v == "Ready"
            else "background-color: #FFC107; color: black" if v == "Crossing Soon"
            else "background-color: #FF9800; color: white" if v == "Intraday False Ready"
            else "background-color: #9E9E9E; color: white"
            for v in col
        ]

    # FINAL PANEL
    st.subheader("🚀 Primary Universe — Filtered Results")

    st.dataframe(
        ranking_display.style.apply(color_execution_column, subset=["Execution_Status"]),
        hide_index=True,
        width="stretch"
    )

# ---------------------------------------------------------
# SUMMARY AND DEFINITIONS
# ---------------------------------------------------------
st.markdown("### 📘 Intraday Ranker — Parameter Definitions (2026 Edition)")

st.markdown("""
---
# Executive Summary — SPC Intraday Readiness Model (2026 Edition)

The Intraday Readiness Model is a multi‑factor, PCA‑enhanced intraday scanning system designed to identify statistically favorable trading locations during the trading session. It integrates daily historical structure with real‑time intraday momentum, volatility, and participation signals to produce a ranked universe of tickers and classify each into two orthogonal dimensions:

1. **Buy_Signal** — value‑momentum alignment  
2. **Execution_Status** — intraday actionability  

The model is intentionally **location‑based**, not predictive. It identifies *where* opportunity exists, while the trader determines *when* to act through visual inspection and intraday monitoring.

---

## Two‑Stage Scientific Framework

### **Stage 1 — Daily Structure (Value + Trend Foundation)**  
Using 3 months of daily OHLCV data, the model computes:
- Trend alignment (EMA9, EMA20, EMA50)  
- Volatility (ATR%)  
- Participation (RVOL)  
- Opening pressure (Gap%)  
- Value zones (BuyZone10, BuyZone5 via rolling percentiles)  

This establishes the macro context and identifies statistically favorable value areas.
---

### **Stage 2 — Intraday PCA Engine (Momentum + Volatility + Participation)**  
Using intraday‑sensitive indicators, the model applies StandardScaler + PCA to extract three orthogonal “super‑signals”:
- **PCA1 — Momentum cluster**  
- **PCA2 — Volatility cluster**  
- **PCA3 — Participation cluster**  

These components update intraday, giving the model real‑time sensitivity to market shifts.
---

## Decision Layer — BuyZone + PCA + Trend Integration

The model combines:
- Value proximity (BuyZone10/5)  
- Trend alignment  
- PCA1 momentum  
- VMAS (value–momentum alignment)  
- Distance from value  
- Heatmap classification  

To produce the **Buy_Signal** classification:
- **Strong Buy Zone**  
- **Buy Zone**  
- **Neutral**  
- **Avoid**  
- **Extended**

This dimension answers:  
**“Is the ticker in a statistically favorable location?”**
---

## NEW (2026) — Execution Readiness Status

Execution readiness evaluates whether a ticker is **actionable intraday**, combining trend structure, VWAP behavior, RVOL participation, volatility state, and PCA momentum slope.

The model classifies each ticker into:
- **Ready** — Trend aligned, VWAP reclaimed, momentum supportive  
- **Crossing Soon** — Alignment forming, momentum improving  
- **Intraday False Ready** — Appears ready but fails VWAP/RVOL/volatility confirmation  
- **Setup Only** — Structure forming but missing momentum or participation  
- **UNKNOWN** — Insufficient or conflicting data  

This dimension answers:  
**“Is the ticker actionable *right now*?”**
---

## Trader Workflow

The trader runs the model periodically (e.g., every 15–30 minutes), identifies **Strong Buy Zone + Ready** candidates, and then uses visual inspection to confirm timing based on:
- EMA9 reclaim  
- RSI > 50  
- PCA1 slope  
- RVOL trend  
- Higher‑low formation  
- VWAP behavior  

This creates a disciplined intraday workflow where the model finds **location**, and the trader confirms **timing**.
---

## Summary
The Intraday Readiness Model is a multi‑factor, PCA‑enhanced, value‑momentum classifier that identifies statistically favorable intraday trading locations using daily historical structure and real‑time intraday signals. It produces two orthogonal classifications — **Buy_Signal** and **Execution_Status** — enabling traders to focus only on high‑quality, actionable intraday opportunities. The model is location‑based, not predictive: it finds *where* opportunity exists, while the trader confirms *when* to act.
          

## 🏛 Daily Structure (Value + Trend Foundation)

### **BuyZone10 / BuyZone5**
Rolling percentile-based value zones computed from 3 months of daily data:
- **BuyZone10:** 10th percentile of closing prices  
- **BuyZone5:** 5th percentile of closing prices  

### **BuyZone Distance%**
Measures how far current price is from the BuyZone:
- **Distance < 0%:** Inside BuyZone  
- **0–3%:** Near Value  
- **3–10%:** Normal  
- **>10%:** Extended  

---

## 🎨 BuyZone Heatmap Classification

| State | Meaning |
|-------|---------|
| **Inside_5** | Deepest value zone |
| **Inside_10** | Broad value zone |
| **Near_Value** | Within 3% of BuyZone10 |
| **Normal** | 3–10% above BuyZone10 |
| **Extended** | >10% above BuyZone10 |

---

## 📈 Trend Structure (EMA9 / EMA20 / EMA50)
- **Bullish:** EMA9 > EMA20 > EMA50  
- **Bearish:** EMA9 < EMA20 < EMA50  
- **Flat:** No clear alignment  

**Trend Score:**  
- EMA9 > EMA20 → +10  
- EMA20 > EMA50 → +10  
(Max = 20)

---

## 🔥 Volatility & Participation

### **ATR% (14‑period)**
Measures volatility relative to price.  
- ATR% > 5% → +15 points  

### **RVOL — Relative Volume**
Compares current volume to the 20‑day average.  
- RVOL > 2 → +20 points  

### **Gap%**
Measures opening pressure.  
- 2%–5% gap → +10 points  

---

## 🧮 Baseline Readiness
Every ticker receives **+5 points**.

---

## 🧠 PCA Engine (Momentum + Volatility + Participation)

### **PCA1 — Momentum**
RSI, MACD, ROC, Stochastics, EMA curvature.

### **PCA2 — Volatility**
BBW, ATR, range expansion.

### **PCA3 — Participation**
RVOL trend, VWAP distance, volume delta.

---

## 🔗 VMAS — Value–Momentum Alignment Score
Measures whether momentum aligns with value proximity.

---

## 🚦 Buy_Signal Engine

| Signal | Meaning |
|--------|---------|
| **Strong Buy Zone** | Inside BuyZone5/10 + strong momentum |
| **Buy Zone** | Near value + improving momentum |
| **Neutral** | Mixed signals |
| **Avoid** | Weak momentum or extended |
| **Extended** | Price too far above value |

---

## 🟩 Execution Readiness Status (NEW — 2026 Edition)

Execution readiness evaluates whether a ticker is **actionable intraday**, combining trend, momentum, VWAP, RVOL, and volatility structure.

| Status | Meaning |
|--------|---------|
| **Ready** | Trend aligned, momentum strong, VWAP reclaimed, RVOL supportive — fully actionable |
| **Crossing Soon** | Momentum improving, VWAP/EMA alignment forming — actionable soon |
| **Intraday False Ready** | Appears ready but fails VWAP, RVOL, or volatility confirmation |
| **Setup Only** | Structure forming but missing momentum or participation — watchlist only |
| **UNKNOWN** | Insufficient data or conflicting signals |

---

## 🧮 Total Score Formula (0–70)

**Score = Trend (0–20) + RVOL (0–20) + Gap (0–10) + ATR (0–15) + Baseline (5)**  

PCA, VMAS, and BuyZone logic determine **Buy_Signal** and **Execution_Status**.

---

## 🌐 Universe Logic (Updated)

### **Primary Universe Only**
SP500 + NASDAQ500 + price-filtered tickers.

Exploration Universe (leveraged ETFs, AI cluster, space cluster)  
is **not included** on this page.

---

# 📘 Glossary — Acronym Definitions

### **OHLCV — Open, High, Low, Close, Volume**
Fundamental market data used for all daily and intraday calculations.

### **VWAP — Volume‑Weighted Average Price**
Represents the market’s average cost basis for the day, weighted by volume.  
Used to confirm trend strength, reclaim attempts, and execution readiness.

### **RVOL — Relative Volume**
Measures current volume relative to the 20‑day average.  
High RVOL confirms participation; low RVOL indicates weak or unreliable signals.

### **PCA — Principal Component Analysis**
Statistical technique that compresses many correlated indicators into three “super‑signals”:  
- **PCA1:** Momentum  
- **PCA2:** Volatility  
- **PCA3:** Participation  

### **EMA — Exponential Moving Average**
Trend indicators used for alignment and execution readiness (EMA9, EMA20, EMA50).

### **ATR — Average True Range**
Volatility measure used to detect expansion, contraction, and risk conditions.

### **VMAS — Value–Momentum Alignment Score**
Measures whether momentum (PCA1) aligns with value proximity (BuyZone).

---
""")
