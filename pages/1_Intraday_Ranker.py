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
max_price = st.number_input("Maximum Price", value=400.0, key="intraday_max_price")

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
# PARAMETER DEFINITIONS
# ---------------------------------------------------------
st.markdown("### 📘 Intraday Ranker — Parameter Definitions (2026 Edition)")

st.markdown("""
---

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
- ATR% > 5% → +15 points  

### **RVOL**
- RVOL > 2 → +20 points  

### **Gap%**
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

## 🧮 Total Score Formula (0–70)

**Score = Trend (0–20) + RVOL (0–20) + Gap (0–10) + ATR (0–15) + Baseline (5)**  

PCA, VMAS, and BuyZone logic determine **readiness state** and **Buy_Signal**.

---

## 🌐 Universe Logic

### **Primary Universe**
SP500 + NASDAQ500 + price-filtered tickers.

### **Exploration Universe**
Always included:
- Leveraged ETFs  
- AI cluster  
- Space cluster  

Shown separately for high-opportunity monitoring.

---
""")
