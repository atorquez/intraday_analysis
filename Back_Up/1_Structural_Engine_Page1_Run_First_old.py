import importlib
import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)
print(">>> USING FILE:", v3.__file__)

import analysis.intraday_ranker_v3 as v3
rank_universe = v3.rank_universe

import streamlit as st
import pandas as pd
from utils.data_fetch import load_universe, get_universe_source

st.set_page_config(layout="wide")
st.caption("Version: 2026-07-21")
st.title("📈 Structural Engine Page1")

# ---------------------------------------------------------
# CACHED WRAPPER FOR RANK_UNIVERSE (FIXED: Prevents re-download)
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_rank_universe(tickers_tuple, buy_zone_percentile=0.15):
    return rank_universe(list(tickers_tuple), buy_zone_percentile)

# ---------------------------------------------------------
# COLOR HELPER
# ---------------------------------------------------------
def color_execution_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)

    if "Execution" in df.columns:
        style_df["Execution"] = [
            "background-color: #4CAF50; color: white;" if v == "Watch List"
            else "background-color: #FFC107; color: black;" if v == "Crossing Soon"
            else "background-color: #FF9800; color: white;" if v == "Not Watch List"
            else "background-color: #9E9E9E; color: white;"
            for v in df["Execution"]
        ]
    return style_df

# ---------------------------------------------------------
# RESTORE PREVIOUS FILTERED RESULTS IF AVAILABLE
# ---------------------------------------------------------
filtered_results = st.session_state.get("intraday_visual_results", None)
structural_results = st.session_state.get("intraday_filtered_results", None)

# ---------------------------------------------------------
# PRICE FILTERS
# ---------------------------------------------------------
st.markdown("### 🔍 Price Filter")
min_price = st.number_input("Minimum Price", value=50.0, key="intraday_min_price")
max_price = st.number_input("Maximum Price", value=200.0, key="intraday_max_price")

# ---------------------------------------------------------
# FILTER PANEL
# ---------------------------------------------------------
st.markdown("### 🎛️ Filters")

buy_signal_filter = st.multiselect(
    "Filter by Buy Signal Zone",
    ["DEEP_VALUE_ZONE", "MID_VALUE_ZONE", "NEAR_VALUE_ZONE", "EXTENDED_ZONE", "UNKNOWN"],
    default=["DEEP_VALUE_ZONE", "MID_VALUE_ZONE"],
    key="intraday_buy_signal_filter"
)

execution_filter = st.multiselect(
    "Filter by Execution Status",
    ["Watch List", "Not Watch List", "Crossing Soon", "Setup Only"],
    default=["Crossing Soon"],
    key="intraday_execution_filter"
)

# ---------------------------------------------------------
# RUN RANKER
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model", key="intraday_run_button")

if run_model:
    import time
    start_time = time.time()
    st.write(f"⏱️ Start Time: {pd.Timestamp.now()}")

    progress_bar = st.progress(0, text="Loading universe...")

    try:
        base_universe = load_universe()
        if not base_universe:
            st.error("The source stock universe list returned empty.")
            st.stop()

        progress_bar.progress(0.1, text=f"Scanning {len(base_universe)} tickers...")

        ranking = cached_rank_universe(tuple(base_universe))

        progress_bar.progress(0.6, text="Applying filters...")

        if ranking is None or ranking.empty:
            st.warning("No stock configurations satisfied the technical requirements.")
        else:
            st.markdown(f"**Structural Universe (Daily):** {len(ranking)} tickers")

            # Add Universe Source column (SP500 / NASDAQ1000)
            ranking["Universe"] = ranking["Ticker"].apply(get_universe_source)

            filtered = ranking.copy()

            # Price filters
            filtered = filtered[
                (filtered["Close"] >= min_price) &
                (filtered["Close"] <= max_price)
            ]

            # Execution filter
            if execution_filter:
                filtered = filtered[
                    filtered["Execution"].isin(execution_filter)
                ]

            # Store results for Page2
            st.session_state["intraday_filtered_results"] = ranking
            st.session_state["intraday_visual_results"] = filtered

            st.markdown(f"**User Visual Filters Applied:** {len(filtered)} tickers shown")
            st.markdown(f"**Tickers Passed to Page 2 (Structural):** {len(ranking)} tickers")

            if filtered.empty:
                st.info("No companies matched the selected filters.")
            else:
                # ---------------------------------------------------------
                # DISPLAY SECTION — FIXED TO SHOW TICKER (UNIVERSE)
                # ---------------------------------------------------------

                # Build display-only DataFrame
                display_df = filtered.copy()

                # Add Universe Source next to ticker name
                display_df["Ticker"] = display_df.apply(
                    lambda row: f"{row['Ticker']} ({row['Universe']})",
                    axis=1
                )

                # Remove Universe column from display
                display_df = display_df.drop(columns=["Universe"])

                # Apply styling AFTER modifying the DataFrame
                styled = display_df.style.apply(color_execution_column, axis=None)

                st.subheader(f"🚀 Primary Universe — {len(display_df)} Filtered Results")
                st.dataframe(
                    styled,
                    hide_index=True,
                    use_container_width=True
                )

        progress_bar.empty()
        end_time = time.time()
        elapsed = end_time - start_time
        st.write(f"⏱️ End Time: {pd.Timestamp.now()}")
        st.write(f"⚡ Total Runtime: {elapsed:.2f} seconds")
    
    except Exception as e:
        progress_bar.empty()
        st.error(f"Model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# RENDER STORED RESULTS
# ---------------------------------------------------------
elif filtered_results is not None:
    if structural_results is not None:
        st.markdown(f"**Structural Universe (Daily):** {len(structural_results)} tickers")
        st.markdown(f"**User Visual Filters Applied (Stored):** {len(filtered_results)} tickers shown")
        st.markdown(f"**Tickers Passed to Page 2 (Structural):** {len(structural_results)} tickers")

    st.subheader(f"🚀 Primary Universe — {len(filtered_results)} Stored Results")

    st.dataframe(
        filtered_results.style.apply(color_execution_column, axis=None),
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------------
# SUMMARY AND DEFINITIONS — SPC Intraday Ranker v3
# ---------------------------------------------------------
st.markdown("### 📘 Structural Engine")

st.markdown("""
---
# Executive Summary

It is **hybrid institutional‑grade scanning engine** that merges
daily structural trend analysis with real‑time intraday momentum diagnostics.  
It is designed to identify **high‑quality, statistically favorable trading environments**
across large universes (1000–1500 tickers) while remaining robust in choppy or down‑trend regimes.

The model evaluates each ticker across **two orthogonal dimensions**:

1. **Daily Structural Readiness**  
   - Trend stack (EMA9 > EMA20 > EMA50)  
   - EMA slope alignment  
   - PCA1 (daily momentum cluster)  
   - ATR% (volatility availability)  
   - RVOL (participation strength)  
   - Gap% (opening pressure)

2. **Intraday Actionability**  
   - EMA9/EMA20 micro‑trend  
   - VWAP reclaim  
   - PCA1_slope (intraday acceleration)  
   - RVOL spikes  
   - ATR% friction clearance  
   - Price extension diagnostics

The engine is **location‑based**, not predictive:  
it identifies *where* structural advantage exists, while the trader confirms *when* to execute.

---

# Stage 1 — Daily Structural Engine (Trend + Momentum Foundation)

Using 3 months of daily OHLCV data, v3 computes:

### **Trend Stack**
- **UP:** `EMA9 > EMA20 > EMA50`
- **DOWN:** `EMA9 < EMA20 < EMA50`
- **FLAT:** No alignment

### **EMA Slope Alignment**
- Positive EMA9 slope → short‑term velocity  
- Positive EMA20 slope → medium‑term stability  

### **PCA1 (Daily Momentum Cluster)**
Extracted from:
- RSI  
- Bollinger Width  
- ROC  
- StochK  
- EMA curvature  
- Volume delta  
- VWAP distance  

PCA1 > 0 indicates **momentum support**.

### **Volatility & Participation**
- **ATR%** → daily range availability  
- **RVOL** → institutional participation  
- **Gap%** → opening pressure  

These define the **macro structural readiness** of the ticker.

---

# Stage 2 — Intraday PCA Engine (Real‑Time Momentum + Participation)

Using 1‑minute bars, v3 computes:

### **EMA Micro‑Trend**
- EMA9 vs EMA20  
- Spread acceleration  

### **VWAP Alignment**
- Price above VWAP → institutional support  
- Price below VWAP → liquidity drag  

### **PCA1_slope (Intraday Acceleration)**
Derived from:
- EMA spread  
- VWAP distance  
- Intraday returns  
- ATR%  
- RVOL  

PCA1_slope > 0 indicates **real‑time acceleration**.

### **RVOL Spikes**
Detect institutional block execution.

### **ATR% Intraday**
Ensures friction costs can be cleared.

---

# Execution Readiness — v3 Classification

The v3 engine classifies each ticker into:

### **Watch List**
- Trend stack aligned (EMA9 > EMA20 > EMA50)  
- EMA slopes positive  
- PCA1 > 0  
- PCA1_slope > 0  
- Structural momentum forming  

### **Not Watch List**
- Trend stack aligned  
- EMA slopes weakening  
- PCA1 positive or neutral  
- Momentum fading  

### **Crossing Soon**
- EMA9 and EMA20 compressing  
- Trend transition forming  
- PCA1 positive  
- Momentum coil pattern  

### **Setup Only**
- Trend not aligned  
- Momentum weak  
- PCA1 neutral or negative  
- Observation only  

These labels define **structural readiness**, not intraday entry timing.

---

# Trader Workflow — SPC v3

The trader uses v3 in a **two‑stage workflow**:

### **Stage A — Structural Scan (Page 1)**
Identify:
- Trend stack alignment  
- PCA1 strength  
- EMA slope stability  
- RVOL support  
- Price tier (>50 recommended in choppy regimes)

### **Stage B — Intraday Confirmation (Page 2)**
Confirm:
1. EMA9 reclaim  
2. VWAP reclaim  
3. PCA1_slope > 0  
4. RVOL expansion  
5. Higher‑low formation  
6. Clean intraday structure  

This ensures **location → timing → execution** discipline.

---

# Technical Glossary

### **EMA Stack**
Defines directional bias across short, medium, and long‑term trend layers.

### **PCA1**
Primary momentum cluster capturing multi‑indicator acceleration.

### **PCA1_slope**
Real‑time acceleration of momentum.

### **VWAP**
Institutional cost basis; reclaim indicates trend validation.

### **RVOL**
Relative volume; detects institutional participation.

### **ATR%**
Volatility availability; ensures friction clearance.

### **Trend Stack**
EMA9 > EMA20 > EMA50 alignment indicating structural trend strength.

---
""")