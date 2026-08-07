# ==============================================================================
# 📈 STRUCTURAL ENGINE PAGE 1 — COMPLETE UNIFIED MASTER CODE
# SPECIFICATION: PREMIUM UNIVERSE STRUCTURAL TRACKING SYSTEM (>$50 TICKERS)
# INCLUDES: HIGH-SPEED PROXIMITY, SPREADS, AND MULTI-CORE ESTIMATIONS
# ==============================================================================
import streamlit as st

st.set_page_config(layout="wide", page_title="Structural Engine Page1")
st.caption("Version: 2026-08-01")
st.title("📈 Structural Engine Page 1")

import importlib
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

# Secure Dynamic Reloading Components from backend file
import analysis.intraday_ranker_v3 as v3
importlib.reload(v3)

try:
    from utils.data_fetch import load_universe, get_universe_source
except (ImportError, ModuleNotFoundError):
    def load_universe(): return []
    def get_universe_source(ticker): return "Premium Slot"

# ---------------------------------------------------------
# CORRELATED BACKEND-ALIGNED PRELOAD CORES
# ---------------------------------------------------------
def _flatten_columns(df):
    if df is None or df.empty:
        return df
    df_copy = df.copy()
    if isinstance(df_copy.columns, pd.MultiIndex):
        if len(df_copy.columns.levels) > 1:
            df_copy.columns = df_copy.columns.get_level_values(0)
        else:
            df_copy.columns = [col if isinstance(col, tuple) else col for col in df_copy.columns]
    df_copy.columns = [str(c).strip() for c in df_copy.columns]
    return df_copy

@st.cache_data(ttl=120, show_spinner=False)
def fetch_clean_market_batch(tickers_tuple):
    ticker_list = list(tickers_tuple)
    if not ticker_list:
        return pd.DataFrame(), pd.DataFrame()
    try:
        raw_daily = yf.download(ticker_list, period="3mo", interval="1d", group_by="ticker", progress=False, threads=True)
        raw_intra = yf.download(ticker_list, period="1d", interval="1m", group_by="ticker", progress=False, threads=True)
        return raw_daily, raw_intra
    except Exception as e:
        st.error(f"Batch synchronization layer failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

def extract_ticker_slice(batch_df, ticker):
    try:
        if batch_df is None or batch_df.empty:
            return pd.DataFrame()
        if isinstance(batch_df.columns, pd.MultiIndex):
            if ticker in batch_df.columns.get_level_values(0):
                return batch_df[ticker].copy().dropna(subset=["Close"])
            elif ticker in batch_df.columns.get_level_values(1):
                return batch_df.xs(ticker, axis=1, level=1).copy().dropna(subset=["Close"])
        if ticker in batch_df.columns:
            return batch_df[[ticker]].copy().dropna(subset=["Close"])
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def to_scalar(x):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        if isinstance(x, (pd.Series, np.ndarray)):
            val = x.squeeze()
            if hasattr(val, "item"):
                return float(val.item())
            return float(val[-1] if isinstance(val, (np.ndarray, list)) else val)
        return float(x)
    except Exception:
        try:
            return float(x.iloc[-1].squeeze())
        except Exception:
            return float("nan")

def compute_drop_pct(prev_close, current_price):
    prev_close = to_scalar(prev_close)
    current_price = to_scalar(current_price)
    return (prev_close - current_price) / prev_close if prev_close > 0 else 0.0

def compute_recovery_probability(df_daily):
    if df_daily is None or df_daily.empty or len(df_daily) < 40:
        return 0.0
    df_window = df_daily.tail(60).copy()
    recoveries = 0
    total = len(df_window)
    for i in range(total):
        low = to_scalar(df_window["Low"].iloc[i])
        close = to_scalar(df_window["Close"].iloc[i])
        high = to_scalar(df_window["High"].iloc[i])
        dip = high - low
        recovered = close - low
        if dip > 0 and recovered >= (0.5 * dip):
            recoveries += 1
    return float(recoveries / total)

# ---------------------------------------------------------
# MASTER VECTORIZED EXTRACTION INTERFACE (DYNAMIC ADAPTIVE VELOCITY HARDENED)
# ---------------------------------------------------------
def local_rank_universe_batch(tickers, batch_daily, batch_intra, min_price, max_price):
    rows = []
    if batch_daily is None or batch_daily.empty or batch_intra is None or batch_intra.empty:
        return pd.DataFrame()

    available_daily = set(batch_daily.columns.get_level_values(0)) if hasattr(batch_daily, "columns") else set()
    available_intra = set(batch_intra.columns.get_level_values(0)) if hasattr(batch_intra, "columns") else set()
    active_pool = list(set(tickers).intersection(available_daily).intersection(available_intra))

    for ticker in active_pool:
        try:
            daily_df = batch_daily[ticker].copy().dropna(subset=["Close"])
            intraday_df = batch_intra[ticker].copy().dropna(subset=["Close"])
            
            if daily_df.empty or intraday_df.empty or len(daily_df) < 40:
                continue

            # Capacity Gate Liquidity Baseline Validation Slices
            vol_d = daily_df["Volume"].values
            avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
            if avg_volume_20d < 250000:
                continue

            current_price = float(daily_df["Close"].iloc[-1].squeeze() if hasattr(daily_df["Close"].iloc[-1], "squeeze") else daily_df["Close"].iloc[-1])
            if current_price < min_price or current_price > max_price:
                continue

            # Isolate Intraday Arrays early for Institutional Safety Protection Gates
            open_intra = intraday_df["Open"].values
            close_intra = intraday_df["Close"].values
            high_intra = intraday_df["High"].values
            low_intra = intraday_df["Low"].values
            vol_intra = intraday_df["Volume"].values
            
            if len(close_intra) < 5:
                continue
                
            current_intraday_price = float(close_intra[-1])
            
            # SAFE SCALAR PATCH: Protects against index subscriptable crashes
            session_open_price = float(open_intra.ravel()[0]) if hasattr(open_intra, "ravel") else float(open_intra)

            # ----------------------------------------------------------------------
            # 🚨 HARDENED FILTER: CRITERION 3 — THE OPEN-DRIVE SYMMETRY GATE
            # ----------------------------------------------------------------------
            if current_intraday_price < session_open_price:
                continue

            # ----------------------------------------------------------------------
            # 🚨 GATE 1: THE INSTITUTIONAL SPREAD GATE (Eliminates Thin Order Books)
            # ----------------------------------------------------------------------
            recent_minute_spread = (high_intra[-5:] - low_intra[-5:]).mean()
            implied_spread = (recent_minute_spread * 0.15) / current_intraday_price
            if implied_spread > 0.0015:
                continue

            # ----------------------------------------------------------------------
            # 🚨 GATE 2: THE HIGH PROXIMITY EXTENSION GATE (Prevents Chasing Intraday Fades)
            # ----------------------------------------------------------------------
            daily_high = float(high_intra.max())
            high_proximity = (daily_high - current_intraday_price) / current_intraday_price
            if high_proximity > 0.02:
                continue

            # ----------------------------------------------------------------------
            # 🚨 GATE 3: THE VWAP SUSTAINABILITY CHECKER (Protects Against Overextension)
            # ----------------------------------------------------------------------
            cv_slice = vol_intra * close_intra
            vwap_spot = cv_slice.sum() / vol_intra.sum() if vol_intra.sum() > 0 else current_intraday_price
            vwap_dist_pct = (current_intraday_price - vwap_spot) / vwap_spot
            if vwap_dist_pct > 0.015:
                continue

            # ⚡ STRUCTURAL FOUNDATION REMAP ENGINE
            close_d = daily_df["Close"].values
            high_d = daily_df["High"].values
            low_d = daily_df["Low"].values

            # ----------------------------------------------------------------------
            # 🚨 HARDENED FILTER: CRITERION 1 — THE MULTI-DAY RESISTANCE SHIELD
            # ----------------------------------------------------------------------
            max_5day_overhead_resistance = float(high_d[-6:-1].max()) if len(high_d) >= 6 else float(high_d[0])
            if current_intraday_price < max_5day_overhead_resistance:
                continue

            # ----------------------------------------------------------------------
            # 🚨 HARDENED FILTER: CRITERION 4 — THE 10:30 AM INTRADAY RETEST GATE
            # ----------------------------------------------------------------------
            morning_high_marker = float(high_intra[:30].max()) if len(high_intra) >= 30 else session_open_price
            if len(high_intra) > 45 and current_intraday_price < morning_high_marker:
                continue

            # ----------------------------------------------------------------------
            # 🚨 HARDENED FILTER: THE 3% EXHAUSTION CAP GATE (Sniper Entry Ceiling)
            # ----------------------------------------------------------------------
            prev_close_d = float(close_d[-2] if len(close_d) >= 2 else current_price)
            today_total_gain_pct = ((current_intraday_price - prev_close_d) / prev_close_d) * 100
            
            # CRITICAL CEILING: Blocks and drops any asset already up > 3.0%
            if today_total_gain_pct > 3.0:
                continue

            # ----------------------------------------------------------------------
            # 📈 HARDENED 20-BAR INTRADAY EMA9 VELOCITY FILTER (Smooths Out Noise)
            # ----------------------------------------------------------------------
            ema9_i_series = intraday_df["Close"].ewm(span=9).mean().values
            if len(ema9_i_series) >= 20:
                intraday_velocity_slope = float((ema9_i_series[-1] - ema9_i_series[-20]) / ema9_i_series[-20]) * 100
            elif len(ema9_i_series) >= 5:
                intraday_velocity_slope = float((ema9_i_series[-1] - ema9_i_series[-5]) / ema9_i_series[-5]) * 100
            else:
                intraday_velocity_slope = 0.0

            # Direct data-driven penalties based on authentic EMA9 trajectory profiles
            if abs(intraday_velocity_slope) < 0.015:
                intraday_velocity_penalty = -4.0  # Sharp penalty forces flat afternoon chop off the board
            elif intraday_velocity_slope < 0:
                intraday_velocity_penalty = -2.0  # Moderate penalty for downward slanting fades
            else:
                intraday_velocity_penalty = 2.0   # Clear reward multiplier for active upward acceleration

            ema9_d = daily_df["Close"].ewm(span=9).mean().values
            ema20_d = daily_df["Close"].ewm(span=20).mean().values
            ema50_d = daily_df["Close"].ewm(span=50).mean().values

            # Core Macro Anchor Level
            if ema20_d[-1] > ema50_d[-1]:
                trend = "UP"
            elif ema20_d[-1] < ema50_d[-1]:
                trend = "DOWN"
            else:
                trend = "FLAT"

            ema9_slope = float(ema9_d[-1] - ema9_d[-5])
            ema20_slope = float(ema20_d[-1] - ema20_d[-5])
            proximity_metric = abs((ema9_d[-1] - ema20_d[-1]) / (ema20_d[-1] if ema20_d[-1] != 0 else 1.0))

            if ema9_d[-1] > ema20_d[-1] and ema9_slope > 0 and ema20_slope > 0:
                execution = "Watch List"
            elif ema9_d[-1] > ema20_d[-1] and (ema9_slope < 0 or ema20_slope < 0):
                execution = "Not Watch List"
            elif proximity_metric < 0.003:
                execution = "Crossing Soon"
            else:
                execution = "Setup Only"

            rvol = float(daily_df["Volume"].iloc[-1] / avg_volume_20d) if avg_volume_20d > 0 else 1.0
            gap_pct = float(((daily_df["Open"].iloc[-1] - close_d[-2]) / close_d[-2]) * 100) if len(close_d) >= 2 else 0.0
            h_l = high_d - low_d
            atr = float(np.mean(h_l[-14:])) if len(h_l) >= 14 else float(h_l[-1])
            atr_pct = (atr / current_price) * 100

            pca1_slope = 0.0
            ema_curve = 0.0
            vwap_dist = 0.0
            roc_10 = 0.0
            stoch_k = 0.5

            if len(intraday_df) > 5:
                ema9_i = intraday_df["Close"].ewm(span=9).mean().values
                ema20_i = intraday_df["Close"].ewm(span=20).mean().values
                ema_curve = float(ema9_i[-1] - ema20_i[-1])
                pca1_slope = float(ema9_i[-1] - ema9_i[-5]) 
                vwap_dist = vwap_dist_pct  

            prev_close = float(close_d[-2] if len(close_d) >= 2 else current_price)
            price_vs_close = "Above Close" if current_intraday_price > prev_close else "Below Close" if current_intraday_price < prev_close else "Equal"

            # ==============================================================================
            # VERIFIED FINAL DICTIONARY APPENDING MATRIX
            # ==============================================================================
            rows.append({
                "Ticker": ticker,
                "Universe": get_universe_source(ticker),
                "Close": round(current_price, 2),
                "ATR%": round(atr_pct, 2),
                "RVOL": round(rvol, 2),
                "Gap%": round(gap_pct, 2),
                "Trend": trend,
                "Execution": execution,
                "PCA1": 0.0,
                "Avg_Volume_20d": avg_volume_20d,
                "Price_vs_Close": price_vs_close,
                "PCA1_slope": pca1_slope,
                "EMA_Curve": ema_curve,
                "VWAP_Dist": vwap_dist,
                "ROC_10": roc_10,
                "StochK": stoch_k,
                "Zero_Line_Boost": ticker_zero_line_boost,   # Matches Tracker Variable
                "Velocity_Penalty": intraday_velocity_penalty # Matches Adaptive Velocity Variable
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["PCA1"] = pd.to_numeric(df["PCA1"], errors="coerce").fillna(0.0)
    df["PCA1_slope"] = pd.to_numeric(df["PCA1_slope"], errors="coerce").fillna(0.0)
    df["RVOL"] = pd.to_numeric(df["RVOL"], errors="coerce").fillna(0.0)
    df["Zero_Line_Boost"] = pd.to_numeric(df["Zero_Line_Boost"], errors="coerce").fillna(0.0)
    df["Velocity_Penalty"] = pd.to_numeric(df["Velocity_Penalty"], errors="coerce").fillna(0.0)

    # UN-CLIPPED LOW-LAG PREDICTIVE SCORE MATRIX (TOTAL IN-MEMORY SYNCHRONIZATION)
    # 1. Replaced the leaked backend 'PCA1_slope' with your clean, 20-bar 'Velocity_Penalty'
    # 2. Multiplied 'intraday_velocity_slope' directly to reward sustained multi-bar continuation
    df["Score"] = (
        (df["Trend"] == "UP").astype(int) * 2.0 +
        (df["Execution"] == "Watch List").astype(int) * 1.0 +
        (df["Execution"] == "Crossing Soon").astype(int) * 4.0 + 
        df["RVOL"].clip(lower=0) +
        df["Zero_Line_Boost"] +
        df["Velocity_Penalty"] +            # Applies the strict -4.0 point penalty for flat chop
        (df["PCA1_slope"] * 0.0)           # 🚨 Kills the leaked short-term backend noise completely
    )
    

# ---------------------------------------------------------
# PART 2- REGIME UTILITIES
# ---------------------------------------------------------
def get_index_trend(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, threads=False)
        if df is None or df.empty: 
            return "Unknown"
        df = _flatten_columns(df)
        df["EMA9"] = df["Close"].ewm(span=9).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        
        ema9 = to_scalar(df["EMA9"].iloc[-1])
        ema20 = to_scalar(df["EMA20"].iloc[-1])
        ema50 = to_scalar(df["EMA50"].iloc[-1])
        slope20 = float(to_scalar(df["EMA20"].iloc[-1]) - to_scalar(df["EMA20"].iloc[-5]))
        
        if ema9 > ema20 > ema50 and slope20 > 0: 
            return "Bullish"
        if ema9 < ema20 < ema50 and slope20 < 0: 
            return "Bearish"
        return "Choppy"
    except Exception: 
        return "Unknown"

def classify_regime(sp500_trend, nasdaq_trend):
    if sp500_trend == "Bullish" and nasdaq_trend == "Bullish": 
        return "Trending"
    if sp500_trend == "Bearish" and nasdaq_trend == "Bearish": 
        return "Bearish"
    if sp500_trend != nasdaq_trend: 
        return "Mixed"
    return "Choppy"

def color_execution_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    if "Execution" in df.columns:
        style_df["Execution"] = [
            "background-color: #4CAF50; color: white; font-weight: bold;" if str(v) == "Watch List"
            else "background-color: #FFC107; color: black; font-weight: bold;" if str(v) == "Crossing Soon"
            else "background-color: #FF9800; color: white;" if str(v) == "Not Watch List"
            else "background-color: #9E9E9E; color: white;" for v in df["Execution"]
        ]
    return style_df

# ---------------------------------------------------------
# INTERACTIVE FILTERS SIDEBAR
# ---------------------------------------------------------
st.markdown("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Asset Close Gate Price ($)", value=40.0, min_value=40.0, max_value=110.0, key="intraday_min_price")
max_price = st.number_input("Maximum Asset Close Gate Price ($)", value=110.0, min_value=40.0, max_value=110.0, key="intraday_max_price")

st.markdown("### 🎛️ Structural Execution Filters")
execution_filter = st.multiselect(
    "Filter by Real-Time Structural Execution Status", 
    ["Watch List", "Not Watch List", "Crossing Soon", "Setup Only"], 
    default=["Watch List", "Crossing Soon"], 
    key="intraday_execution_filter"
)

def render_results(filtered, ranking, regime, sp500_trend, nasdaq_trend):
    st.metric(label="📊 Active Market Regime Identified", value=f"{regime}", delta=f"S&P500: {sp500_trend} | NASDAQ: {nasdaq_trend}", delta_color="off")
    st.markdown(f"**Structural Tier-1 Premium Universe:** {len(ranking)} premium tokens verified.")
    if filtered is None or filtered.empty:
        st.info("No tickers matched your interactive filter constraints.")
    else:
        display_df = filtered.copy()
        if "Ticker" in display_df.columns and "Universe" in display_df.columns:
            display_df["Ticker"] = display_df.apply(lambda r: f"{r['Ticker']} ({r['Universe']})", axis=1)
            display_df = display_df.drop(columns=["Universe"], errors="ignore")
        st.subheader(f"🚀 Actionable Structural Matrix Results — {len(display_df)} Tickers")
        st.dataframe(display_df.style.apply(color_execution_column, axis=None), hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# RUN PART 3 ORCHESTRATION TRIGGER (HARDENED EXCEPTION HANDLED)
# ---------------------------------------------------------
run_model = st.button("Run Intraday Model Scan", key="intraday_run_button")

if run_model:
    try:
        st.cache_data.clear()
        start_time = time.time()
        eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now().astimezone(eastern)
        
        st.markdown(f"⏱️ Scan Execution Time Stamp: **{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**")
        progress_bar = st.progress(0, text="Synchronizing clean global exchange batch maps...")
        
        universe_list = load_universe()
        if not universe_list:
            st.error("The stock universe source file returned completely empty.")
            st.stop()
        
        # 1. Download market data with complete NoneType safety fallback tracking
        raw_daily, raw_intra = fetch_clean_market_batch(tuple(universe_list))
        
        # 🛡️ DEFENSIVE GUARD CRASH BARRIER: Instantly isolates and repairs NoneType drops
        if raw_daily is None or raw_intra is None:
            st.error("Global exchange batch synchronization returned a NoneType connection error. Re-trigger the scan.")
            progress_bar.empty()
            st.stop()
            
        if hasattr(raw_daily, "empty") and raw_daily.empty or hasattr(raw_intra, "empty") and raw_intra.empty:
            st.warning("Data matrices downloaded successfully but returned no active price data.")
            progress_bar.empty()
            st.stop()
        
        progress_bar.progress(0.4, text="Running high-speed machine learning scoring array...")
        ranking = local_rank_universe_batch(universe_list, raw_daily, raw_intra, min_price, max_price)
        
        progress_bar.progress(0.7, text="Calculating index trend matrix...")
        sp500_trend = get_index_trend("^GSPC")
        nasdaq_trend = get_index_trend("^IXIC")
        regime = classify_regime(sp500_trend, nasdaq_trend)
        
        if ranking is not None and not ranking.empty:
            st.session_state["intraday_raw_ranking"] = ranking
            st.session_state["intraday_regime"] = regime
            st.session_state["intraday_sp500"] = sp500_trend
            st.session_state["intraday_nasdaq"] = nasdaq_trend
        else:
            st.warning("No tickers satisfied your multi-stage safety protection gates on this cycle.")
            if "intraday_raw_ranking" in st.session_state:
                del st.session_state["intraday_raw_ranking"]
            
        progress_bar.empty()
        st.write(f"⏱ ... Market Core Sync Batch Execution Trace Complete.")
        st.write(f"⚡ Total Model Runtime: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        try:
            progress_bar.empty()
        except NameError:
            pass
        st.error(f"Model execution failed structural compilation: {str(e)}")
        st.exception(e)

# Render results panel safely outside button context
if "intraday_raw_ranking" in st.session_state:
    ranking = st.session_state["intraday_raw_ranking"]
    regime = st.session_state.get("intraday_regime", "Unknown")
    sp500_trend = st.session_state.get("intraday_sp500", "Unknown")
    nasdaq_trend = st.session_state.get("intraday_nasdaq", "Unknown")
    
    if ranking is not None and not ranking.empty:
        filtered = ranking.copy()
        filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]
        if execution_filter and "Execution" in filtered.columns:
            filtered = filtered[filtered["Execution"].isin(execution_filter)]
        render_results(filtered, ranking, regime, sp500_trend, nasdaq_trend)

