# ==============================================================================
# 🚀 MOMENTUM ENGINE — UNIVERSAL MODEL + TOP5 TRACKER
# ==============================================================================
import streamlit as st

st.set_page_config(layout="wide", page_title="Momentum Model")
st.caption("Version: 2026-08-20")
st.title("🚀 Momentum Model — Universal Momentum Scanner")

import importlib
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import pytz

# ---------------------------------------------------------
# SHARED UTILITIES
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
        raw_daily = yf.download(ticker_list, period="3mo", interval="1d",
                                group_by="ticker", progress=False, threads=True)
        raw_intra = yf.download(ticker_list, period="1d", interval="1m",
                                group_by="ticker", progress=False, threads=True)
        return raw_daily, raw_intra
    except Exception as e:
        st.error(f"Batch synchronization layer failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

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

# ---------------------------------------------------------
# UNIVERSAL MOMENTUM ENGINE + TREND + DRIFT PENALTY
# ---------------------------------------------------------
def momentum_rank_universe_batch(tickers, batch_daily, batch_intra, min_price, max_price):
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

            # Basic liquidity + price bounds
            vol_d = daily_df["Volume"].values
            avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
            if avg_volume_20d < 250000:
                continue

            current_price = float(daily_df["Close"].iloc[-1])
            if current_price < min_price or current_price > max_price:
                continue

            # Intraday arrays
            open_intra = intraday_df["Open"].values
            close_intra = intraday_df["Close"].values
            high_intra = intraday_df["High"].values
            low_intra = intraday_df["Low"].values
            vol_intra = intraday_df["Volume"].values

            # Allow afternoon scans (Yahoo often returns fewer bars)
            if len(close_intra) < 5:
                continue

            current_intraday_price = float(close_intra[-1])
            session_open_price = float(open_intra[0])

            # Daily context
            close_d = daily_df["Close"].values
            high_d = daily_df["High"].values
            prev_close_d = float(close_d[-2] if len(close_d) >= 2 else current_price)
            yesterday_high = float(high_d[-2] if len(high_d) >= 2 else high_d[-1])

            current_vs_close_pct = ((current_intraday_price - prev_close_d) / prev_close_d) * 100

            # ---------------------------------------------------------
            # ⭐ A. GAP QUALITY (UNIVERSAL)
            # ---------------------------------------------------------
            gap_pct = ((session_open_price - prev_close_d) / prev_close_d) * 100

            if gap_pct > 20.0:
                gap_score = 0.0
            elif gap_pct > 12.0:
                gap_score = 1.0
            elif gap_pct > 8.0:
                gap_score = 2.0
            elif gap_pct > 3.0:
                gap_score = 3.0
            elif gap_pct > 1.0:
                gap_score = 2.0
            elif gap_pct > 0.0:
                gap_score = 1.0
            else:
                gap_score = 0.0

            # ---------------------------------------------------------
            # ⭐ B. VELOCITY (EMA9 SLOPE — UNIVERSAL)
            # ---------------------------------------------------------
            ema9_i_series = intraday_df["Close"].ewm(span=9).mean().values
            if len(ema9_i_series) >= 5:
                ema9_slope_10 = float((ema9_i_series[-1] - ema9_i_series[-5]) / ema9_i_series[-5]) * 100
            else:
                ema9_slope_10 = 0.0

            if ema9_slope_10 > 0.60:
                velocity_score = 4.0
            elif ema9_slope_10 > 0.30:
                velocity_score = 3.0
            elif ema9_slope_10 > 0.15:
                velocity_score = 2.0
            elif ema9_slope_10 > 0.00:
                velocity_score = 1.0
            else:
                velocity_score = 0.0

            # ---------------------------------------------------------
            # ⭐ C. RVOL IGNITION (UNIVERSAL)
            # ---------------------------------------------------------
            intraday_total_volume = float(vol_intra.sum())
            rvol = float(intraday_total_volume / avg_volume_20d) if avg_volume_20d > 0 else 1.0

            if rvol > 5.0:
                rvol_score = 4.0
            elif rvol > 3.0:
                rvol_score = 3.0
            elif rvol > 2.0:
                rvol_score = 2.0
            elif rvol > 1.2:
                rvol_score = 1.0
            else:
                rvol_score = 0.0

            # ---------------------------------------------------------
            # ⭐ D. BREAKOUT QUALITY (UNIVERSAL)
            # ---------------------------------------------------------
            premarket_window = intraday_df.head(10)
            premarket_high = float(premarket_window["High"].max()) if not premarket_window.empty else session_open_price

            first_15m_window = intraday_df.head(15)
            first_15m_high = float(first_15m_window["High"].max()) if not first_15m_window.empty else session_open_price

            breakout_score = 0.0
            if current_intraday_price > premarket_high:
                breakout_score += 3.0
            if current_intraday_price > first_15m_high:
                breakout_score += 3.0
            if current_intraday_price > yesterday_high:
                breakout_score += 2.0

            # ---------------------------------------------------------
            # ⭐ E. VOLATILITY COHERENCE (UNIVERSAL)
            # ---------------------------------------------------------
            intraday_range_pct = float((high_intra[-1] - low_intra[-1]) / current_intraday_price * 100)

            if intraday_range_pct > 1.5:
                volatility_score = 0.0
            elif intraday_range_pct > 0.8:
                volatility_score = 2.0
            elif intraday_range_pct > 0.4:
                volatility_score = 1.0
            else:
                volatility_score = 0.0

            # ---------------------------------------------------------
            # ⭐ F. DAILY EMA STACK (UNIVERSAL)
            # ---------------------------------------------------------
            ema9_d = daily_df["Close"].ewm(span=9).mean().values
            ema20_d = daily_df["Close"].ewm(span=20).mean().values
            ema50_d = daily_df["Close"].ewm(span=50).mean().values

            if ema9_d[-1] > ema20_d[-1] > ema50_d[-1]:
                ema_stack_score = 2.0
            else:
                ema_stack_score = 0.0

            # ---------------------------------------------------------
            # ⭐ G. VWAP RELATIONSHIP (UNIVERSAL)
            # ---------------------------------------------------------
            cv_slice = vol_intra * close_intra
            vwap_spot = cv_slice.sum() / vol_intra.sum() if vol_intra.sum() > 0 else current_intraday_price

            if current_intraday_price > vwap_spot:
                vwap_reclaim_score = 2.0
            else:
                vwap_reclaim_score = 0.0

            # ---------------------------------------------------------
            # ⭐ TREND (INTRADAY EMA STACK)
            # ---------------------------------------------------------
            ema9_i = intraday_df["Close"].ewm(span=9).mean().iloc[-1]
            ema20_i = intraday_df["Close"].ewm(span=20).mean().iloc[-1]
            ema50_i = intraday_df["Close"].ewm(span=50).mean().iloc[-1]

            if ema9_i > ema20_i > ema50_i:
                trend = "UP"
            elif ema9_i < ema20_i < ema50_i:
                trend = "DOWN"
            else:
                trend = "FLAT"

            # ---------------------------------------------------------
            # ⭐ H. DRIFT PENALTY (UNIVERSAL)
            # ---------------------------------------------------------
            drift_flags = 0

            if abs(ema9_slope_10) < 0.05:
                drift_flags += 1
            if rvol < 1.2:
                drift_flags += 1
            if intraday_range_pct < 0.4:
                drift_flags += 1
            vwap_dist_pct = abs(current_intraday_price - vwap_spot) / current_intraday_price * 100
            if vwap_dist_pct < 0.3:
                drift_flags += 1

            if drift_flags == 0:
                drift_multiplier = 1.0
            elif drift_flags == 1:
                drift_multiplier = 0.8
            elif drift_flags == 2:
                drift_multiplier = 0.6
            elif drift_flags == 3:
                drift_multiplier = 0.4
            else:
                drift_multiplier = 0.3

            # ---------------------------------------------------------
            # ⭐ FINAL UNIVERSAL MOMENTUM SCORE
            # ---------------------------------------------------------
            base_momentum_score = (
                gap_score +
                velocity_score +
                rvol_score +
                breakout_score +
                volatility_score +
                ema_stack_score +
                vwap_reclaim_score
            )

            momentum_score = base_momentum_score * drift_multiplier

            rows.append({
                "Ticker": ticker,
                "Trend": trend,
                "Close": round(current_price, 2),
                "Gap%": round(gap_pct, 2),
                "Current_vs_Close%": round(current_vs_close_pct, 2),
                "RVOL": round(rvol, 2),
                "EMA9_Slope_10": round(ema9_slope_10, 3),
                "Intraday_Range%": round(intraday_range_pct, 2),
                "VWAP": round(vwap_spot, 2),
                "Momentum_Score": round(momentum_score, 2)
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Momentum_Score"] = pd.to_numeric(df["Momentum_Score"], errors="coerce").fillna(0.0)
    return df

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.markdown("### 🔍 Price Boundaries Filter (Momentum Universe)")
min_price = st.number_input("Minimum Asset Price ($)", value=40.0, min_value=40.0, max_value=110.0)
max_price = st.number_input("Maximum Asset Price ($)", value=110.0, min_value=40.0, max_value=110.0)

st.markdown("### 🎛️ Momentum Score Filter")
min_momentum_score = st.number_input("Minimum Momentum Score", value=4.0, min_value=0.0, max_value=30.0, step=0.5)

# ---------------------------------------------------------
# COLOR CODING
# ---------------------------------------------------------
def color_momentum_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    if "Momentum_Score" in df.columns:
        style_df["Momentum_Score"] = [
            "background-color: #4CAF50; color: white; font-weight: bold;" if v >= 12
            else "background-color: #FFC107; color: black; font-weight: bold;" if v >= 8
            else "background-color: #F44336; color: white;" for v in df["Momentum_Score"]
        ]
    return style_df

def color_trend_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    if "Trend" in df.columns:
        for i in range(len(df)):
            t = df.iloc[i]["Trend"]
            if t == "UP":
                style_df.loc[df.index[i], "Trend"] = "background-color:#4CAF50;color:white;font-weight:bold;"
            elif t == "FLAT":
                style_df.loc[df.index[i], "Trend"] = "background-color:#FFC107;color:black;font-weight:bold;"
            else:
                style_df.loc[df.index[i], "Trend"] = "background-color:#F44336;color:white;"
    return style_df

# ---------------------------------------------------------
# RUN MOMENTUM ENGINE
# ---------------------------------------------------------
run_momentum = st.button("Run Momentum Model Scan")

if run_momentum:
    try:
        st.cache_data.clear()
        start_time = time.time()
        eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now().astimezone(eastern)

        st.markdown(f"⏱️ Momentum Scan Time Stamp: **{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**")

        progress_bar = st.progress(0, text="Synchronizing clean global exchange batch maps for momentum engine...")

        from utils.data_fetch import load_universe
        universe_list = load_universe()

        raw_daily, raw_intra = fetch_clean_market_batch(tuple(universe_list))

        progress_bar.progress(0.4, text="Running universal momentum scoring array...")
        ranking = momentum_rank_universe_batch(universe_list, raw_daily, raw_intra, min_price, max_price)

        if ranking is not None and not ranking.empty:
            st.session_state["momentum_raw_ranking"] = ranking

        progress_bar.empty()
        st.write(f"⏱ ... Momentum Engine Execution Trace Complete.")
        st.write(f"⚡ Total Momentum Model Runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        try:
            progress_bar.empty()
        except NameError:
            pass
        st.error(f"Momentum model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# RENDER RESULTS PANEL
# ---------------------------------------------------------
if "momentum_raw_ranking" in st.session_state:
    ranking = st.session_state["momentum_raw_ranking"]

    if ranking is not None and not ranking.empty:
        filtered = ranking.copy()
        filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]
        filtered = filtered[filtered["Momentum_Score"] >= min_momentum_score]

        if filtered.empty:
            st.info("No tickers matched your momentum score and price filters.")
        else:
            display_df = filtered.copy()
            display_df = display_df.sort_values(by="Momentum_Score", ascending=False)

            st.subheader(f"🔥 Universal Momentum Matrix — {len(display_df)} Tickers")
            st.dataframe(
                display_df.style.apply(color_momentum_column, axis=None)
                                 .apply(color_trend_column, axis=None),
                hide_index=True,
                use_container_width=True
            )

            # ---------------------------------------------------------
            # ⭐ RESET MOMENTUM TRACKER BUTTON
            # ---------------------------------------------------------
            if st.button("Reset Momentum Tracker"):
                st.session_state["momentum_history"] = []
                st.session_state["momentum_history_date"] = date.today()
                st.success("Momentum tracker has been reset.")

            # ---------------------------------------------------------
            # ⭐ TOP 5 MOMENTUM HISTORY TRACKER
            # ---------------------------------------------------------
            if "momentum_history_date" not in st.session_state:
                st.session_state["momentum_history_date"] = date.today()

            if "momentum_history" not in st.session_state:
                st.session_state["momentum_history"] = []

            if st.session_state["momentum_history_date"] != date.today():
                st.session_state["momentum_history"] = []
                st.session_state["momentum_history_date"] = date.today()

            top5 = display_df.head(5).copy()
            top5["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.session_state["momentum_history"].append(top5)

            history_df = pd.concat(st.session_state["momentum_history"], ignore_index=True)

            st.subheader("📊 Momentum Timeline — Top 5 per Scan")

            st.dataframe(
                history_df.style.apply(color_momentum_column, axis=None)
                                .apply(color_trend_column, axis=None),
                hide_index=True,
                use_container_width=True
            )
