# ==============================================================================
# 💹 INSTITUTIONAL DIP-BUY MODEL (PAGE 3)
# SPEC: Uses Page2 backbone, but targets intraday dips with upward momentum
# ==============================================================================
import streamlit as st

st.set_page_config(layout="wide", page_title="Institutional Dip-Buy Model")
st.caption("Version: 2026-08-27 — Institutional Backbone + Intraday Dip-Reversal Scanner")
st.title("💹 Institutional Dip-Buy Opportunities")

import importlib
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

# Reuse universe utilities
try:
    from utils.data_fetch import load_universe, get_universe_source
except (ImportError, ModuleNotFoundError):
    def load_universe(): return []
    def get_universe_source(ticker): return "Premium Slot"

# ---------------------------------------------------------
# SHARED UTILITIES (aligned with Page2)
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
        raw_daily = yf.download(
            ticker_list,
            period="3mo",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True
        )
        raw_intra = yf.download(
            ticker_list,
            period="1d",
            interval="1m",
            group_by="ticker",
            progress=False,
            threads=True
        )
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
# PAGE 3 CORE: DIP-BUY RANKER
# ---------------------------------------------------------
def local_rank_dip_buy_universe(tickers, batch_daily, batch_intra, min_price, max_price):
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

            # Institutional liquidity baseline (same as Page2)
            vol_d = daily_df["Volume"].values
            avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
            if avg_volume_20d < 250000:
                continue

            current_price = float(
                daily_df["Close"].iloc[-1].squeeze()
                if hasattr(daily_df["Close"].iloc[-1], "squeeze")
                else daily_df["Close"].iloc[-1]
            )
            if current_price < min_price or current_price > max_price:
                continue

            # Intraday arrays
            open_intra = intraday_df["Open"].values
            close_intra = intraday_df["Close"].values
            high_intra = intraday_df["High"].values
            low_intra = intraday_df["Low"].values
            vol_intra = intraday_df["Volume"].values

            if len(close_intra) < 20:
                continue

            current_intraday_price = float(close_intra[-1])
            session_open_price = float(open_intra.ravel()[0]) if hasattr(open_intra, "ravel") else float(open_intra)

            # --- Institutional backbone trend (same as Page2) ---
            close_d = daily_df["Close"].values
            high_d = daily_df["High"].values
            low_d = daily_df["Low"].values

            ema9_d = daily_df["Close"].ewm(span=9).mean().values
            ema20_d = daily_df["Close"].ewm(span=20).mean().values
            ema50_d = daily_df["Close"].ewm(span=50).mean().values

            if ema20_d[-1] > ema50_d[-1]:
                trend = "UP"
            elif ema20_d[-1] < ema50_d[-1]:
                trend = "DOWN"
            else:
                trend = "FLAT"

            # Require institutional UP trend backbone
            if trend != "UP":
                continue

            # --- Dip condition: price below yesterday's close ---
            prev_close_d = float(close_d[-2] if len(close_d) >= 2 else current_price)
            dip_pct = (prev_close_d - current_intraday_price) / prev_close_d * 100.0
            is_dip = current_intraday_price < prev_close_d and 0.2 <= dip_pct <= 3.0
            if not is_dip:
                continue

            # --- Intraday momentum: EMA9 slope up ---
            ema9_i_series = intraday_df["Close"].ewm(span=9).mean().values
            ema20_i_series = intraday_df["Close"].ewm(span=20).mean().values

            if len(ema9_i_series) >= 20:
                intraday_velocity_slope = float(
                    (ema9_i_series[-1] - ema9_i_series[-20]) / ema9_i_series[-20]
                ) * 100
            else:
                intraday_velocity_slope = float(
                    (ema9_i_series[-1] - ema9_i_series[0]) / ema9_i_series[0]
                ) * 100

            # Require positive intraday slope (reversal from dip)
            if intraday_velocity_slope <= 0.10:
                continue

            # --- VWAP proximity (reuse Page2 logic but softer) ---
            cv_slice = vol_intra * close_intra
            vwap_spot = cv_slice.sum() / vol_intra.sum() if vol_intra.sum() > 0 else current_intraday_price
            vwap_dist_pct = (current_intraday_price - vwap_spot) / vwap_spot * 100.0

            # For dip-buy, we WANT price near or slightly below VWAP, not blowout above
            if vwap_dist_pct > 2.0:
                continue

            # --- Multi-day context: price not breaking structure ---
            max_3day_high = float(high_d[-4:-1].max()) if len(high_d) >= 4 else float(high_d[-1])
            min_3day_low = float(low_d[-4:-1].min()) if len(low_d) >= 4 else float(low_d[-1])

            # Require current price above 3-day midpoint to avoid deep breakdowns
            mid_3day = (max_3day_high + min_3day_low) / 2.0
            if current_intraday_price < mid_3day:
                continue

            # --- Execution label for Page3 ---
            ema9_slope_d = float(ema9_d[-1] - ema9_d[-5])
            ema20_slope_d = float(ema20_d[-1] - ema20_d[-5])

            if ema9_slope_d > 0 and ema20_slope_d > 0:
                execution = "Dip-Buy Candidate"
            else:
                execution = "Weak Dip"

            # --- Risk metrics ---
            h_l = high_d - low_d
            atr = float(np.mean(h_l[-14:])) if len(h_l) >= 14 else float(h_l[-1])
            atr_pct = (atr / current_price) * 100.0

            rvol = float(daily_df["Volume"].iloc[-1] / avg_volume_20d) if avg_volume_20d > 0 else 1.0
            gap_pct = float(((daily_df["Open"].iloc[-1] - prev_close_d) / prev_close_d) * 100.0) if len(close_d) >= 2 else 0.0

            price_vs_close = "Below Close" if current_intraday_price < prev_close_d else "Above Close"

            rows.append({
                "Ticker": ticker,
                "Universe": get_universe_source(ticker),
                "Trend": trend,
                "Execution": execution,
                "Close": round(current_price, 2),
                "Intraday_Price": round(current_intraday_price, 2),
                "Dip%_vs_YClose": round(dip_pct, 2),
                "VWAP_Dist%": round(vwap_dist_pct, 2),
                "Intraday_EMA9_Slope%": round(intraday_velocity_slope, 2),
                "ATR%": round(atr_pct, 2),
                "RVOL": round(rvol, 2),
                "Gap%": round(gap_pct, 2),
                "Price_vs_Close": price_vs_close
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Simple dip-buy score
    df["Score"] = (
        (df["Execution"] == "Dip-Buy Candidate").astype(int) * 3.0 +
        df["RVOL"].clip(lower=0) +
        (df["Intraday_EMA9_Slope%"] / 2.0) -
        df["Dip%_vs_YClose"].clip(lower=0) / 2.0
    )

    df = df.sort_values("Score", ascending=False)
    return df

# ---------------------------------------------------------
# UI FILTERS
# ---------------------------------------------------------
st.markdown("### 🔍 Price Boundaries Filter")
min_price = st.number_input(
    "Minimum Asset Close Gate Price ($)",
    value=40.0,
    min_value=40.0,
    max_value=110.0,
    key="page3_min_price"
)
max_price = st.number_input(
    "Maximum Asset Close Gate Price ($)",
    value=110.0,
    min_value=40.0,
    max_value=110.0,
    key="page3_max_price"
)

st.markdown("### 🎛️ Execution Filter")
execution_filter = st.multiselect(
    "Filter by Dip-Buy Execution Status",
    ["Dip-Buy Candidate", "Weak Dip"],
    default=["Dip-Buy Candidate"],
    key="page3_execution_filter"
)

def color_execution_column(df):
    style_df = pd.DataFrame('', index=df.index, columns=df.columns)
    if "Execution" in df.columns:
        style_df["Execution"] = [
            "background-color: #4CAF50; color: white; font-weight: bold;" if str(v) == "Dip-Buy Candidate"
            else "background-color: #FFC107; color: black; font-weight: bold;" for v in df["Execution"]
        ]
    return style_df

def render_results(filtered):
    if filtered is None or filtered.empty:
        st.info("No dip-buy candidates matched your constraints on this cycle.")
    else:
        display_df = filtered.copy()
        if "Ticker" in display_df.columns and "Universe" in display_df.columns:
            display_df["Ticker"] = display_df.apply(
                lambda r: f"{r['Ticker']} ({r['Universe']})", axis=1
            )
            display_df = display_df.drop(columns=["Universe"], errors="ignore")
        st.subheader(f"💹 Institutional Dip-Buy Matrix — {len(display_df)} Tickers")
        st.dataframe(
            display_df.style.apply(color_execution_column, axis=None),
            hide_index=True,
            use_container_width=True
        )

# ---------------------------------------------------------
# RUN BUTTON
# ---------------------------------------------------------
run_model = st.button("Run Dip-Buy Scan", key="page3_run_button")

if run_model:
    try:
        st.cache_data.clear()
        start_time = time.time()
        eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now().astimezone(eastern)

        st.markdown(f"⏱️ Scan Execution Time Stamp: **{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**")
        progress_bar = st.progress(0, text="Synchronizing institutional backbone batch...")

        universe_list = load_universe()
        if not universe_list:
            st.error("The stock universe source file returned completely empty.")
            st.stop()

        raw_daily, raw_intra = fetch_clean_market_batch(tuple(universe_list))

        if raw_daily is None or raw_intra is None:
            st.error("Global exchange batch synchronization returned a NoneType connection error. Re-trigger the scan.")
            progress_bar.empty()
            st.stop()

        if hasattr(raw_daily, "empty") and raw_daily.empty or hasattr(raw_intra, "empty") and raw_intra.empty:
            st.warning("Data matrices downloaded successfully but returned no active price data.")
            progress_bar.empty()
            st.stop()

        progress_bar.progress(0.4, text="Running institutional dip-buy scoring engine...")
        ranking = local_rank_dip_buy_universe(universe_list, raw_daily, raw_intra, min_price, max_price)

        if ranking is not None and not ranking.empty:
            st.session_state["page3_ranking"] = ranking
        else:
            st.warning("No institutional dip-buy candidates on this cycle.")
            if "page3_ranking" in st.session_state:
                del st.session_state["page3_ranking"]

        progress_bar.empty()
        st.write(f"⏱ ... Dip-Buy Scan Complete.")
        st.write(f"⚡ Total Model Runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        try:
            progress_bar.empty()
        except NameError:
            pass
        st.error(f"Dip-buy model execution failed: {str(e)}")
        st.exception(e)

# ---------------------------------------------------------
# RENDER PANEL
# ---------------------------------------------------------
if "page3_ranking" in st.session_state:
    ranking = st.session_state["page3_ranking"]
    if ranking is not None and not ranking.empty:
        filtered = ranking.copy()
        filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]
        if execution_filter and "Execution" in filtered.columns:
            filtered = filtered[filtered["Execution"].isin(execution_filter)]
        render_results(filtered)
