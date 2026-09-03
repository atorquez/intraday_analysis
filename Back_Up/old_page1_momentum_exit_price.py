import streamlit as st
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import pytz

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="Momentum Model")
st.caption("Version: 2026-08-28 — Momentum + Continuation + Momentum Exit")
st.title("🚀 Universal Momentum Scanner — With Momentum Exit Signal")

# =========================================================
# SESSION STATE
# =========================================================
for key in ["entry_prices", "stop_limits", "safe_thresholds",
            "profit_targets", "momentum_history",
            "momentum_raw_ranking"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "prices" in key else []

if "momentum_history_date" not in st.session_state:
    st.session_state["momentum_history_date"] = date.today()

# =========================================================
# PRICE POSITION LOGIC
# =========================================================
def compute_price_position(entry_price, current_price):
    if entry_price is None or current_price is None:
        return "UNKNOWN"
    if not np.isfinite(entry_price) or not np.isfinite(current_price):
        return "UNKNOWN"

    safe_threshold = entry_price * 1.0005
    stop_limit = entry_price * 0.9995

    if current_price <= stop_limit:
        return "STOP"
    elif current_price < entry_price:
        return "DANGER"
    elif current_price <= safe_threshold:
        return "CAUTION"
    else:
        return "SAFE"

# =========================================================
# PROFIT TARGET
# =========================================================
def compute_profit_target(entry_price, pct=0.015):
    if entry_price is None or not np.isfinite(entry_price):
        return None
    return entry_price * (1.0 + pct)

# =========================================================
# PRICE-BASED EXIT SIGNAL
# =========================================================
def compute_exit_signal(entry_price, current_price):
    if entry_price is None or current_price is None:
        return "HOLD"
    if not np.isfinite(entry_price) or not np.isfinite(current_price):
        return "HOLD"

    stop_limit = entry_price * 0.9995
    profit_target = compute_profit_target(entry_price, pct=0.015)

    if profit_target is None:
        return "HOLD"

    if current_price >= profit_target:
        return "EXIT_PROFIT"
    elif current_price <= stop_limit:
        return "EXIT_STOP"
    else:
        return "HOLD"

# =========================================================
# MOMENTUM EXIT SIGNAL (EMA9 SLOPE)
# =========================================================
def compute_momentum_exit(ema9_slope_10):
    try:
        slope = float(ema9_slope_10)
    except Exception:
        return "HOLD"

    if slope < 0.00:
        return "EXIT_MOMENTUM"
    elif slope < 0.10:
        return "CAUTION"
    else:
        return "HOLD"

# =========================================================
# CONTINUATION SCORE
# =========================================================
def continuation_score(float_val, market_cap, rvol, range_pct):
    try:
        float_val = float(float_val)
        market_cap = float(market_cap)
        rvol = float(rvol)
        range_pct = float(range_pct)
    except:
        return 0

    # Float score
    if float_val <= 0:
        A = 0
        C = 0
    else:
        if float_val < 50_000_000: A = 4
        elif float_val < 150_000_000: A = 3
        elif float_val < 300_000_000: A = 2
        else: A = 1

        ratio = rvol / (float_val / 1_000_000)
        if ratio > 0.20: C = 4
        elif ratio > 0.10: C = 3
        elif ratio > 0.05: C = 2
        else: C = 1

    # Market cap score
    if market_cap <= 0: B = 0
    elif market_cap < 5e9: B = 4
    elif market_cap < 20e9: B = 3
    elif market_cap < 50e9: B = 2
    else: B = 1

    # Range expansion
    if range_pct < 0: D = 0
    elif range_pct > 2.0: D = 4
    elif range_pct > 1.2: D = 3
    elif range_pct > 0.8: D = 2
    else: D = 1

    return A + B + C + D

# =========================================================
# COLOR CODING
# =========================================================
def color_momentum_exit(df):
    style_df = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Momentum_Exit" not in df.columns:
        return style_df

    for i in range(len(df)):
        sig = df.iloc[i]["Momentum_Exit"]
        if sig == "EXIT_MOMENTUM":
            style_df.loc[df.index[i], "Momentum_Exit"] = (
                "background-color:#F44336; color:white; font-weight:bold;"
            )
        elif sig == "CAUTION":
            style_df.loc[df.index[i], "Momentum_Exit"] = (
                "background-color:#FFC107; color:black; font-weight:bold;"
            )
        else:
            style_df.loc[df.index[i], "Momentum_Exit"] = (
                "background-color:#4CAF50; color:white; font-weight:bold;"
            )
    return style_df

def color_continuation(df):
    style_df = pd.DataFrame("", index=df.index, columns=df.columns)
    if "Continuation_Score" not in df.columns:
        return style_df
    for i in range(len(df)):
        score = float(df.iloc[i]["Continuation_Score"])
        if score >= 14:
            style_df.loc[df.index[i], "Continuation_Score"] = (
                "background-color:#006400; color:white; font-weight:bold;"
            )
        elif score >= 10:
            style_df.loc[df.index[i], "Continuation_Score"] = (
                "background-color:#32CD32; color:black; font-weight:bold;"
            )
        elif score >= 7:
            style_df.loc[df.index[i], "Continuation_Score"] = (
                "background-color:#FFD700; color:black; font-weight:bold;"
            )
        else:
            style_df.loc[df.index[i], "Continuation_Score"] = (
                "background-color:#FF4500; color:white; font-weight:bold;"
            )
    return style_df

# =========================================================
# DATA FETCH
# =========================================================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_clean_market_batch(tickers_tuple):
    try:
        raw_daily = yf.download(
            list(tickers_tuple),
            period="3mo",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=False
        )
        raw_intra = yf.download(
            list(tickers_tuple),
            period="1d",
            interval="1m",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=False
        )
        return raw_daily, raw_intra
    except:
        return pd.DataFrame(), pd.DataFrame()

# =========================================================
# FLOAT + MARKET CAP
# =========================================================
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_float_marketcap(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        float_val = float(info.get("floatShares", 0) or 0)
        market_cap = float(info.get("marketCap", 0) or 0)
        shares_outstanding = float(info.get("sharesOutstanding", 0) or 0)
        if float_val <= 0:
            float_val = shares_outstanding
        return float_val, market_cap, shares_outstanding, "Yahoo"
    except:
        return 0.0, 0.0, 0.0, "Unavailable"

# =========================================================
# MOMENTUM ENGINE
# =========================================================
def momentum_rank_universe_batch(tickers, batch_daily, batch_intra, min_price, max_price):
    rows = []
    if batch_daily.empty or batch_intra.empty:
        return pd.DataFrame()

    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now(eastern)
    today = now_est.date()

    available_daily = set(batch_daily.columns.get_level_values(0))
    available_intra = set(batch_intra.columns.get_level_values(0))
    active_pool = list(set(tickers).intersection(available_daily).intersection(available_intra))

    for ticker in active_pool:
        try:
            daily_df = batch_daily[ticker].dropna(subset=["Close"])
            intraday_df = batch_intra[ticker].dropna(subset=["Close"])
            if daily_df.empty or intraday_df.empty or len(daily_df) < 40:
                continue

            # Timestamp alignment
            idx = pd.DatetimeIndex(intraday_df.index)
            if idx.tz is None:
                idx = idx.tz_localize("US/Eastern")
            else:
                idx = idx.tz_convert("US/Eastern")
            intraday_df.index = idx

            if intraday_df.index[-1].date() != today:
                continue

            current_price = float(intraday_df["Close"].iloc[-1])
            if current_price < min_price or current_price > max_price:
                continue

            vol_d = daily_df["Volume"].values
            avg_volume_20d = float(np.mean(vol_d[-20:])) if len(vol_d) >= 20 else float(vol_d[-1])
            if avg_volume_20d < 250000:
                continue

            close_i = intraday_df["Close"].values
            vol_i = intraday_df["Volume"].values
            if len(close_i) < 5:
                continue

            # EMA9 slope
            ema9_i = intraday_df["Close"].ewm(span=9).mean().values
            ema9_slope_10 = ((ema9_i[-1] - ema9_i[-5]) / ema9_i[-5]) * 100 if ema9_i[-5] != 0 else 0.0

            # Time-of-day RVOL
            latest_bar = intraday_df.index[-1]
            market_open = latest_bar.replace(hour=9, minute=30, second=0)
            elapsed_minutes = max(1.0, min((latest_bar - market_open).total_seconds() / 60.0, 390.0))
            expected_volume = avg_volume_20d * (elapsed_minutes / 390.0)
            rvol = (vol_i.sum() / expected_volume) if expected_volume > 0 else 1.0

            # VWAP
            cv = vol_i * close_i
            vwap_spot = cv.sum() / vol_i.sum() if vol_i.sum() > 0 else current_price

            # Range expansion
            high_i = intraday_df["High"].iloc[-1]
            low_i = intraday_df["Low"].iloc[-1]
            range_pct = ((high_i - low_i) / low_i) * 100 if low_i > 0 else 0

            # Float + market cap
            float_val, market_cap, shares_outstanding, float_source = fetch_float_marketcap(ticker)

            # Momentum score
            velocity_score = 4 if ema9_slope_10 > 0.60 else \
                             3 if ema9_slope_10 > 0.30 else \
                             2 if ema9_slope_10 > 0.15 else \
                             1 if ema9_slope_10 > 0.00 else 0

            rvol_score = 4 if rvol > 5 else \
                         3 if rvol > 3 else \
                         2 if rvol > 2 else \
                         1 if rvol > 1.2 else 0

            momentum_score = velocity_score + rvol_score
            cont_score = continuation_score(float_val, market_cap, rvol, range_pct)

            rows.append({
                "Ticker": ticker,
                "Close": round(current_price, 2),
                "Momentum_Score": round(momentum_score, 2),
                "Continuation_Score": cont_score,
                "RVOL": round(rvol, 2),
                "Range_Pct": round(range_pct, 2),
                "Float": float_val,
                "Market_Cap": market_cap,
                "VWAP": round(vwap_spot, 2),
                "EMA9_Slope_10": round(ema9_slope_10, 3),
                "Momentum_Exit": compute_momentum_exit(ema9_slope_10),
                "Data_As_Of": intraday_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            })

        except Exception as e:
            print(f"ERROR processing {ticker}: {type(e).__name__}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Momentum_Score"] = pd.to_numeric(df["Momentum_Score"], errors="coerce").fillna(0.0)
    df["Continuation_Score"] = pd.to_numeric(df["Continuation_Score"], errors="coerce").fillna(0.0)
    return df

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.markdown("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Price ($)", value=40.0, min_value=40.0, max_value=110.0)
max_price = st.number_input("Maximum Price ($)", value=110.0, min_value=40.0, max_value=110.0)

st.markdown("### 🎛️ Momentum Score Filter")
min_momentum_score = st.number_input("Minimum Momentum Score", value=4.0, min_value=0.0, max_value=30.0, step=0.5)

# =========================================================
# RUN MOMENTUM ENGINE
# =========================================================
run_momentum = st.button("Run Momentum Model Scan")

if run_momentum:
    try:
        st.cache_data.clear()
        start_time = time.time()

        eastern = pytz.timezone("US/Eastern")
        now_est = datetime.now(eastern)

        if now_est.weekday() >= 5:
            st.warning("⚠️ U.S. stock market is closed today.")
            st.stop()

        st.markdown(f"⏱️ Scan Time: **{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**")

        progress_bar = st.progress(0, text="Loading universe...")
        from utils.data_fetch import load_universe
        universe_list = load_universe()

        progress_bar.progress(40, text="Loading market data...")
        raw_daily, raw_intra = fetch_clean_market_batch(tuple(universe_list))

        progress_bar.progress(70, text="Running momentum engine...")
        ranking = momentum_rank_universe_batch(universe_list, raw_daily, raw_intra, min_price, max_price)

        st.session_state["momentum_raw_ranking"] = ranking if not ranking.empty else pd.DataFrame()

        progress_bar.progress(100, text="Scan complete")
        progress_bar.empty()

        st.write(f"⚡ Total Runtime: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        try:
            progress_bar.empty()
        except:
            pass
        st.error(f"Momentum model execution failed: {str(e)}")
        st.exception(e)

# =========================================================
# RENDER RESULTS PANEL
# =========================================================
if "momentum_raw_ranking" in st.session_state:
    ranking = st.session_state["momentum_raw_ranking"] = pd.DataFrame()

    if ranking is not None and not ranking.empty:
        filtered = ranking.copy()
        filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]
        filtered = filtered[filtered["Momentum_Score"] >= min_momentum_score]

        if filtered.empty:
            st.info("No tickers matched your filters.")
        else:
            display_df = filtered.sort_values(
                by=["Momentum_Score", "Continuation_Score"],
                ascending=False
            )

            if "Data_As_Of" in display_df.columns:
                st.caption(f"📡 Intraday Data As Of: **{display_df['Data_As_Of'].iloc[0]} EST**")

            st.subheader(f"🔥 Momentum Matrix — {len(display_df)} Tickers")

            st.dataframe(
                display_df.style
                    .apply(color_continuation, axis=None)
                    .apply(color_momentum_exit, axis=None),
                hide_index=True,
                use_container_width=True
            )
