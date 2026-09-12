# ==============================================================================
# 🚀 UNIVERSAL MOMENTUM SCANNER — IMPROVED
# Version: 2026-09-12
#
# DESIGN PRINCIPLE:
#   The model identifies and ranks opportunities.
#   Execution validates the opportunity and manages the trade.
#
# IMPORTANT:
#   This version intentionally removes the old Profit Target / Price Position /
#   Exit Signal framework. The scanner does not assume an entry price and does
#   not generate an automatic profit target or exit decision.
#
# CORE SIGNALS RETAINED:
#   - EMA9 velocity
#   - Time-of-day adjusted RVOL
#   - Momentum Score (0-8)
#   - Continuation Score (0-16)
#   - Intraday range expansion
#   - Float / Market Cap
#   - VWAP
#   - Current intraday price
#   - Current intraday data timestamp
#
# IMPROVEMENTS:
#   1. Removed Profit Target / Exit / assumed-entry tracking.
#   2. Standardized EMA9 to adjust=False.
#   3. Added 5-bar actual price movement as an informational signal.
#      It does NOT change the Momentum Score.
#   4. Uses America/New_York timezone.
#   5. Timeline records scanner observations only; it does not pretend that
#      the scanner price was the trader's actual entry price.
# ==============================================================================

import streamlit as st
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from zoneinfo import ZoneInfo

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Momentum Model"
)

EST = ZoneInfo("America/New_York")

st.caption(
    "Version: 2026-09-12 — Momentum + Continuation + Actual Price Movement"
)

st.title("🚀 Universal Momentum Scanner — With Continuation Probability")

st.info(
    "Model role: identify and rank opportunities. "
    "Execution role: validate price action, volume, spread, RSI, VWAP, "
    "resistance, market conditions, entry, stop and exit."
)

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "momentum_history" not in st.session_state:
    st.session_state["momentum_history"] = []

if "momentum_history_date" not in st.session_state:
    st.session_state["momentum_history_date"] = date.today()

if "momentum_raw_ranking" not in st.session_state:
    st.session_state["momentum_raw_ranking"] = pd.DataFrame()

# ==============================================================================
# CONTINUATION SCORE
#
# A = Float
# B = Market Cap
# C = RVOL / Float
# D = Intraday Range Expansion
# Maximum = 16
# ==============================================================================
def continuation_score(
    float_val,
    market_cap,
    rvol,
    range_pct
):

    try:
        float_val = float(float_val)
        market_cap = float(market_cap)
        rvol = float(rvol)
        range_pct = float(range_pct)
    except (TypeError, ValueError):
        return 0

    # --------------------------------------------------------------------------
    # A — Public Float Score
    # --------------------------------------------------------------------------
    if not np.isfinite(float_val) or float_val <= 0:
        A = 0
        C = 0
    else:
        if float_val < 50_000_000:
            A = 4
        elif float_val < 150_000_000:
            A = 3
        elif float_val < 300_000_000:
            A = 2
        else:
            A = 1

        # ----------------------------------------------------------------------
        # C — RVOL / Float Score
        # ----------------------------------------------------------------------
        float_millions = float_val / 1_000_000

        if float_millions > 0:
            ratio = rvol / float_millions

            if ratio > 0.20:
                C = 4
            elif ratio > 0.10:
                C = 3
            elif ratio > 0.05:
                C = 2
            else:
                C = 1
        else:
            C = 0

    # --------------------------------------------------------------------------
    # B — Market Cap Score
    # --------------------------------------------------------------------------
    if not np.isfinite(market_cap) or market_cap <= 0:
        B = 0
    elif market_cap < 5_000_000_000:
        B = 4
    elif market_cap < 20_000_000_000:
        B = 3
    elif market_cap < 50_000_000_000:
        B = 2
    else:
        B = 1

    # --------------------------------------------------------------------------
    # D — Intraday Range Expansion
    # --------------------------------------------------------------------------
    if not np.isfinite(range_pct) or range_pct < 0:
        D = 0
    elif range_pct > 2.0:
        D = 4
    elif range_pct > 1.2:
        D = 3
    elif range_pct > 0.8:
        D = 2
    else:
        D = 1

    return A + B + C + D


# ==============================================================================
# ACTUAL SHORT-TERM PRICE MOVEMENT
#
# This is deliberately informational/ranking-only.
# It does NOT alter Momentum Score.
# ==============================================================================
def price_movement_score(price_change_pct):

    try:
        pct = float(price_change_pct)
    except (TypeError, ValueError):
        return 0

    if not np.isfinite(pct):
        return 0

    if pct >= 1.00:
        return 3
    elif pct >= 0.50:
        return 2
    elif pct >= 0.25:
        return 1
    else:
        return 0


# ==============================================================================
# COLOR CODING — CONTINUATION SCORE
# ==============================================================================
def color_continuation(df):

    style_df = pd.DataFrame(
        "",
        index=df.index,
        columns=df.columns
    )

    if "Continuation_Score" not in df.columns:
        return style_df

    for i in range(len(df)):
        score = df.iloc[i]["Continuation_Score"]

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        if score >= 14:
            style_df.loc[
                df.index[i],
                "Continuation_Score"
            ] = (
                "background-color:#006400;"
                "color:white;"
                "font-weight:bold;"
            )
        elif score >= 10:
            style_df.loc[
                df.index[i],
                "Continuation_Score"
            ] = (
                "background-color:#32CD32;"
                "color:black;"
                "font-weight:bold;"
            )
        elif score >= 7:
            style_df.loc[
                df.index[i],
                "Continuation_Score"
            ] = (
                "background-color:#FFD700;"
                "color:black;"
                "font-weight:bold;"
            )
        else:
            style_df.loc[
                df.index[i],
                "Continuation_Score"
            ] = (
                "background-color:#FF4500;"
                "color:white;"
                "font-weight:bold;"
            )

    return style_df


# ==============================================================================
# COLOR CODING — ACTUAL PRICE MOVEMENT SCORE
# ==============================================================================
def color_price_movement(df):

    style_df = pd.DataFrame(
        "",
        index=df.index,
        columns=df.columns
    )

    if "Price_Movement_Score" not in df.columns:
        return style_df

    for i in range(len(df)):
        score = df.iloc[i]["Price_Movement_Score"]

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        if score >= 3:
            style_df.loc[
                df.index[i],
                "Price_Movement_Score"
            ] = (
                "background-color:#006400;"
                "color:white;"
                "font-weight:bold;"
            )
        elif score >= 2:
            style_df.loc[
                df.index[i],
                "Price_Movement_Score"
            ] = (
                "background-color:#32CD32;"
                "color:black;"
                "font-weight:bold;"
            )
        elif score >= 1:
            style_df.loc[
                df.index[i],
                "Price_Movement_Score"
            ] = (
                "background-color:#FFD700;"
                "color:black;"
                "font-weight:bold;"
            )

    return style_df


# ==============================================================================
# DATA FETCH
# ==============================================================================
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
            threads=True,
            auto_adjust=False
        )

        raw_intra = yf.download(
            ticker_list,
            period="1d",
            interval="1m",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=False
        )

        return raw_daily, raw_intra

    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ==============================================================================
# FETCH FLOAT + MARKET CAP
# ==============================================================================
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_float_marketcap(ticker):

    try:
        ticker_obj = yf.Ticker(ticker)

        float_val = 0.0
        market_cap = 0.0
        shares_outstanding = 0.0
        float_source = "Unavailable"

        try:
            fast = ticker_obj.fast_info

            try:
                market_cap = float(
                    fast.get("market_cap", 0) or 0
                )
            except Exception:
                market_cap = 0.0

            try:
                shares_outstanding = float(
                    fast.get("shares_outstanding", 0) or 0
                )
            except Exception:
                shares_outstanding = 0.0

        except Exception:
            fast = None

        try:
            info = ticker_obj.info

            yahoo_float = info.get(
                "floatShares",
                0
            )

            if yahoo_float:
                try:
                    float_val = float(yahoo_float)
                    if float_val > 0:
                        float_source = "floatShares"
                except Exception:
                    float_val = 0.0

            if shares_outstanding <= 0:
                yahoo_shares = info.get(
                    "sharesOutstanding",
                    0
                )

                if yahoo_shares:
                    try:
                        shares_outstanding = float(
                            yahoo_shares
                        )
                    except Exception:
                        shares_outstanding = 0.0

            if market_cap <= 0:
                yahoo_market_cap = info.get(
                    "marketCap",
                    0
                )

                if yahoo_market_cap:
                    try:
                        market_cap = float(
                            yahoo_market_cap
                        )
                    except Exception:
                        market_cap = 0.0

        except Exception:
            pass

        if float_val <= 0 and shares_outstanding > 0:
            float_val = shares_outstanding
            float_source = "sharesOutstanding_fallback"

        if not np.isfinite(float_val) or float_val < 0:
            float_val = 0.0

        if not np.isfinite(market_cap) or market_cap < 0:
            market_cap = 0.0

        if not np.isfinite(shares_outstanding) or shares_outstanding < 0:
            shares_outstanding = 0.0

        return (
            float_val,
            market_cap,
            shares_outstanding,
            float_source
        )

    except Exception:
        return (
            0.0,
            0.0,
            0.0,
            "Unavailable"
        )


# ==============================================================================
# MOMENTUM ENGINE
# ==============================================================================
def momentum_rank_universe_batch(
    tickers,
    batch_daily,
    batch_intra,
    min_price,
    max_price
):

    rows = []

    if (
        batch_daily is None
        or batch_daily.empty
        or batch_intra is None
        or batch_intra.empty
    ):
        return pd.DataFrame()

    now_est = datetime.now(EST)
    current_date_est = now_est.date()

    # --------------------------------------------------------------------------
    # Determine available tickers safely for both normal MultiIndex layouts.
    # --------------------------------------------------------------------------
    try:
        if isinstance(batch_daily.columns, pd.MultiIndex):
            available_daily = set(
                batch_daily.columns.get_level_values(0)
            )
        else:
            available_daily = set(tickers)
    except Exception:
        available_daily = set(tickers)

    try:
        if isinstance(batch_intra.columns, pd.MultiIndex):
            available_intra = set(
                batch_intra.columns.get_level_values(0)
            )
        else:
            available_intra = set(tickers)
    except Exception:
        available_intra = set(tickers)

    active_pool = sorted(
        set(tickers)
        .intersection(available_daily)
        .intersection(available_intra)
    )

    for ticker in active_pool:

        try:
            # ------------------------------------------------------------------
            # Extract ticker slices.
            # ------------------------------------------------------------------
            daily_df = (
                batch_daily[ticker]
                .copy()
                .dropna(subset=["Close"])
            )

            intraday_df = (
                batch_intra[ticker]
                .copy()
                .dropna(subset=["Close"])
            )

            if (
                daily_df.empty
                or intraday_df.empty
                or len(daily_df) < 40
            ):
                continue

            # ------------------------------------------------------------------
            # Normalize intraday timezone.
            # ------------------------------------------------------------------
            try:
                intra_index = pd.DatetimeIndex(
                    intraday_df.index
                )

                if intra_index.tz is not None:
                    intra_index = intra_index.tz_convert(
                        "America/New_York"
                    )
                else:
                    intra_index = intra_index.tz_localize(
                        "America/New_York"
                    )

                intraday_df.index = intra_index

            except Exception:
                continue

            if len(intraday_df.index) == 0:
                continue

            latest_intraday_timestamp = intraday_df.index[-1]
            latest_intraday_date = latest_intraday_timestamp.date()

            # ------------------------------------------------------------------
            # CRITICAL CURRENT-DAY PROTECTION
            # ------------------------------------------------------------------
            # Never allow yesterday's intraday data to appear as today's signal.
            # ------------------------------------------------------------------
            if latest_intraday_date != current_date_est:
                continue

            data_as_of = latest_intraday_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # ------------------------------------------------------------------
            # 20-day average volume
            # ------------------------------------------------------------------
            vol_d = pd.to_numeric(
                daily_df["Volume"],
                errors="coerce"
            ).fillna(0).values

            if len(vol_d) < 20:
                continue

            avg_volume_20d = float(
                np.mean(vol_d[-20:])
            )

            if avg_volume_20d < 250000:
                continue

            # ------------------------------------------------------------------
            # Current intraday price is the scanner's current price.
            # ------------------------------------------------------------------
            current_price = float(
                intraday_df["Close"].iloc[-1]
            )

            if (
                current_price < min_price
                or current_price > max_price
            ):
                continue

            close_i = pd.to_numeric(
                intraday_df["Close"],
                errors="coerce"
            ).dropna().values

            vol_i = pd.to_numeric(
                intraday_df["Volume"],
                errors="coerce"
            ).fillna(0).values

            if len(close_i) < 5:
                continue

            # ------------------------------------------------------------------
            # EMA9 — standardized to adjust=False.
            # ------------------------------------------------------------------
            ema9_i_series = (
                pd.Series(close_i)
                .ewm(
                    span=9,
                    adjust=False
                )
                .mean()
                .values
            )

            # ------------------------------------------------------------------
            # EMA9 velocity: 5-bar relative change.
            # ------------------------------------------------------------------
            if len(ema9_i_series) >= 5:
                previous_ema9 = float(
                    ema9_i_series[-5]
                )

                if previous_ema9 != 0:
                    ema9_slope_10 = (
                        (
                            ema9_i_series[-1]
                            - previous_ema9
                        )
                        / previous_ema9
                    ) * 100
                else:
                    ema9_slope_10 = 0.0
            else:
                ema9_slope_10 = 0.0

            # ------------------------------------------------------------------
            # Actual short-term price movement.
            #
            # This is deliberately separate from Momentum Score so Monday's
            # test can show whether it adds useful information without changing
            # the existing 0-8 momentum calculation.
            # ------------------------------------------------------------------
            last5 = close_i[-5:]

            if last5[0] > 0:
                price_change_5bar_pct = (
                    (last5[-1] - last5[0])
                    / last5[0]
                ) * 100
            else:
                price_change_5bar_pct = 0.0

            movement_score = price_movement_score(
                price_change_5bar_pct
            )

            # ------------------------------------------------------------------
            # TIME-OF-DAY ADJUSTED RVOL
            # Regular session = 390 minutes.
            # ------------------------------------------------------------------
            intraday_total_volume = float(
                np.sum(vol_i)
            )

            latest_bar_time = intraday_df.index[-1]

            if latest_bar_time.tzinfo is None:
                latest_bar_time = latest_bar_time.replace(
                    tzinfo=EST
                )
            else:
                latest_bar_time = latest_bar_time.astimezone(
                    EST
                )

            market_open = latest_bar_time.replace(
                hour=9,
                minute=30,
                second=0,
                microsecond=0
            )

            elapsed_minutes = (
                latest_bar_time - market_open
            ).total_seconds() / 60.0

            elapsed_minutes = max(
                1.0,
                min(
                    elapsed_minutes,
                    390.0
                )
            )

            session_fraction = (
                elapsed_minutes / 390.0
            )

            expected_volume_by_now = (
                avg_volume_20d
                * session_fraction
            )

            if expected_volume_by_now > 0:
                rvol = (
                    intraday_total_volume
                    / expected_volume_by_now
                )
            else:
                rvol = 1.0

            # ------------------------------------------------------------------
            # VWAP proxy from accumulated intraday close * volume.
            # ------------------------------------------------------------------
            cv_slice = vol_i * close_i

            vwap_spot = (
                cv_slice.sum() / vol_i.sum()
                if vol_i.sum() > 0
                else current_price
            )

            # ------------------------------------------------------------------
            # Current intraday bar range.
            # ------------------------------------------------------------------
            high_i = float(
                intraday_df["High"].iloc[-1]
            )
            low_i = float(
                intraday_df["Low"].iloc[-1]
            )

            range_pct = (
                ((high_i - low_i) / low_i) * 100
                if low_i > 0
                else 0.0
            )

            # ------------------------------------------------------------------
            # Float / market cap.
            # ------------------------------------------------------------------
            (
                float_val,
                market_cap,
                shares_outstanding,
                float_source
            ) = fetch_float_marketcap(ticker)

            # ------------------------------------------------------------------
            # Existing Momentum Score — intentionally unchanged.
            # Maximum = 8.
            # ------------------------------------------------------------------
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

            momentum_score = (
                velocity_score
                + rvol_score
            )

            # ------------------------------------------------------------------
            # Continuation Score — unchanged.
            # ------------------------------------------------------------------
            cont_score = continuation_score(
                float_val,
                market_cap,
                rvol,
                range_pct
            )

            rows.append({
                "Ticker": ticker,
                "Close": round(current_price, 2),
                "Momentum_Score": round(
                    momentum_score,
                    2
                ),
                "Continuation_Score": cont_score,
                "Price_Change_5B_Pct": round(
                    price_change_5bar_pct,
                    2
                ),
                "Price_Movement_Score": movement_score,
                "RVOL": round(rvol, 2),
                "Range_Pct": round(range_pct, 2),
                "Float": float_val,
                "Market_Cap": market_cap,
                "VWAP": round(vwap_spot, 2),
                "EMA9_Slope_10": round(
                    ema9_slope_10,
                    3
                ),
                "Data_As_Of": data_as_of
            })

        except Exception as e:
            print(
                f"ERROR processing {ticker}: "
                f"{type(e).__name__}: {e}"
            )
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for col in [
        "Momentum_Score",
        "Continuation_Score",
        "Price_Change_5B_Pct",
        "Price_Movement_Score"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0.0)

    return df


# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
st.markdown("### 🔍 Price Boundaries Filter")

min_price = st.number_input(
    "Minimum Price ($)",
    value=40.0,
    min_value=40.0,
    max_value=120.0,
    key="momentum_min_price"
)

max_price = st.number_input(
    "Maximum Price ($)",
    value=120.0,
    min_value=40.0,
    max_value=120.0,
    key="momentum_max_price"
)

st.markdown("### 🎛️ Momentum Score Filter")

min_momentum_score = st.number_input(
    "Minimum Momentum Score",
    value=4.0,
    min_value=0.0,
    max_value=8.0,
    step=0.5,
    key="momentum_min_score"
)

st.caption(
    "Momentum Score remains 0–8. "
    "5-bar actual price movement is displayed separately and does not change it."
)

# ==============================================================================
# RUN MOMENTUM ENGINE
# ==============================================================================
run_momentum = st.button(
    "Run Momentum Model Scan",
    key="run_momentum_model"
)

if run_momentum:

    progress_bar = None

    try:
        # Clear cached market data so a manual Run requests a fresh snapshot.
        st.cache_data.clear()

        start_time = time.time()
        now_est = datetime.now(EST)

        # ----------------------------------------------------------------------
        # WEEKEND CHECK
        # ----------------------------------------------------------------------
        if now_est.weekday() >= 5:
            st.warning(
                "⚠️ U.S. stock market is closed today."
            )

            st.info(
                "The Momentum Model requires current-day 1-minute intraday data. "
                "No scan was performed, so Friday's data cannot appear as "
                "weekend momentum."
            )

            st.stop()

        st.markdown(
            "⏱️ Scan Time: "
            f"**{now_est.strftime('%Y-%m-%d %H:%M:%S')} EST**"
        )

        progress_bar = st.progress(
            0,
            text="Loading universe..."
        )

        from utils.data_fetch import load_universe

        universe_list = load_universe()

        progress_bar.progress(
            40,
            text="Loading market data..."
        )

        raw_daily, raw_intra = fetch_clean_market_batch(
            tuple(universe_list)
        )

        progress_bar.progress(
            70,
            text="Running momentum + continuation engine..."
        )

        ranking = momentum_rank_universe_batch(
            universe_list,
            raw_daily,
            raw_intra,
            min_price,
            max_price
        )

        if ranking is not None and not ranking.empty:
            st.session_state[
                "momentum_raw_ranking"
            ] = ranking
        else:
            st.session_state[
                "momentum_raw_ranking"
            ] = pd.DataFrame()

        progress_bar.progress(
            100,
            text="Scan complete"
        )

        progress_bar.empty()

        st.write(
            "⚡ Total Runtime: "
            f"{time.time() - start_time:.2f} seconds"
        )

    except Exception as e:

        if progress_bar is not None:
            try:
                progress_bar.empty()
            except Exception:
                pass

        st.error(
            "Momentum model execution failed: "
            f"{str(e)}"
        )

        st.exception(e)


# ==============================================================================
# RENDER RESULTS PANEL
# ==============================================================================
if "momentum_raw_ranking" in st.session_state:

    ranking = st.session_state[
        "momentum_raw_ranking"
    ]

    if ranking is not None and not ranking.empty:

        filtered = ranking.copy()

        filtered = filtered[
            (filtered["Close"] >= min_price)
            &
            (filtered["Close"] <= max_price)
        ]

        filtered = filtered[
            filtered["Momentum_Score"] >= min_momentum_score
        ]

        if filtered.empty:
            st.info(
                "No tickers matched your filters."
            )

        else:
            # ------------------------------------------------------------------
            # Primary ranking remains Momentum Score, then Continuation Score.
            # Actual price movement is a secondary ranking observation.
            # ------------------------------------------------------------------
            display_df = filtered.copy()

            display_df = display_df.sort_values(
                by=[
                    "Momentum_Score",
                    "Continuation_Score",
                    "Price_Movement_Score",
                    "Price_Change_5B_Pct"
                ],
                ascending=False
            )

            if "Data_As_Of" in display_df.columns:
                data_as_of_values = (
                    display_df["Data_As_Of"]
                    .dropna()
                    .unique()
                )

                if len(data_as_of_values) > 0:
                    st.caption(
                        "📡 Intraday Data As Of: "
                        f"**{data_as_of_values[0]} EST**"
                    )

            st.subheader(
                f"🔥 Momentum Matrix — {len(display_df)} Tickers"
            )

            st.caption(
                "Momentum Score = core momentum signal. "
                "Continuation Score = structural continuation context. "
                "Price Change 5B = actual recent price movement for validation/ranking."
            )

            st.dataframe(
                display_df.style
                .apply(
                    color_continuation,
                    axis=None
                )
                .apply(
                    color_price_movement,
                    axis=None
                ),
                hide_index=True,
                use_container_width=True
            )

            # ==================================================================
            # RESET OBSERVATION TIMELINE
            # ==================================================================
            if st.button(
                "Reset Momentum Timeline",
                key="reset_momentum_tracker"
            ):
                st.session_state[
                    "momentum_history"
                ] = []

                st.session_state[
                    "momentum_history_date"
                ] = date.today()

                st.success(
                    "Momentum observation timeline reset."
                )

            # ------------------------------------------------------------------
            # New day = new observation timeline.
            # ------------------------------------------------------------------
            if (
                st.session_state["momentum_history_date"]
                != date.today()
            ):
                st.session_state[
                    "momentum_history"
                ] = []

                st.session_state[
                    "momentum_history_date"
                ] = date.today()

            # ==================================================================
            # MOMENTUM TIMELINE — TOP 5 OBSERVATIONS
            #
            # This is deliberately NOT an entry/exit tracker.
            # Each row is simply what the scanner observed at that run.
            # ==================================================================
            top5 = display_df.head(5).copy()

            top5["Scan_Timestamp"] = datetime.now(EST).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            st.session_state[
                "momentum_history"
            ].append(top5)

            # Keep a practical in-session history rather than allowing an
            # unlimited dataframe to grow during repeated two-minute scans.
            if len(st.session_state["momentum_history"]) > 60:
                st.session_state[
                    "momentum_history"
                ] = st.session_state[
                    "momentum_history"
                ][-60:]

            history_df = pd.concat(
                st.session_state[
                    "momentum_history"
                ],
                ignore_index=True
            )

            st.subheader(
                "📊 Momentum Observation Timeline — Top 5"
            )

            st.caption(
                "Observation history only. No assumed entry price, profit target, "
                "stop calculation, or automatic exit signal is generated."
            )

            st.dataframe(
                history_df.style
                .apply(
                    color_continuation,
                    axis=None
                )
                .apply(
                    color_price_movement,
                    axis=None
                ),
                hide_index=True,
                use_container_width=True
            )
