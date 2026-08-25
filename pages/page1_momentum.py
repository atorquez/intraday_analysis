#PART 1
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
st.set_page_config(
    layout="wide",
    page_title="Momentum Model"
)

st.caption(
    "Version: 2026-08-24 — Momentum + Price Position + "
    "Profit Target + Continuation Score"
)

st.title("🚀 Universal Momentum Scanner — With Continuation Probability")

# =========================================================
# SESSION STATE
# =========================================================
if "entry_prices" not in st.session_state:
    st.session_state["entry_prices"] = {}

if "stop_limits" not in st.session_state:
    st.session_state["stop_limits"] = {}

if "safe_thresholds" not in st.session_state:
    st.session_state["safe_thresholds"] = {}

if "profit_targets" not in st.session_state:
    st.session_state["profit_targets"] = {}

if "momentum_history" not in st.session_state:
    st.session_state["momentum_history"] = []

if "momentum_history_date" not in st.session_state:
    st.session_state["momentum_history_date"] = date.today()

if "momentum_raw_ranking" not in st.session_state:
    st.session_state["momentum_raw_ranking"] = pd.DataFrame()

# =========================================================
# PRICE POSITION + PROFIT TARGET LOGIC
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

    if entry_price is None:
        return None

    if not np.isfinite(entry_price):
        return None

    return entry_price * (1.0 + pct)

# =========================================================
# EXIT SIGNAL
# =========================================================
def compute_exit_signal(entry_price, current_price):

    if entry_price is None or current_price is None:
        return "HOLD"

    if not np.isfinite(entry_price) or not np.isfinite(current_price):
        return "HOLD"

    stop_limit = entry_price * 0.9995
    profit_target = compute_profit_target(
        entry_price,
        pct=0.015
    )

    if profit_target is None:
        return "HOLD"

    if current_price >= profit_target:
        return "EXIT_PROFIT"

    elif current_price <= stop_limit:
        return "EXIT_STOP"

    else:
        return "HOLD"

# =========================================================
# CONTINUATION SCORE
#
# A = Float
# B = Market Cap
# C = RVOL / Float
# D = Intraday Range Expansion
#
# Maximum = 16
# =========================================================
def continuation_score(
    float_val,
    market_cap,
    rvol,
    range_pct
):

    # -----------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------
    try:
        float_val = float(float_val)
        market_cap = float(market_cap)
        rvol = float(rvol)
        range_pct = float(range_pct)
    except (TypeError, ValueError):
        return 0

    # -----------------------------------------------------
    # Invalid float
    # -----------------------------------------------------
    if (
        not np.isfinite(float_val)
        or float_val <= 0
    ):
        A = 0
        C = 0
    else:

        # -------------------------------------------------
        # A — Public Float Score
        # -------------------------------------------------
        if float_val < 50_000_000:
            A = 4

        elif float_val < 150_000_000:
            A = 3

        elif float_val < 300_000_000:
            A = 2

        else:
            A = 1

        # -------------------------------------------------
        # C — RVOL / Float Score
        # -------------------------------------------------
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

    # -----------------------------------------------------
    # B — Market Cap Score
    # -----------------------------------------------------
    if (
        not np.isfinite(market_cap)
        or market_cap <= 0
    ):
        B = 0

    elif market_cap < 5_000_000_000:
        B = 4

    elif market_cap < 20_000_000_000:
        B = 3

    elif market_cap < 50_000_000_000:
        B = 2

    else:
        B = 1

    # -----------------------------------------------------
    # D — Intraday Range Expansion
    # -----------------------------------------------------
    if (
        not np.isfinite(range_pct)
        or range_pct < 0
    ):
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

# =========================================================
# COLOR CODING — CONTINUATION SCORE
# =========================================================
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

# =========================================================
# COLOR CODING — PRICE POSITION
# =========================================================
def color_price_position(df):

    style_df = pd.DataFrame(
        "",
        index=df.index,
        columns=df.columns
    )

    if "Price_Position" not in df.columns:
        return style_df

    for i in range(len(df)):

        pos = df.iloc[i]["Price_Position"]

        if pos == "SAFE":

            style_df.loc[
                df.index[i],
                "Price_Position"
            ] = (
                "background-color:#4CAF50;"
                "color:white;"
                "font-weight:bold;"
            )

        elif pos == "CAUTION":

            style_df.loc[
                df.index[i],
                "Price_Position"
            ] = (
                "background-color:#FFC107;"
                "color:black;"
                "font-weight:bold;"
            )

        elif pos == "DANGER":

            style_df.loc[
                df.index[i],
                "Price_Position"
            ] = (
                "background-color:#FF5722;"
                "color:white;"
                "font-weight:bold;"
            )

        elif pos == "STOP":

            style_df.loc[
                df.index[i],
                "Price_Position"
            ] = (
                "background-color:#F44336;"
                "color:white;"
                "font-weight:bold;"
            )

    return style_df

# =========================================================
# COLOR CODING — EXIT SIGNAL
# =========================================================
def color_exit_signal(df):

    style_df = pd.DataFrame(
        "",
        index=df.index,
        columns=df.columns
    )

    if "Exit_Signal" not in df.columns:
        return style_df

    for i in range(len(df)):

        sig = df.iloc[i]["Exit_Signal"]

        if sig == "EXIT_PROFIT":

            style_df.loc[
                df.index[i],
                "Exit_Signal"
            ] = (
                "background-color:#4CAF50;"
                "color:white;"
                "font-weight:bold;"
            )

        elif sig == "EXIT_STOP":

            style_df.loc[
                df.index[i],
                "Exit_Signal"
            ] = (
                "background-color:#F44336;"
                "color:white;"
                "font-weight:bold;"
            )

        else:

            style_df.loc[
                df.index[i],
                "Exit_Signal"
            ] = (
                "background-color:#FFC107;"
                "color:black;"
                "font-weight:bold;"
            )

    return style_df

# =========================================================
# DATA FETCH
# =========================================================
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

        # IMPORTANT:
        # Diagnostics DO NOT belong here because `ticker` does not exist.
        # Diagnostics must be placed inside the per‑ticker loop in Page1.

        return raw_daily, raw_intra

    except Exception:

        return pd.DataFrame(), pd.DataFrame()

# =========================================================
# FETCH FLOAT + MARKET CAP
# =========================================================
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

        if float_val <= 0:

            if shares_outstanding > 0:

                float_val = shares_outstanding
                float_source = "sharesOutstanding_fallback"

        if (
            not np.isfinite(float_val)
            or float_val < 0
        ):
            float_val = 0.0

        if (
            not np.isfinite(market_cap)
            or market_cap < 0
        ):
            market_cap = 0.0

        if (
            not np.isfinite(shares_outstanding)
            or shares_outstanding < 0
        ):
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
#PART 2
# =========================================================
# MOMENTUM ENGINE (with continuation score)
# =========================================================
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

    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now(eastern)
    current_date_est = now_est.date()

    available_daily = set(
        batch_daily.columns.get_level_values(0)
    )

    available_intra = set(
        batch_intra.columns.get_level_values(0)
    )

    active_pool = list(
        set(tickers)
        .intersection(available_daily)
        .intersection(available_intra)
    )

    for ticker in active_pool:

        try:

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

            try:

                intra_index = pd.DatetimeIndex(
                    intraday_df.index
                )

                if intra_index.tz is not None:

                    intra_index = intra_index.tz_convert(
                        "US/Eastern"
                    )

                else:

                    intra_index = intra_index.tz_localize(
                        "US/Eastern"
                    )

                intraday_df.index = intra_index

            except Exception:

                continue

            if len(intraday_df.index) == 0:
                continue

            latest_intraday_timestamp = (
                intraday_df.index[-1]
            )

            latest_intraday_date = (
                latest_intraday_timestamp.date()
            )

            if latest_intraday_date != current_date_est:
                continue

            data_as_of = (
                latest_intraday_timestamp
                .strftime("%Y-%m-%d %H:%M:%S")
                )   
           
            vol_d = daily_df["Volume"].values

            avg_volume_20d = (
                float(np.mean(vol_d[-20:]))
                if len(vol_d) >= 20
                else float(vol_d[-1])
            )

            if avg_volume_20d < 250000:
                continue

            current_price = float(
                intraday_df["Close"].iloc[-1]
            )

            if (
                current_price < min_price
                or current_price > max_price
            ):
                continue

            close_i = intraday_df["Close"].values
            vol_i = intraday_df["Volume"].values

            if len(close_i) < 5:
                continue

            ema9_i_series = (
                intraday_df["Close"]
                .ewm(span=9)
                .mean()
                .values
            )

            if len(ema9_i_series) >= 5:

                previous_ema9 = (
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

            intraday_total_volume = float(
                vol_i.sum()
            )
            #OLD RVOL ==========================================
            #rvol = (
            #                float(
            #                    intraday_total_volume
            #                    / avg_volume_20d
            #                )
            #                if avg_volume_20d > 0
            #                else 1.0
            #            )

            #NEW RVOL CALCULATION
            # =================================================
            # TIME-OF-DAY ADJUSTED RVOL
            #
            # We compare today's accumulated volume against
            # the amount of volume normally expected to have
            # traded by this time of day.
            #
            # Regular US session:
            # 09:30 - 16:00 = 390 minutes
            # =================================================

            intraday_total_volume = float(
                vol_i.sum()
            )

            latest_bar_time = intraday_df.index[-1]

            # Convert to Eastern if necessary
            if latest_bar_time.tzinfo is None:
                latest_bar_time = eastern.localize(
                    latest_bar_time
                )
            else:
                latest_bar_time = latest_bar_time.astimezone(
                    eastern
                )

            market_open = latest_bar_time.replace(
                        hour=9,
                        minute=30,
                        second=0,
                        microsecond=0
            )

            market_close = latest_bar_time.replace(
                hour=16,
                minute=0,
                second=0,
                microsecond=0
            )

            elapsed_minutes = (
                        latest_bar_time - market_open
            ).total_seconds() / 60.0

            #Keep elapsed time inside the regular session
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

            # Expected volume assuming the average daily volume
            # is distributed across the regular session.
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

            cv_slice = vol_i * close_i

            vwap_spot = (
                        cv_slice.sum() / vol_i.sum()
                        if vol_i.sum() > 0
                        else current_price
            )

            high_i = intraday_df["High"].iloc[-1]
            low_i = intraday_df["Low"].iloc[-1]

            range_pct = (
            ((high_i - low_i) / low_i) * 100
            if low_i > 0
            else 0
            )

            float_val, market_cap, shares_outstanding, float_source = fetch_float_marketcap(ticker)

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

            cont_score = continuation_score(
                float_val,
                market_cap,
                rvol,
                range_pct
            )

            rows.append({

                "Ticker": ticker,

                "Close": round(
                    current_price,
                    2
                ),

                "Momentum_Score": round(
                    momentum_score,
                    2
                ),

                "Continuation_Score": cont_score,

                "RVOL": round(
                    rvol,
                    2
                ),

                "Range_Pct": round(
                    range_pct,
                    2
                ),

                "Float": float_val,

                "Market_Cap": market_cap,

                "VWAP": round(
                    vwap_spot,
                    2
                ),

                "EMA9_Slope_10": round(
                    ema9_slope_10,
                    3
                ),

                "Data_As_Of": data_as_of
            })

        except Exception:

            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Momentum_Score"] = pd.to_numeric(
        df["Momentum_Score"],
        errors="coerce"
    ).fillna(0.0)

    df["Continuation_Score"] = pd.to_numeric(
        df["Continuation_Score"],
        errors="coerce"
    ).fillna(0.0)

    return df

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.markdown("### 🔍 Price Boundaries Filter")

min_price = st.number_input(
    "Minimum Price ($)",
    value=40.0,
    min_value=40.0,
    max_value=110.0,
    key="momentum_min_price"
)

max_price = st.number_input(
    "Maximum Price ($)",
    value=110.0,
    min_value=40.0,
    max_value=110.0,
    key="momentum_max_price"
)

st.markdown("### 🎛️ Momentum Score Filter")

min_momentum_score = st.number_input(
    "Minimum Momentum Score",
    value=4.0,
    min_value=0.0,
    max_value=30.0,
    step=0.5,
    key="momentum_min_score"
)
#PART 3
# =========================================================
# RUN MOMENTUM ENGINE
# =========================================================
run_momentum = st.button(
    "Run Momentum Model Scan",
    key="run_momentum_model"
)

if run_momentum:

    try:

        st.cache_data.clear()

        start_time = time.time()

        eastern = pytz.timezone(
            "US/Eastern"
        )

        now_est = datetime.now(
            eastern
        )

        if now_est.weekday() >= 5:

            st.warning(
                "⚠️ U.S. stock market is closed today."
            )

            st.info(
                "The Momentum Model requires "
                "current-day 1-minute intraday data. "
                "No scan was performed, so Friday's "
                "data cannot appear as Sunday's momentum."
            )

            st.stop()

        st.markdown(
            f"⏱️ Scan Time: "
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

        raw_daily, raw_intra = (
            fetch_clean_market_batch(
                tuple(universe_list)
            )
        )

        progress_bar.progress(
            70,
            text=(
                "Running momentum + "
                "continuation engine..."
            )
        )

        ranking = momentum_rank_universe_batch(
            universe_list,
            raw_daily,
            raw_intra,
            min_price,
            max_price
        )

        if (
            ranking is not None
            and not ranking.empty
        ):

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
            f"⚡ Total Runtime: "
            f"{time.time() - start_time:.2f} seconds"
        )

    except Exception as e:

        try:
            progress_bar.empty()
        except NameError:
            pass

        st.error(
            f"Momentum model execution failed: "
            f"{str(e)}"
        )

        st.exception(e)

# =========================================================
# RENDER RESULTS PANEL
# =========================================================
if (
    "momentum_raw_ranking"
    in st.session_state
):

    ranking = st.session_state[
        "momentum_raw_ranking"
    ]

    if (
        ranking is not None
        and not ranking.empty
    ):

        filtered = ranking.copy()

        filtered = filtered[
            (filtered["Close"] >= min_price)
            &
            (filtered["Close"] <= max_price)
        ]

        filtered = filtered[
            filtered["Momentum_Score"]
            >= min_momentum_score
        ]

        if filtered.empty:

            st.info(
                "No tickers matched your filters."
            )

        else:

            display_df = filtered.copy()

            display_df = display_df.sort_values(
                by=[
                    "Momentum_Score",
                    "Continuation_Score"
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
                f"🔥 Momentum Matrix — "
                f"{len(display_df)} Tickers"
            )

            st.dataframe(
                display_df.style.apply(
                    color_continuation,
                    axis=None
                ),
                hide_index=True,
                use_container_width=True
            )

            if st.button(
                "Reset Momentum Tracker",
                key="reset_momentum_tracker"
            ):

                st.session_state[
                    "momentum_history"
                ] = []

                st.session_state[
                    "momentum_history_date"
                ] = date.today()

                st.session_state[
                    "entry_prices"
                ] = {}

                st.session_state[
                    "stop_limits"
                ] = {}

                st.session_state[
                    "safe_thresholds"
                ] = {}

                st.session_state[
                    "profit_targets"
                ] = {}

                st.success(
                    "Momentum tracker reset."
                )

            if (
                st.session_state[
                    "momentum_history_date"
                ]
                != date.today()
            ):

                st.session_state[
                    "momentum_history"
                ] = []

                st.session_state[
                    "momentum_history_date"
                ] = date.today()

                st.session_state[
                    "entry_prices"
                ] = {}

                st.session_state[
                    "stop_limits"
                ] = {}

                st.session_state[
                    "safe_thresholds"
                ] = {}

                st.session_state[
                    "profit_targets"
                ] = {}

            top5 = display_df.head(5).copy()

            top5["Timestamp"] = (
                datetime.now(
                    pytz.timezone("US/Eastern")
                ).strftime(
                    "%Y-%m-%d %Y-%m-%d %H:%M:%S"
                )
            )

            for idx, row in top5.iterrows():

                ticker = row["Ticker"]

                current_price = row["Close"]

                if (
                    ticker
                    not in st.session_state[
                        "entry_prices"
                    ]
                ):

                    st.session_state[
                        "entry_prices"
                    ][ticker] = current_price

                    st.session_state[
                        "stop_limits"
                    ][ticker] = (
                        current_price
                        * 0.9995
                    )

                    st.session_state[
                        "safe_thresholds"
                    ][ticker] = (
                        current_price
                        * 1.0005
                    )

                    st.session_state[
                        "profit_targets"
                    ][ticker] = (
                        compute_profit_target(
                            current_price,
                            pct=0.015
                        )
                    )

                entry_price = (
                    st.session_state[
                        "entry_prices"
                    ][ticker]
                )

                profit_target_price = (
                    st.session_state[
                        "profit_targets"
                    ][ticker]
                )

                price_position = (
                    compute_price_position(
                        entry_price,
                        current_price
                    )
                )

                exit_signal = (
                    compute_exit_signal(
                        entry_price,
                        current_price
                    )
                )

                profit_target_hit = (
                    current_price
                    >= profit_target_price
                )

                top5.loc[
                    idx,
                    "Price_Position"
                ] = price_position

                top5.loc[
                    idx,
                    "Exit_Signal"
                ] = exit_signal

                top5.loc[
                    idx,
                    "Profit_Target_Hit"
                ] = (
                    "YES"
                    if profit_target_hit
                    else "NO"
                )

            st.session_state[
                "momentum_history"
            ].append(top5)

            history_df = pd.concat(
                st.session_state[
                    "momentum_history"
                ],
                ignore_index=True
            )

            st.subheader(
                "📊 Momentum Timeline — Top 5"
            )

            st.dataframe(
                history_df.style
                .apply(
                    color_continuation,
                    axis=None
                )
                .apply(
                    color_price_position,
                    axis=None
                )
                .apply(
                    color_exit_signal,
                    axis=None
                ),
                hide_index=True,
                use_container_width=True
            )
