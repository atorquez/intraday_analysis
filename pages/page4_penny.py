# ==============================================================================
# 📈 PENNY MODEL — Clean & Patched Version (Standalone Page 4)
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(layout="wide", page_title="Penny Model")
st.caption("Version: V3 — Daily-first pre-filter, threaded intraday fetch")
st.title("📈 Penny Model")

# ==============================================================================
# MODEL PARAMETERS
# ==============================================================================
MIN_DAILY_HISTORY = 40
MIN_INTRADAY_BARS = 5
MIN_REAL_DAY_BARS = 10
MIN_AVG_VOLUME_20D = 80000

# How much slack to give the daily-close price filter before fetching
# intraday data. Intraday price can drift from the prior daily close, so
# the pre-filter range is intentionally a bit wider than the user's actual
# min/max — the final, exact price check still happens later using the
# real intraday price. This just avoids fetching minute bars for tickers
# that have no realistic chance of landing in range.
PREFILTER_PRICE_BUFFER_PCT = 0.15  # 15% slack on each side

# Max concurrent Yahoo requests for the intraday fetch stage.
INTRADAY_MAX_WORKERS = 20

# Where the top-5-per-run log persists across separate script runs (not
# just within one browser session). Lives next to this script file so it
# works the same whether launched via `streamlit run` from the terminal
# or clicked through the sidebar.
import os
TOP5_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top5_log.csv")
TOP5_LOG_COLUMNS = [
    "Run_Timestamp_ET", "Ticker", "Close", "Price_Increase_%_5Bars",
    "Status", "Development_Signal", "Latest_Real_Day", "Stale_Bars_Last5",
]

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "ema_alignment_raw_ranking" not in st.session_state:
    st.session_state["ema_alignment_raw_ranking"] = pd.DataFrame()

if "ema_alignment_rejections" not in st.session_state:
    st.session_state["ema_alignment_rejections"] = pd.DataFrame()

# ==============================================================================
# HELPERS
# ==============================================================================
def _flatten_columns(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(1)
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if all(x in known for x in level0):
            df.columns = level0
        elif all(x in known for x in level1):
            df.columns = level1
    return df

def _extract_ticker_slice(batch, ticker):
    if batch is None or batch.empty:
        return pd.DataFrame()

    ticker = str(ticker).upper()

    if not isinstance(batch.columns, pd.MultiIndex):
        return _flatten_columns(batch.copy())

    try:
        if ticker in batch.columns.get_level_values(0):
            return _flatten_columns(batch[ticker].copy())
    except Exception:
        pass

    try:
        if ticker in batch.columns.get_level_values(1):
            return _flatten_columns(
                batch.xs(ticker, axis=1, level=1, drop_level=True).copy()
            )
    except Exception:
        pass

    return pd.DataFrame()

def _to_eastern_index(df):
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    eastern = ZoneInfo("America/New_York")
    if idx.tz is not None:
        df.index = idx.tz_convert(eastern)
    else:
        df.index = idx.tz_localize(eastern)
    return df

def _load_universe():
    try:
        import utils.data_fetch as data_fetch_module
        import data.us_universe_list as universe_module
        universe_module = importlib.reload(universe_module)
        data_fetch_module = importlib.reload(data_fetch_module)
        load_universe = data_fetch_module.load_universe
        tickers = load_universe()

        st.write("Universe size:", len(tickers))

        if tickers is None:
            st.error("load_universe() returned None.")
            return []
        normalized = sorted(set(str(x).strip().upper() for x in tickers if x))
        return normalized
    except Exception as e:
        st.error(f"Unable to load universe: {e}")
        return []

def _load_movers():
    """Load the trader's curated watchlist of known pre-market/after-hours
    movers from data/movers_list.py — a small, manually-maintained module
    mirroring the pattern of data/us_universe_list.py. Reloaded fresh every
    run so editing the movers list takes effect immediately, no restart
    needed. Missing module or empty list is NOT an error — mover tagging
    is an optional overlay on top of the full-universe scan, so the whole
    page should keep working fine with zero movers loaded.

    Expected file: data/movers_list.py
        def load_movers():
            return ["GLND", "GRML", "ABCD"]  # your curated list for today
    """
    try:
        import data.movers_list as movers_module
        movers_module = importlib.reload(movers_module)
        load_movers = movers_module.load_movers
        movers = load_movers()
        if movers is None:
            return []
        return sorted(set(str(x).strip().upper() for x in movers if x))
    except ModuleNotFoundError:
        # Expected until the trader creates data/movers_list.py — silent,
        # not an error, since mover tagging is optional.
        return []
    except Exception as e:
        st.warning(f"Movers list found but couldn't be loaded: {e}")
        return []

def _rejection_row(ticker, reason=""):
    return {
        "Ticker": ticker,
        "Status": "REJECTED",
        "Reason": reason,
        "Price": np.nan,
        "Opportunity_Category": "N/A",
        "Session_Phase": "N/A",
        "Avg_Volume_20d": np.nan,
        "EMA_Score": np.nan,
        "Price_Increase_%_5Bars": np.nan,
        "Price_Increase_Score": np.nan,
        "Price_Increase_Label": "N/A",
        "Development_Signal": "N/A",
        "Development_Reason": "N/A",
        "Latest_Real_Day": "N/A",
        "Stale_Bars_Last5": np.nan,
    }

def log_top5(ema_only_df, log_path=TOP5_LOG_PATH, top_n=5):
    """Append the current run's top-N tickers (by Price_Increase_%_5Bars,
    which ema_only_df is already sorted by descending) to a persistent CSV
    log, tagged with the run's Eastern-time timestamp. Each run adds a new
    batch of rows on top of prior runs — this is a running history, not a
    snapshot that gets overwritten, so a trader watching the log doesn't
    need to keep separate paper notes for "what showed up 3 minutes ago."
    """
    if ema_only_df is None or ema_only_df.empty:
        return 0

    run_ts = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")

    top = ema_only_df.head(top_n).copy()
    entry = pd.DataFrame({
        "Run_Timestamp_ET": run_ts,
        "Ticker": top["Ticker"].values,
        "Close": top["Close"].values,
        "Price_Increase_%_5Bars": top["Price_Increase_%_5Bars"].values,
        "Status": top["Status"].values,
        "Development_Signal": top["Development_Signal"].values,
        "Latest_Real_Day": top["Latest_Real_Day"].values,
        "Stale_Bars_Last5": top["Stale_Bars_Last5"].values,
    })

    file_exists = os.path.exists(log_path)
    entry.to_csv(log_path, mode="a", header=not file_exists, index=False)
    return len(entry)

def reset_top5_log(log_path=TOP5_LOG_PATH):
    """Wipe the log back to an empty file with just the header row."""
    pd.DataFrame(columns=TOP5_LOG_COLUMNS).to_csv(log_path, index=False)

def load_top5_log(log_path=TOP5_LOG_PATH):
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=TOP5_LOG_COLUMNS)
    try:
        return pd.read_csv(log_path)
    except Exception:
        return pd.DataFrame(columns=TOP5_LOG_COLUMNS)

# ==============================================================================
# MARKET DATA
# ==============================================================================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_daily_batch(tickers_tuple):
    """One batched call for daily history across the whole universe.
    This is cheap regardless of universe size, so it always runs on the
    full ticker list — the expensive per-ticker intraday fetch below is
    what we shrink with the pre-filter."""
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()
    try:
        daily = yf.download(
            tickers, period="3mo", interval="1d",
            group_by="ticker", auto_adjust=False,
            progress=False, threads=True
        )
        return daily
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday_batch(tickers_tuple, max_workers=INTRADAY_MAX_WORKERS):
    """Threaded, per-ticker intraday fetch — call this ONLY with the
    pre-filtered candidate list, not the full universe. Yahoo has no
    reliable batched minute-bar endpoint across arbitrary tickers, so this
    stays per-ticker, but running the requests concurrently instead of
    sequentially cuts wall-clock time roughly by the worker count, and
    shrinking the input list first cuts it further (and avoids Yahoo
    rate-limiting on 1,000+ back-to-back requests)."""
    tickers = list(tickers_tuple)
    if not tickers:
        return pd.DataFrame()

    def _fetch_one(t):
        try:
            df = yf.download(t, period="1d", interval="1m", progress=False)
        except Exception:
            df = pd.DataFrame()
        return t, df

    intra_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            t, df = future.result()

            # Some tickers come back empty (delisted, no data, request
            # failure, etc.). An empty DataFrame's default index is
            # tz-naive, while every ticker that DID return data has a
            # tz-aware DatetimeIndex — mixing the two makes pd.concat
            # below raise "Cannot join tz-naive with tz-aware
            # DatetimeIndex". Simplest fix: just don't include empty
            # frames. Any ticker missing from intra_dict is picked up
            # downstream as "Ticker missing from Yahoo data" anyway.
            if df is None or df.empty:
                continue

            # Extra safety: if a non-empty frame somehow comes back
            # tz-naive (has happened with some yfinance versions/edge
            # cases), localize it to UTC so every frame in the dict
            # shares a consistent tz-aware index before concatenation.
            try:
                if df.index.tz is None:
                    df = df.tz_localize("UTC")
            except (TypeError, AttributeError):
                continue

            intra_dict[t] = df

    if not intra_dict:
        return pd.DataFrame()

    return pd.concat(intra_dict, axis=1)


def daily_prefilter(tickers, daily_batch, min_price, max_price):
    """Use the already-fetched daily batch (free — no extra network calls)
    to shrink the universe down to tickers worth fetching intraday data
    for. Filters on: daily history depth, prior-close price roughly in
    range, and 20-day average volume. This is the step that makes the
    model fast enough to re-run every 2-3 minutes: it turns "1,043
    sequential minute-bar fetches" into "N sequential minute-bar fetches,"
    where N is usually a small fraction of the universe.

    Returns (candidates: list[str], prefilter_rejects: list[dict]).
    """
    candidates = []
    prefilter_rejects = []

    if daily_batch.empty:
        for t in tickers:
            prefilter_rejects.append(_rejection_row(t, "No daily data"))
        return candidates, prefilter_rejects

    lo = min_price * (1 - PREFILTER_PRICE_BUFFER_PCT)
    hi = max_price * (1 + PREFILTER_PRICE_BUFFER_PCT)

    for ticker in tickers:
        daily = _flatten_columns(_extract_ticker_slice(daily_batch, ticker))

        if daily.empty:
            prefilter_rejects.append(_rejection_row(ticker, "No daily data"))
            continue

        daily = daily.dropna(subset=["Close"])

        if len(daily) < MIN_DAILY_HISTORY:
            prefilter_rejects.append(_rejection_row(ticker, "Insufficient daily history"))
            continue

        last_close = pd.to_numeric(daily["Close"], errors="coerce").dropna()
        if last_close.empty:
            prefilter_rejects.append(_rejection_row(ticker, "No valid daily close"))
            continue
        last_close = float(last_close.values[-1])

        if last_close < lo or last_close > hi:
            prefilter_rejects.append(_rejection_row(ticker, "Price outside range (daily pre-filter)"))
            continue

        vol = pd.to_numeric(daily["Volume"], errors="coerce").dropna().values.astype(float)
        if len(vol) < 20:
            prefilter_rejects.append(_rejection_row(ticker, "Insufficient volume history"))
            continue

        avg_vol = float(np.mean(vol[-20:]))
        if avg_vol < MIN_AVG_VOLUME_20D:
            prefilter_rejects.append(_rejection_row(ticker, "Low volume (daily pre-filter)"))
            continue

        candidates.append(ticker)

    return candidates, prefilter_rejects

# ==============================================================================
# SCORING
# ==============================================================================
def price_increase_score(pct):
    try:
        x = float(pct)
    except Exception:
        return 0
    if x >= 1.0: return 3
    if x >= 0.5: return 2
    if x >= 0.25: return 1
    return 0

def opportunity_category(gap):
    try:
        x = float(gap)
    except Exception:
        return "Unknown"
    if x <= -5: return "Deep Recovery"
    if x < -2: return "Moderate Recovery"
    if x <= 2: return "Near Previous Close"
    if x <= 5: return "Continuation"
    return "Extended"

def session_phase(ts):
    m = ts.hour * 60 + ts.minute
    if m < 570: return "Pre-Market"
    if m < 610: return "Early Session"
    if m < 630: return "Morning"
    if m < 720: return "Late Morning"
    if m < 840: return "Midday"
    return "Afternoon"

# ==============================================================================
# EMA ALIGNMENT ENGINE
# ==============================================================================
def ema_alignment_engine(tickers, daily_batch, intra_batch, min_price, max_price, max_stale_bars_last5=2):
    rows = []
    rejects = []
    MAX_STALE_BARS_LAST5 = max_stale_bars_last5

    if daily_batch.empty:
        for t in tickers:
            rejects.append(_rejection_row(t, "No daily data"))
        return pd.DataFrame(), pd.DataFrame(rejects)

    if intra_batch.empty:
        for t in tickers:
            rejects.append(_rejection_row(t, "No intraday data"))
        return pd.DataFrame(), pd.DataFrame(rejects)

    # --- Pre-filter tickers missing from the Yahoo batch entirely -----------
    # FIX: previously referenced undefined `daily`/`intra` (only the params
    # `daily_batch`/`intra_batch` existed at this point), which raised a
    # NameError caught by the bare except below and silently fell back to
    # `active = sorted(tickers)` every single run. Now correctly checks the
    # actual batch frames passed into the function.
    try:
        available_daily = set(daily_batch.columns.get_level_values(0))
        available_intra = set(intra_batch.columns.get_level_values(0))
        active = sorted(set(tickers).intersection(available_daily).intersection(available_intra))
    except Exception:
        active = sorted(tickers)

    for t in sorted(set(tickers) - set(active)):
        rejects.append(_rejection_row(t, "Ticker missing from Yahoo data"))

    for ticker in active:
        r = _rejection_row(ticker)

        daily = _flatten_columns(_extract_ticker_slice(daily_batch, ticker))
        intra = _flatten_columns(_extract_ticker_slice(intra_batch, ticker))

        if daily.empty:
            r["Reason"] = "Daily empty"
            rejects.append(r)
            continue

        if intra.empty:
            r["Reason"] = "Intraday empty"
            rejects.append(r)
            continue

        daily = daily.dropna(subset=["Close"])
        intra = intra[intra["Close"].notna() | intra["Open"].notna()]

        if len(daily) < MIN_DAILY_HISTORY:
            r["Reason"] = "Insufficient daily history"
            rejects.append(r)
            continue

        intra = _to_eastern_index(intra).sort_index()

        intra_regular = intra.between_time("09:30", "16:00")
        intra_regular = intra_regular.loc[~intra_regular.index.duplicated(keep='last')]

        # REGULAR-HOURS ONLY — no extended-hours fallback
        if len(intra_regular) < MIN_REAL_DAY_BARS:
            r["Reason"] = f"Insufficient regular-hours bars ({len(intra_regular)})"
            rejects.append(r)
            continue

        intra_day = intra_regular

        if intra_day.empty:
            r["Reason"] = "No intraday bars (REGULAR)"
            rejects.append(r)
            continue

        # Determine eligible regular-hours trading days
        day_counts = pd.Series(intra_day.index.date).value_counts()

        # Only accept days with enough regular-hours bars
        eligible = sorted(day_counts[day_counts >= MIN_REAL_DAY_BARS].index)

        if not eligible:
            r["Reason"] = "No eligible regular-hours trading day"
            rejects.append(r)
            continue

        latest_day = eligible[-1]

        # Filter intraday data to the latest eligible day
        intra = intra_day[intra_day.index.date == latest_day]
        intra = intra.loc[~intra.index.duplicated(keep='last')]

        if len(intra) < MIN_REAL_DAY_BARS:
            r["Reason"] = "Insufficient bars on eligible regular-hours day"
            rejects.append(r)
            continue

        close_series_clean = pd.to_numeric(intra["Close"], errors="coerce").dropna()
        close_raw = close_series_clean.values.astype(float)

        if len(close_raw) < MIN_INTRADAY_BARS:
            r["Reason"] = "Too few intraday bars"
            rejects.append(r)
            continue

        price = float(close_raw[-1])
        r["Price"] = round(price, 2)

        if price < min_price or price > max_price:
            r["Reason"] = "Price outside range"
            rejects.append(r)
            continue

        vol = pd.to_numeric(daily["Volume"], errors="coerce").dropna().values.astype(float)
        if len(vol) < 20:
            r["Reason"] = "Insufficient volume history"
            rejects.append(r)
            continue

        avg_vol = float(np.mean(vol[-20:]))
        r["Avg_Volume_20d"] = round(avg_vol, 0)

        if avg_vol < MIN_AVG_VOLUME_20D:
            r["Reason"] = "Low volume"
            rejects.append(r)
            continue

        # --- Stale (no-trade) bar check -------------------------------------
        # Thinly-traded penny stocks often have 1-minute windows with zero
        # trades. Yahoo doesn't drop or NaN those bars — it forward-fills
        # OHLC with the last traded price (Volume = 0), which is why you'll
        # sometimes see Bar4_Close == Bar5_Close. A "5-bar breakout" built
        # mostly out of forward-filled bars is a weaker signal: the price
        # "held" because nothing traded, not because buyers defended it.
        # We check intraday minute-bar Volume (not the daily 20d average
        # already checked above) on exactly the same 5 bars used for last5,
        # via the shared index, so this lines up 1:1 with what's displayed.
        last5_index = close_series_clean.tail(5).index
        bar_vol_last5 = pd.to_numeric(intra["Volume"], errors="coerce").reindex(last5_index).fillna(0.0)
        stale_bars_last5 = int((bar_vol_last5 <= 0).sum())

        if stale_bars_last5 > MAX_STALE_BARS_LAST5:
            r["Reason"] = f"Too many no-trade bars in last 5 ({stale_bars_last5})"
            rejects.append(r)
            continue

        ema9_series = intra["Close"].ewm(span=9, adjust=False).mean()
        ema20_series = intra["Close"].ewm(span=20, adjust=False).mean()

        last5 = close_raw[-5:]

        ema9_last5 = ema9_series.tail(5).values.astype(float)
        ema20_last5 = ema20_series.tail(5).values.astype(float)

        Bar4_EMA9 = ema9_last5[-2]
        Bar5_EMA9 = ema9_last5[-1]
        Bar4_EMA20 = ema20_last5[-2]
        Bar5_EMA20 = ema20_last5[-1]

        ema9_now = Bar5_EMA9
        ema20_now = Bar5_EMA20
        ema9_prev = Bar4_EMA9
        ema20_prev = Bar4_EMA20

        ema9_slope = ema9_now - ema9_prev
        ema20_slope = ema20_now - ema20_prev

        x = np.array([1, 2, 3, 4, 5], dtype=float)
        reg_slope = np.polyfit(x, last5, 1)[0]

        # ============================
        # Consistency score (0 to 1)
        # Measures how stable the 5-bar trend is
        # ============================
        diffs = np.diff(last5)
        up_moves = np.sum(diffs > 0)
        down_moves = np.sum(diffs < 0)

        # Consistency = proportion of bars moving in the dominant direction
        if up_moves >= down_moves:
            consistency_score = up_moves / 4.0
        else:
            consistency_score = down_moves / 4.0

        last5_above_ema9 = bool(np.all(last5 > ema9_last5))
        pressure = (last5[-1] - ema9_now) / last5[-1]

        acceleration = (
            (reg_slope > 0) and
            (ema9_slope > 0) and
            last5_above_ema9
        )

        recent2_above = np.all(last5[-2:] > ema9_last5[-2:])
        preceding3_below = np.all(last5[:3] <= ema9_last5[:3])
        previous4_below = np.all(last5[:-1] <= ema9_last5[:-1])

        dev_one = price > ema9_now and previous4_below
        dev_two = recent2_above and preceding3_below
        dev_cross = dev_one or dev_two

        dev_base = (
            price > ema20_now and
            ema9_now > ema20_now and
            ema9_slope > 0 and
            ema20_slope > 0
        )

        dev_signal = dev_base and dev_cross and acceleration

        # ============================
        # Strength thresholds
        # FIX: slopes and regression slope are now expressed as a fraction
        # of price (like `pressure` already was) instead of raw dollar
        # amounts. Raw-dollar thresholds made the filter far stricter for
        # low-priced names and far looser for higher-priced names within
        # the same $1-$10 scan range, which isn't apples-to-apples for a
        # penny-stock screener.
        # ============================
        MIN_EMA9_SLOPE_PCT = 0.0015       # 0.15% of price per bar
        MIN_EMA20_SLOPE_PCT = 0.0008      # 0.08% of price per bar
        MIN_REGRESSION_SLOPE_PCT = 0.0012 # 0.12% of price per bar
        MIN_PRESSURE = 0.003
        MIN_CONSISTENCY = 0.60

        ema9_slope_pct = ema9_slope / price if price else 0
        ema20_slope_pct = ema20_slope / price if price else 0
        reg_slope_pct = reg_slope / price if price else 0

        # ============================
        # EMA (strong alignment)
        # ============================
        cond_align = (
            price > ema9_now and
            price > ema20_now and
            ema9_now > ema20_now and
            ema9_slope_pct > MIN_EMA9_SLOPE_PCT and
            ema20_slope_pct > MIN_EMA20_SLOPE_PCT and
            reg_slope_pct > MIN_REGRESSION_SLOPE_PCT and
            pressure > MIN_PRESSURE and
            consistency_score >= MIN_CONSISTENCY
        )

        # ============================
        # Classification (EMA only)
        # ============================
        if not cond_align:
            r["Reason"] = "Failed EMA criteria"
            rejects.append(r)
            continue

        status = "EMA"
        pct = (last5[-1] - last5[0]) / last5[0] * 100 if last5[0] > 0 else 0

        bar_vol_values = bar_vol_last5.values.astype(float)

        rows.append({
            "Ticker": ticker,
            "Close": round(price, 2),
            "Bar1_Close": round(last5[0], 4),
            "Bar2_Close": round(last5[1], 4),
            "Bar3_Close": round(last5[2], 4),
            "Bar4_Close": round(last5[3], 4),
            "Bar5_Close": round(last5[4], 4),
            # Per-bar (1-minute) traded volume — NOT the 20-day daily
            # average. This is the actual shares that traded in each of
            # the last 5 individual minute bars, so you can see the
            # difference between e.g. a ~50-share print that barely moves
            # price vs. a ~1,000+ share print that does. Distinct from
            # Stale_Bars_Last5 (which only asks "was it > 0"), this shows
            # magnitude, not just presence/absence of trading.
            "Bar1_Volume": int(bar_vol_values[0]),
            "Bar2_Volume": int(bar_vol_values[1]),
            "Bar3_Volume": int(bar_vol_values[2]),
            "Bar4_Volume": int(bar_vol_values[3]),
            "Bar5_Volume": int(bar_vol_values[4]),
            "Bar4_EMA9": round(Bar4_EMA9, 4),
            "Bar4_EMA20": round(Bar4_EMA20, 4),
            "Bar5_EMA9": round(Bar5_EMA9, 4),
            "Bar5_EMA20": round(Bar5_EMA20, 4),
            "Price_Increase_%_5Bars": round(pct, 3),
            "Development_Signal": bool(dev_signal),
            "Status": status,
            # Which session these bars actually came from. yfinance's
            # period="1d" returns the most recently COMPLETED session, not
            # strictly "today" — e.g. on a Sunday it silently hands back
            # Friday's bars. This column makes that visible in the
            # dataframe instead of it being an invisible assumption.
            "Latest_Real_Day": str(latest_day),
            # Count of bars among the last 5 with zero intraday volume —
            # i.e. no actual trade happened, Yahoo forward-filled the price
            # from the prior bar. High values mean the "5-bar move" is
            # partly/mostly not real trading activity.
            "Stale_Bars_Last5": stale_bars_last5,
        })
        # NOTE: loop continues to the next ticker here — this is the critical
        # fix. Previously the final-assembly/return block below was indented
        # one level too deep (inside this `for` loop), so the function
        # returned after the FIRST ticker that passed the EMA filter and
        # every ticker after it in the universe was never evaluated.

    # ============================
    # Final assembly and return
    # FIX: moved OUTSIDE the `for ticker in active:` loop so the full
    # universe is scanned before returning.
    # ============================
    ranking = pd.DataFrame(rows)
    rejects_df = pd.DataFrame(rejects)

    if not ranking.empty:
        ranking = ranking.sort_values(
            ["Price_Increase_%_5Bars", "Bar5_Close", "Ticker"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

    return ranking, rejects_df

# ==============================================================================
# PAGE EXECUTION
# ==============================================================================
# ==============================================================================
# MARKET-DAY / DATA-FRESHNESS CHECK
# ==============================================================================
# yfinance's period="1d" intraday request returns the most recently
# COMPLETED session's bars, not "today's bars or nothing." Outside market
# hours (weekends, holidays, pre-open) that means every ticker that scores
# is being scored on a STALE session (e.g. last Friday), which looks
# identical in the table to a live signal unless you check the date. This
# banner makes the distinction explicit so results are never mistaken for
# live intraday signals when the market isn't actually open.
_eastern = ZoneInfo("America/New_York")
_now_et = datetime.now(_eastern)
_is_weekday = _now_et.weekday() < 5  # Mon=0 ... Sun=6
_minutes_now = _now_et.hour * 60 + _now_et.minute
_is_market_hours = _is_weekday and (570 <= _minutes_now < 960)  # 09:30-16:00 ET

if _is_market_hours:
    st.success(f"🟢 Market is open — {_now_et.strftime('%A %Y-%m-%d %H:%M %Z')}. Results below reflect live/current-session bars.")
else:
    st.warning(
        f"🟠 Market is CLOSED right now — {_now_et.strftime('%A %Y-%m-%d %H:%M %Z')}. "
        "Any tickers that qualify below were scored on the most recent completed "
        "session (check the Latest_Real_Day column), not live data. Treat this as a "
        "historical/backtest-style run, not a trading signal."
    )

tickers = _load_universe()

movers = _load_movers()
if movers:
    st.caption(f"🔥 Movers list loaded: {len(movers)} ticker(s) — {', '.join(movers)}")
else:
    st.caption("No movers list loaded (optional). Add `data/movers_list.py` with a `load_movers()` function to enable mover tagging.")

st.write("### 🔍 Price Boundaries Filter")
min_price = st.number_input("Minimum Price ($)", min_value=0.0, value=1.00, step=0.25)
max_price = st.number_input("Maximum Price ($)", min_value=0.0, value=10.00, step=0.25)

max_stale_bars_last5 = st.slider(
    "Max no-trade bars allowed in last 5",
    min_value=0, max_value=5, value=2,
    help=(
        "Thinly-traded penny stocks can have 1-minute bars with zero volume, "
        "where Yahoo just repeats the last traded price (you'll see "
        "Bar4_Close == Bar5_Close). Tickers with more no-trade bars than this "
        "in their last 5 are rejected — 'Too many no-trade bars in last 5' in "
        "the diagnostics. 0 = require every one of the last 5 bars to have a "
        "real trade; 5 = disable this check entirely."
    ),
)

import time

# ==============================================================================
# RUN BUTTON
# ==============================================================================
# The full pipeline (daily fetch + pre-filter + threaded intraday fetch +
# engine) is expensive — tens of seconds even with caching helping on
# reruns within the same 2-minute TTL window. Without a run button,
# Streamlit re-executes this whole script top-to-bottom on EVERY widget
# interaction anywhere on the page — including unrelated ones, like
# picking a ticker in the rejection-diagnostics selectbox further down —
# which means the entire fetch pipeline would silently refire just from
# clicking around the page, not just when you actually want a fresh scan.
# Gating it behind a button, and caching the results in session_state, so
# the model only runs when you explicitly ask it to, and other widget
# interactions on the page just redraw from the last run's stored results.
run_clicked = st.button("🚀 Run Model", type="primary")

if run_clicked:
    t0 = time.time()
    daily_batch = fetch_daily_batch(tuple(tickers))
    t1 = time.time()

    candidates, prefilter_rejects = daily_prefilter(tickers, daily_batch, min_price, max_price)
    t2 = time.time()

    intra_batch = fetch_intraday_batch(tuple(candidates))
    t3 = time.time()

    ranking, engine_rejects = ema_alignment_engine(
        candidates, daily_batch, intra_batch, min_price, max_price,
        max_stale_bars_last5=max_stale_bars_last5,
    )
    t4 = time.time()

    rejects = pd.concat([pd.DataFrame(prefilter_rejects), engine_rejects], ignore_index=True)

    # Persist everything the display sections below need, so they can
    # render from session_state on reruns triggered by OTHER widgets
    # (e.g. the rejection-ticker selectbox) without re-fetching anything.
    st.session_state["model_results"] = {
        "ranking": ranking,
        "rejects": rejects,
        "tickers": tickers,
        "n_candidates": len(candidates),
        "n_universe": len(tickers),
        "timings": (t0, t1, t2, t3, t4),
        "run_at_et": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

results = st.session_state.get("model_results")

if results is None:
    st.info("Click **🚀 Run Model** above to fetch data and run the scan.")
    ranking = pd.DataFrame()
    rejects = pd.DataFrame()
else:
    ranking = results["ranking"]
    rejects = results["rejects"]
    run_universe = results["tickers"]
    t0, t1, t2, t3, t4 = results["timings"]
    st.caption(f"Last run: {results['run_at_et']}")
    st.write(
        f"### ⏱️ Runtime: {t4 - t0:.2f}s total "
        f"(daily fetch {t1 - t0:.2f}s · pre-filter {t2 - t1:.2f}s · "
        f"intraday fetch {t3 - t2:.2f}s [{results['n_candidates']}/{results['n_universe']} tickers] · "
        f"engine {t4 - t3:.2f}s)"
    )

    # Tag every ranking/rejects row with whether it's on the curated
    # movers list, recomputed fresh from the CURRENT movers list on every
    # script rerun (not frozen at click time) — so editing
    # data/movers_list.py between runs updates tags immediately without
    # needing to click Run Model again.
    movers_set = set(movers)
    if not ranking.empty:
        ranking = ranking.copy()
        ranking["Watchlist"] = ranking["Ticker"].apply(lambda t: "Movers List" if t in movers_set else "Not Movers List")
    if not rejects.empty:
        rejects = rejects.copy()
        rejects["Watchlist"] = rejects["Ticker"].apply(lambda t: "Movers List" if t in movers_set else "Not Movers List")

    # ==========================================================================
    # MOVERS SCORECARD — the direct answer to "what happened to each of my
    # known movers this run?" One row per curated ticker: qualified (with
    # its numbers), rejected (with the exact reason — this is what would
    # have told you immediately why GRML-style misses happen), or not even
    # in the scanned universe at all (catches foreign-listing / wrong-
    # exchange gaps like GRML directly, instead of discovering it after
    # the fact from an outside chart).
    # ==========================================================================
    if movers:
        st.write("### 🔥 Movers List Scorecard")
        scorecard_rows = []
        for m in movers:
            if m not in run_universe:
                scorecard_rows.append({"Ticker": m, "Status": "⚠️ Not in scanned universe (foreign listing, wrong exchange, or delisted — check the ticker)"})
                continue
            if not ranking.empty and m in ranking["Ticker"].values:
                row = ranking[ranking["Ticker"] == m].iloc[0]
                scorecard_rows.append({
                    "Ticker": m,
                    "Status": f"✅ Qualified — Close ${row['Close']}, {row['Price_Increase_%_5Bars']}% (5 bars), Stale bars: {row.get('Stale_Bars_Last5', 'N/A')}",
                })
                continue
            if not rejects.empty and m in rejects["Ticker"].values:
                reasons = rejects[rejects["Ticker"] == m]["Reason"].dropna().unique().tolist()
                scorecard_rows.append({"Ticker": m, "Status": f"❌ Rejected — {'; '.join(reasons) if reasons else 'reason not recorded'}"})
                continue
            scorecard_rows.append({"Ticker": m, "Status": "❔ Not evaluated (in universe but no result recorded this run)"})
        st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# DISPLAY RESULTS (EMA ONLY)
# ==============================================================================
if results is not None:
    if ranking.empty:
        st.warning("No tickers qualified for EMA alignment.")
    else:
        st.write("### 📊 EMA Alignment (Strong)")
        ema_only = ranking[ranking["Status"] == "EMA"]
        if ema_only.empty:
            st.info("No EMA Alignment tickers.")
        else:
            st.dataframe(ema_only, use_container_width=True)
            n_logged = log_top5(ema_only)
            st.caption(f"📝 Logged top {n_logged} (by Price_Increase_%_5Bars) to `{os.path.basename(TOP5_LOG_PATH)}` at this run's timestamp.")

# ==============================================================================
# TOP-5 RUNNING LOG
# ==============================================================================
st.write("### 📝 Top-5 Log (across runs)")

log_col1, log_col2 = st.columns([3, 1])
with log_col2:
    if "confirm_reset_log" not in st.session_state:
        st.session_state["confirm_reset_log"] = False

    if not st.session_state["confirm_reset_log"]:
        if st.button("Reset Log"):
            st.session_state["confirm_reset_log"] = True
    else:
        st.warning("This clears the entire log. Confirm?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, clear it"):
                reset_top5_log()
                st.session_state["confirm_reset_log"] = False
                st.success("Log cleared.")
        with c2:
            if st.button("Cancel"):
                st.session_state["confirm_reset_log"] = False

top5_log_df = load_top5_log()
with log_col1:
    if top5_log_df.empty:
        st.info("No entries logged yet. Run the model at least once with qualifying tickers to start building history.")
    else:
        st.dataframe(top5_log_df.sort_values("Run_Timestamp_ET", ascending=False), use_container_width=True)
        st.caption(f"{len(top5_log_df)} total logged entries · file: `{TOP5_LOG_PATH}`")

# ==============================================================================
# REJECTION DIAGNOSTICS
# ==============================================================================
if results is not None:
    st.write("### 🧪 Rejection Diagnostics")
    st.write(f"Total Rejects: {len(rejects)}")

    if rejects.empty:
        st.info("No rejections recorded.")
    else:
        st.write("### Rejection Summary")
        st.write(rejects["Reason"].value_counts())

        ticker_list = sorted(rejects["Ticker"].unique())
        selected = st.selectbox("Select ticker for rejection diagnostics", ticker_list)

        diag = rejects[rejects["Ticker"] == selected]
        st.dataframe(diag, use_container_width=True)