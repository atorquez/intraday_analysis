# ============================================================
# INTRADAY RANKER — BUYZONE + PCA + VMAS + BUY SIGNAL
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# INTRADAY BUY ZONE (Daily Value Zone for Page 1)
# ---------------------------------------------------------
def intraday_buy_zone(df, lookback=10, percentile=0.15):
    if df is None or df.empty:
        return None

    closes = df["Close"].tail(lookback)
    if len(closes) < lookback:
        return None

    return np.percentile(closes, percentile * 100)

# ---------------------------------------------------------
# COMPANY NAME LOOKUP
# ---------------------------------------------------------
def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName", ticker)
    except Exception:
        return ticker

# ---------------------------------------------------------
# FETCH DAILY DATA
# ---------------------------------------------------------
def fetch_daily(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df

# ---------------------------------------------------------
# FETCH INTRADAY DATA (1-minute)
# ---------------------------------------------------------
def fetch_intraday(ticker):
    try:
        df = yf.download(
            tickers=ticker,
            period="1d",
            interval="1m",
            progress=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        return df

    except Exception:
        return None

# ---------------------------------------------------------
# EXTRA INDICATORS FOR PCA
# ---------------------------------------------------------
def compute_extra_indicators(df):
    df = df.copy()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_Width"] = (df["STD20"] * 2) / df["SMA20"]

    df["ROC"] = df["Close"].pct_change(10)

    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    df["StochK"] = (df["Close"] - low_min) / (high_max - low_min)

    df["EMA_Curve"] = df["Close"].ewm(span=9).mean() - df["Close"].ewm(span=20).mean()

    df["VolDelta"] = df["Volume"].diff()

    df["VWAP"] = (df["Volume"] * df["Close"]).cumsum() / df["Volume"].cumsum()
    df["VWAP_Dist"] = df["Close"] - df["VWAP"]

    df = df.dropna()
    return df

# ---------------------------------------------------------
# PCA COMPONENTS (WITH STANDARDSCALER)
# ---------------------------------------------------------
def compute_pca_components(df):
    df = compute_extra_indicators(df)

    pca_features = df[[
        "RSI",
        "BB_Width",
        "ROC",
        "StochK",
        "EMA_Curve",
        "VolDelta",
        "VWAP_Dist"
    ]].dropna()

    if len(pca_features) < 10:
        return None, None, None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pca_features)

    pca = PCA(n_components=3)
    components = pca.fit_transform(scaled)

    pca1, pca2, pca3 = components[-1]
    return float(pca1), float(pca2), float(pca3)

# ---------------------------------------------------------
# INDICATORS (DAILY)
# ---------------------------------------------------------
def calculate_indicators(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()
    df["ATR%"] = (df["ATR"] / df["Close"]) * 100

    df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Gap%"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100

    last = df.iloc[-1]

    if last["EMA9"] > last["EMA20"] > last["EMA50"]:
        trend = "UP"
    elif last["EMA9"] < last["EMA20"] < last["EMA50"]:
        trend = "DOWN"
    else:
        trend = "FLAT"

    ema9 = last["EMA9"]
    ema20 = last["EMA20"]
    ema9_slope = df["EMA9"].iloc[-1] - df["EMA9"].iloc[-5]
    ema20_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-5]

    try:
        pca1 = df["PCA1"].iloc[-1]
    except Exception:
        pca1 = None

    if ema9 > ema20 and ema9_slope > 0 and ema20_slope > 0 and trend == "UP" and (pca1 is None or pca1 > 0):
        daily_execution = "Ready"
    elif ema9 > ema20 and (ema9_slope < 0 or ema20_slope < 0):
        daily_execution = "False Ready"
    elif abs((ema9 - ema20) / ema20) < 0.003:
        daily_execution = "Crossing Soon"
    else:
        daily_execution = "Setup Only"

    return {
        "Close": round(last["Close"], 2),
        "ATR%": round(last["ATR%"], 2),
        "RVOL": round(last["RVOL"], 2),
        "Gap%": round(last["Gap%"], 2),
        "EMA9": ema9,
        "EMA20": ema20,
        "EMA50": last["EMA50"],
        "Trend": trend,
        "Execution_Status": daily_execution
    }

# ---------------------------------------------------------
# INTRADAY EXECUTION ENGINE
# ---------------------------------------------------------
def calculate_intraday_execution(df_intraday):
    """
    df_intraday must be 1‑minute or 5‑minute data with columns:
    ['Open','High','Low','Close','Volume']
    """
    df = df_intraday.copy()

    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    ema9_slope = df["EMA9"].iloc[-1] - df["EMA9"].iloc[-3]
    ema20_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-3]

    df["ROC"] = df["Close"].pct_change() * 100
    intraday_momentum = df["ROC"].iloc[-1]

    if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1]:
        intraday_trend = "UP"
    else:
        intraday_trend = "DOWN"

    return {
        "intraday_ema9": df["EMA9"].iloc[-1],
        "intraday_ema20": df["EMA20"].iloc[-1],
        "intraday_ema9_slope": ema9_slope,
        "intraday_ema20_slope": ema20_slope,
        "intraday_momentum": intraday_momentum,
        "intraday_trend": intraday_trend
    }

# ---------------------------------------------------------
# SCORING ENGINE
# ---------------------------------------------------------
def score_stock(ind):
    score = 0

    if ind["EMA9"] > ind["EMA20"]:
        score += 10
    if ind["EMA20"] > ind["EMA50"]:
        score += 10

    if ind["RVOL"] > 2:
        score += 20

    if 2 <= ind["Gap%"] <= 5:
        score += 10

    if ind["ATR%"] > 5:
        score += 15

    score += 5
    return score

# ---------------------------------------------------------
# VMAS
# ---------------------------------------------------------
def compute_vmas(price, buyzone10, pca1):
    if price is None or buyzone10 is None or pca1 is None:
        return None

    dist = (price - buyzone10) / buyzone10
    return (1 - dist) * pca1

# ---------------------------------------------------------
# BUYZONE HEATMAP
# ---------------------------------------------------------
def buyzone_heatmap(price, bz10, bz5, dist10):
    if price is None or bz10 is None or bz5 is None:
        return "Unknown"

    if price <= bz5:
        return "Inside_5"

    if price <= bz10:
        return "Inside_10"

    if dist10 is not None and dist10 < 3:
        return "Near_Value"

    if dist10 is not None and dist10 > 10:
        return "Extended"

    return "Normal"

# ---------------------------------------------------------
# BUY SIGNAL ENGINE
# ---------------------------------------------------------
def compute_buy_signal(price, buyzone10, buyzone5, dist10, pca1, trend):
    if price is None or buyzone10 is None or buyzone5 is None:
        return "Avoid"

    if dist10 is not None and dist10 > 10:
        return "Extended"

    if pca1 is None or pca1 < -1.0:
        return "Avoid"

    if price <= buyzone5 and pca1 > 0 and trend == "UP":
        return "Strong Buy Zone"

    if price <= buyzone10 and pca1 > 0:
        return "Buy Zone"

    if dist10 is not None and dist10 < 3 and pca1 > -0.5:
        return "Neutral"

    return "Avoid"

# ---------------------------------------------------------
# RANK UNIVERSE
# ---------------------------------------------------------
def rank_universe(symbols):
    results = []

    for ticker in symbols:
        df = fetch_daily(ticker)

        if df is None or len(df) < 50:
            results.append({
                "Ticker": ticker,
                "Name": get_company_name(ticker),
                "Score": None,
                "Price": None,
                "ATR%": None,
                "RVOL": None,
                "Gap%": None,
                "Trend": "UNKNOWN",
                "BuyZone10": None,
                "BuyZone5": None,
                "BuyZone10_Distance%": None,
                "BuyZone5_Distance%": None,
                "PCA1": None,
                "PCA2": None,
                "PCA3": None,
                "VMAS": None,
                "BuyZone_Heatmap": "Unknown",
                "Buy_Signal": "Avoid",
                "Execution_Status": "UNKNOWN"
            })
            continue

        # DAILY INDICATORS
        ind = calculate_indicators(df)
        score = score_stock(ind)

        # BUYZONE
        buyzone10 = intraday_buy_zone(df, lookback=10, percentile=0.15)
        buyzone5 = intraday_buy_zone(df, lookback=5, percentile=0.15)

        distance10 = ((ind["Close"] - buyzone10) / buyzone10 * 100) if buyzone10 else None
        distance5 = ((ind["Close"] - buyzone5) / buyzone5 * 100) if buyzone5 else None

        # PCA
        pca1, pca2, pca3 = compute_pca_components(df)

        # VMAS
        vmas = compute_vmas(ind["Close"], buyzone10, pca1)

        # HEATMAP
        heatmap = buyzone_heatmap(ind["Close"], buyzone10, buyzone5, distance10)

        # BUY SIGNAL
        buy_signal = compute_buy_signal(
            ind["Close"], buyzone10, buyzone5, distance10, pca1, ind["Trend"]
        )

        # INTRADAY EXECUTION FILTER
        intraday_df = fetch_intraday(ticker)

        if intraday_df is not None and len(intraday_df) > 10:
            intraday = calculate_intraday_execution(intraday_df)

            i_ema9 = intraday["intraday_ema9"]
            i_ema20 = intraday["intraday_ema20"]
            i_slope9 = intraday["intraday_ema9_slope"]
            i_slope20 = intraday["intraday_ema20_slope"]
            i_momentum = intraday["intraday_momentum"]
            i_trend = intraday["intraday_trend"]

            daily_exec = ind["Execution_Status"]

            if (
                daily_exec == "Ready"
                and i_ema9 > i_ema20
                and i_slope9 > 0
                and i_slope20 > 0
                and i_momentum > 0
                and i_trend == "UP"
            ):
                execution_status = "Ready"

            elif (
                daily_exec == "Ready"
                and (i_slope9 < 0 or i_slope20 < 0 or i_trend == "DOWN")
            ):
                execution_status = "Intraday False Ready"

            elif (
                daily_exec == "Crossing Soon"
                and abs((i_ema9 - i_ema20) / i_ema20) < 0.003
            ):
                execution_status = "Crossing Soon"

            else:
                execution_status = daily_exec
        else:
            execution_status = ind["Execution_Status"]

        # APPEND RESULT
        results.append({
            "Ticker": ticker,
            "Name": get_company_name(ticker),
            "Score": score,
            "Price": ind["Close"],
            "ATR%": ind["ATR%"],
            "RVOL": ind["RVOL"],
            "Gap%": ind["Gap%"],
            "Trend": ind["Trend"],
            "BuyZone10": buyzone10,
            "BuyZone5": buyzone5,
            "BuyZone10_Distance%": round(distance10, 2) if distance10 else None,
            "BuyZone5_Distance%": round(distance5, 2) if distance5 else None,
            "PCA1": pca1,
            "PCA2": pca2,
            "PCA3": pca3,
            "VMAS": vmas,
            "BuyZone_Heatmap": heatmap,
            "Buy_Signal": buy_signal,
            "Execution_Status": execution_status
        })

    ranking = (
        pd.DataFrame(results)
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )

    return ranking






