import pandas as pd

def calculate_or15(df):
    """
    Compute the Opening Range 15 (OR15) values.
    Assumes df is a standard intraday OHLCV DataFrame.
    """
    if df is None or df.empty or len(df) < 15:
        return None

    or15 = df.iloc[:15]

    return {
        "OR15_High": or15["High"].max(),
        "OR15_Low": or15["Low"].min(),
        "OR15_Open": or15["Open"].iloc[0],
        "OR15_Close": or15["Close"].iloc[-1]
    }
