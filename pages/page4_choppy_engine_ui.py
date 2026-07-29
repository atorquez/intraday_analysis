import numpy as np
import pandas as pd

# ---------------------------------------------------------
# EMA Compression Engine
# ---------------------------------------------------------
def calculate_ema_compression(df):
    """
    Compute EMA compression signals.
    df must contain Close and Volume columns.
    """
    try:
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()

        df['EMA_Spread'] = (df['EMA9'] - df['EMA50']) / df['EMA50']

        # Structural compression condition
        df['Structural_Ready'] = (
            (df['EMA9'] > df['EMA20']) &
            (df['EMA20'] > df['EMA50']) &
            (df['EMA_Spread'].abs() < 0.02)
        )

        return df

    except Exception:
        return df


# ---------------------------------------------------------
# Residual PCA Engine
# ---------------------------------------------------------
def calculate_residual_pca(ticker_df, spy_df):
    """
    Compute PCA residual between ticker and SPY.
    Both inputs must be aligned on the same index.
    """
    try:
        X = np.vstack([
            ticker_df['Close'].values,
            ticker_df['Volume'].values
        ]).T

        Y = np.vstack([
            spy_df['Close'].values,
            spy_df['Volume'].values
        ]).T

        # Residual = ticker - spy
        residual = X - Y

        # PCA on residual
        residual_centered = residual - residual.mean(axis=0)
        U, S, Vt = np.linalg.svd(residual_centered, full_matrices=False)
        pca1 = U[:, 0].reshape(-1, 1)

        return pca1

    except Exception:
        return np.zeros((len(ticker_df), 1))


# ---------------------------------------------------------
# Intraday Actionability (NO VWAP requirement)
# ---------------------------------------------------------
def check_intraday_actionability(current_price, previous_close, intraday_df):
    """
    Determine if a ticker is intraday actionable.
    VWAP is OPTIONAL — if missing, skip VWAP logic.
    """

    try:
        # Basic reclaim of previous close
        reclaim_prev_close = current_price > previous_close

        # Optional VWAP logic
        if 'VWAP' in intraday_df.columns:
            vwap_reclaim = current_price > intraday_df['VWAP'].iloc[-1]
        else:
            vwap_reclaim = False

        # Momentum check
        if len(intraday_df) > 10:
            momentum = intraday_df['Close'].iloc[-1] > intraday_df['Close'].iloc[-10]
        else:
            momentum = False

        actionable = reclaim_prev_close or vwap_reclaim or momentum

        return actionable

    except Exception:
        return False
