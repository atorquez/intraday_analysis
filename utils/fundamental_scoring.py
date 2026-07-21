# ==============================================================================
# SHARED FUNDAMENTAL SCORING MODULE
# Use this in BOTH Page 2 (integrated scanner) and Page 4 (Ticker Summary)
# to guarantee identical scores across all pages.
# ==============================================================================

import yfinance as yf
import streamlit as st

# ---------------------------------------------------------
# SHARED SCORING FUNCTIONS (Extracted from Ticker_summary)
# ---------------------------------------------------------

def _score_valuation(forward_pe, trailing_pe):
    """Score valuation based on P/E ratios."""
    try:
        pe = forward_pe if forward_pe is not None and forward_pe != "N/A" else trailing_pe
        if pe is None or pe == "N/A":
            return 50
        pe = float(pe)
        if pe < 10: return 90
        elif pe < 20: return 80
        elif pe < 30: return 65
        elif pe < 40: return 50
        else: return 35
    except (TypeError, ValueError):
        return 50


def _score_growth(rev_growth, earn_growth):
    """Score growth based on revenue and earnings growth."""
    try:
        # Handle both decimal (0.087) and percentage string ("8.7%") inputs
        def parse_growth(val):
            if val is None or val == "N/A":
                return 0
            if isinstance(val, str):
                val = val.replace("%", "").strip()
            return float(val)

        rg = parse_growth(rev_growth)
        eg = parse_growth(earn_growth)

        # If values are decimals (yfinance returns 0.087 for 8.7%), convert
        if abs(rg) < 1 and rg != 0:
            rg = rg * 100
        if abs(eg) < 1 and eg != 0:
            eg = eg * 100

        g = (rg + eg) / 2

        if g > 30: return 95
        elif g > 20: return 85
        elif g > 10: return 70
        elif g > 5: return 55
        else: return 40
    except (TypeError, ValueError):
        return 50


def _score_profitability(margin):
    """Score profitability based on profit margin."""
    try:
        if margin is None or margin == "N/A":
            return 50
        if isinstance(margin, str):
            margin = margin.replace("%", "").strip()
        m = float(margin)

        # Convert decimal to percentage if needed
        if abs(m) < 1 and m != 0:
            m = m * 100

        if m > 20: return 95
        elif m > 10: return 85
        elif m > 0: return 70
        else: return 40
    except (TypeError, ValueError):
        return 50


def _score_risk(beta, dte):
    """Score risk based on beta and debt-to-equity."""
    try:
        b = float(beta) if beta is not None else 1.0
        d = float(dte) if dte is not None else 100

        beta_score = 90 if b < 0.8 else 75 if b < 1.0 else 60 if b < 1.2 else 45
        dte_score = 90 if d < 50 else 75 if d < 100 else 60 if d < 150 else 45 if d < 200 else 30

        return int((beta_score + dte_score) / 2)
    except (TypeError, ValueError):
        return 50


# ---------------------------------------------------------
# MAIN FETCH + SCORE FUNCTION
# ---------------------------------------------------------

def get_fundamental_scores(ticker, use_cache=True):
    """
    Fetch and score fundamentals for a single ticker.

    Returns dict with:
      - Fund_Score (overall 0-100)
      - Fund_Valuation, Fund_Growth, Fund_Profit, Fund_Risk (sub-scores)
      - Market_Cap, Sector, Industry
      - Raw data for debugging

    If use_cache=True and running in Streamlit, uses @st.cache_data.
    """
    ticker = str(ticker).strip().upper()

    try:
        info = yf.Ticker(ticker).info
        if not info or len(info) < 5:
            return None

        # Extract raw values
        forward_pe = info.get("forwardPE")
        trailing_pe = info.get("trailingPE")
        rev_growth = info.get("revenueGrowth")
        earn_growth = info.get("earningsGrowth")
        profit_margin = info.get("profitMargins")
        beta = info.get("beta")
        dte = info.get("debtToEquity")
        market_cap = info.get("marketCap")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")

        # Compute scores
        valuation = _score_valuation(forward_pe, trailing_pe)
        growth = _score_growth(rev_growth, earn_growth)
        profit = _score_profitability(profit_margin)
        risk = _score_risk(beta, dte)

        overall = int(valuation * 0.25 + growth * 0.30 + profit * 0.20 + risk * 0.25)

        return {
            "Fund_Score": overall,
            "Fund_Valuation": valuation,
            "Fund_Growth": growth,
            "Fund_Profit": profit,
            "Fund_Risk": risk,
            "Market_Cap": market_cap,
            "Sector": sector,
            "Industry": industry,
            # Raw values for debugging
            "_raw_forward_pe": forward_pe,
            "_raw_trailing_pe": trailing_pe,
            "_raw_rev_growth": rev_growth,
            "_raw_earn_growth": earn_growth,
            "_raw_profit_margin": profit_margin,
            "_raw_beta": beta,
            "_raw_dte": dte,
        }
    except Exception as e:
        return None


# ---------------------------------------------------------
# STREAMLIT CACHED VERSION
# ---------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamental_scores_cached(ticker):
    """Cached version for use in Streamlit pages."""
    return get_fundamental_scores(ticker, use_cache=False)


# ---------------------------------------------------------
# FORMATTER (from Ticker_summary)
# ---------------------------------------------------------

def format_financial_value(val, is_percentage=False):
    """Format large financial numbers for display."""
    if val is None:
        return "N/A"
    try:
        num = float(val)
        if is_percentage:
            if num < 1:
                return f"{round(num * 100, 2)}%"
            else:
                return f"{round(num, 2)}%"
        if num >= 1e12:
            return f"${round(num / 1e12, 2)} Trillion"
        elif num >= 1e9:
            return f"${round(num / 1e9, 2)} Billion"
        elif num >= 1e6:
            return f"${round(num / 1e6, 2)} Million"
        return f"${round(num, 2)}"
    except:
        return "N/A"