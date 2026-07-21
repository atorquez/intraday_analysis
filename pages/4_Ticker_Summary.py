import streamlit as st
import yfinance as yf

# Import shared scoring module
from utils.fundamental_scoring import (
    get_fundamental_scores,
    get_fundamental_scores_cached,
    format_financial_value
)

# ---------------------------------------------------------
# STREAMLIT PAGE 4 — FUNDAMENTALS PANEL
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📘 Corporate Fundamentals & Executive Summary")
st.caption("SPC Version: 2026-08-08 — Fundamentals Module")

# Check if navigated from Page 2 with a ticker
research_ticker = st.session_state.pop("research_ticker", None)

default_ticker = research_ticker if research_ticker else "MSFT"

ticker = st.text_input(
    "Enter a ticker symbol:", 
    value=default_ticker, 
    key="ticker_input_main"
).strip().upper()

if ticker:
    with st.spinner(f"Querying fundamentals for {ticker}..."):
        # Use the SAME scoring function as Page 2
        scores = get_fundamental_scores_cached(ticker)

        # Also fetch full info for display
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            info = None

    if scores is None or info is None:
        st.error(f"⚠️ Fundamentals not available for {ticker}. This is normal for small-cap or low-coverage companies.")

    else:
        # Build summary dict for display (same structure as before)
        summary = {
            "Company": info.get("longName", info.get("shortName", ticker)),
            "Sector": scores.get("Sector", "N/A"),
            "Industry": scores.get("Industry", "N/A"),
            "Country": info.get("country", "N/A"),
            "Employees": info.get("fullTimeEmployees", "N/A"),
            "Market Cap": format_financial_value(scores.get("Market_Cap")),
            "Beta": round(scores.get("_raw_beta", 0.0), 2) if scores.get("_raw_beta") is not None else "N/A",
            "Forward PE": round(scores.get("_raw_forward_pe", 0.0), 2) if scores.get("_raw_forward_pe") is not None else "N/A",
            "Trailing PE": round(scores.get("_raw_trailing_pe", 0.0), 2) if scores.get("_raw_trailing_pe") is not None else "N/A",
            "Profit Margin": format_financial_value(scores.get("_raw_profit_margin"), is_percentage=True),
            "Revenue Growth": format_financial_value(scores.get("_raw_rev_growth"), is_percentage=True),
            "Earnings Growth": format_financial_value(scores.get("_raw_earn_growth"), is_percentage=True),
            "Dividend Yield": format_financial_value(info.get("dividendYield"), is_percentage=True),
            "Debt-to-Equity": round(scores.get("_raw_dte", 0.0), 2) if scores.get("_raw_dte") is not None else "N/A",
            "Shares Outstanding": format_financial_value(info.get("sharesOutstanding")).replace("$", ""),
        }

        # Use scores from shared module (GUARANTEED same as Page 2)
        valuation_score = scores["Fund_Valuation"]
        growth_score = scores["Fund_Growth"]
        profit_score = scores["Fund_Profit"]
        risk_score = scores["Fund_Risk"]
        overall_score = scores["Fund_Score"]

        # ---------------------------------------------------------
        # EXECUTIVE SUMMARY
        # ---------------------------------------------------------
        st.markdown("### 🧠 Executive Summary")

        executive_text = (
            f"**{summary['Company']}** operates in the **{summary['Sector']}** sector, "
            f"specializing in the **{summary['Industry'].lower()}** segment. "
            f"The firm maintains a market capitalization of **{summary['Market Cap']}** and "
            f"shows a Beta of **{summary['Beta']}**, defining its volatility baseline "
            f"relative to the broad market."
        )
        st.info(executive_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏛️ Corporate Profile")
            st.write(f"**Sector:** {summary['Sector']}")
            st.write(f"**Industry:** {summary['Industry']}")
            st.write(f"**Country:** {summary['Country']}")
            st.write(f"**Employees:** {summary['Employees']}")
            st.write(f"**Market Cap:** {summary['Market Cap']}")
            st.write(f"**Beta:** {summary['Beta']}")
            st.write(f"**Shares Outstanding:** {summary['Shares Outstanding']}")

        with col2:
            st.markdown("#### 📊 Valuation & Growth Metrics")
            st.write(f"**Trailing P/E:** {summary['Trailing PE']}")
            st.write(f"**Forward P/E:** {summary['Forward PE']}")
            st.write(f"**Profit Margin:** {summary['Profit Margin']}")
            st.write(f"**Revenue Growth:** {summary['Revenue Growth']}")
            st.write(f"**Earnings Growth:** {summary['Earnings Growth']}")
            st.write(f"**Dividend Yield:** {summary['Dividend Yield']}")
            st.write(f"**Debt-to-Equity:** {summary['Debt-to-Equity']}")

        # ---------------------------------------------------------
        # FUNDAMENTAL SCORING SYSTEM (0–100)
        # ---------------------------------------------------------
        st.markdown("### 📊 Fundamental Scores (0–100)")

        # Color code the overall score
        if overall_score >= 80:
            score_color = "🟢"
        elif overall_score >= 60:
            score_color = "🟡"
        else:
            score_color = "🔴"

        st.success(
            f"""
**{score_color} Overall Fundamental Score:** {overall_score}

**Valuation Score:** {valuation_score}  
**Growth Score:** {growth_score}  
**Profitability Score:** {profit_score}  
**Risk Score:** {risk_score}
"""
        )

        # Debug expander to show raw values (helps verify consistency)
        with st.expander("🔧 Debug: Raw Values & Calculation"):
            st.write("**Raw values from yfinance:**")
            st.json({
                "forwardPE": scores.get("_raw_forward_pe"),
                "trailingPE": scores.get("_raw_trailing_pe"),
                "revenueGrowth": scores.get("_raw_rev_growth"),
                "earningsGrowth": scores.get("_raw_earn_growth"),
                "profitMargins": scores.get("_raw_profit_margin"),
                "beta": scores.get("_raw_beta"),
                "debtToEquity": scores.get("_raw_dte"),
            })
            st.write("**Score calculation:**")
            st.write(f"Valuation: {valuation_score} × 0.25 = {valuation_score * 0.25}")
            st.write(f"Growth: {growth_score} × 0.30 = {growth_score * 0.30}")
            st.write(f"Profit: {profit_score} × 0.20 = {profit_score * 0.20}")
            st.write(f"Risk: {risk_score} × 0.25 = {risk_score * 0.25}")
            st.write(f"**Total:** {valuation_score * 0.25 + growth_score * 0.30 + profit_score * 0.20 + risk_score * 0.25}")

        # ---------------------------------------------------------
        # INTERPRETATION GUIDE
        # ---------------------------------------------------------
        st.markdown("### 📘 Quick Interpretation Guide")

        st.info(
            """
**Beta — Volatility vs Market**
- **< 1.0** → Less volatile than the market (stable)
- **1.0** → Moves with the market
- **> 1.0** → More volatile (riskier, faster)
**Rule of Thumb:** *Beta < 1 = stable; Beta > 1 = fast mover.*

**Debt-to-Equity — Financial Leverage**
- **0–50** → Low leverage (safe)
- **50–100** → Moderate leverage
- **100–200** → High leverage
- **200+** → Very high leverage (fragile)
**Rule of Thumb:** *D/E > 150 = leveraged; requires strong cash flow.*

**Profit Margin — Profitability**
- **> 10%** → Strong
- **0–10%** → Thin but acceptable
- **< 0%** → Losing money
**Rule of Thumb:** *Negative margin = growth‑dependent company.*

**Revenue Growth — Business Momentum**
- **> 20%** → High growth
- **10–20%** → Strong
- **5–10%** → Moderate
- **< 5%** → Slow
**Rule of Thumb:** *Growth < 10% = stable; > 20% = high‑growth.*

**Forward P/E — Valuation**
- **10–20** → Reasonable / value
- **20–40** → Growth pricing
- **40+** → Expensive / high‑growth premium
**Rule of Thumb:** *Forward P/E < 20 = fairly priced; > 40 = premium.*
"""
        )