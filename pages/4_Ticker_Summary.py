import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# PRO-GRADE UTILITY: LARGE FINANCIAL NUMBER FORMATTER
# ---------------------------------------------------------
def format_financial_value(val, is_percentage=False):
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

# ---------------------------------------------------------
# FUNDAMENTAL DATA FETCH ENGINE (YFINANCE.INFO)
# ---------------------------------------------------------
def get_executive_summary(ticker):
    try:
        t = yf.Ticker(str(ticker).strip().upper())
        info = t.info
    except Exception as e:
        return None, f"Network Extraction Failure: {str(e)}"

    if not info or len(info) < 5:
        return None, "⚠️ Fundamentals not available for this ticker. This is normal for small-cap or low-coverage companies."

    summary = {
        "Company": info.get("longName", info.get("shortName", ticker)),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Country": info.get("country", "N/A"),
        "Employees": info.get("fullTimeEmployees", "N/A"),

        "Market Cap": format_financial_value(info.get("marketCap")),
        "Beta": round(info.get("beta", 0.0), 2) if info.get("beta") is not None else "N/A",

        "Forward PE": round(info.get("forwardPE", 0.0), 2) if info.get("forwardPE") is not None else "N/A",
        "Trailing PE": round(info.get("trailingPE", 0.0), 2) if info.get("trailingPE") is not None else "N/A",

        "Profit Margin": format_financial_value(info.get("profitMargins"), is_percentage=True),
        "Revenue Growth": format_financial_value(info.get("revenueGrowth"), is_percentage=True),
        "Earnings Growth": format_financial_value(info.get("earningsGrowth"), is_percentage=True),

        "Dividend Yield": format_financial_value(info.get("dividendYield"), is_percentage=True),
        "Debt-to-Equity": round(info.get("debtToEquity", 0.0), 2) if info.get("debtToEquity") is not None else "N/A",

        "52 Week High": round(info.get("fiftyTwoWeekHigh", 0.0), 2) if info.get("fiftyTwoWeekHigh") is not None else "N/A",
        "52 Week Low": round(info.get("fiftyTwoWeekLow", 0.0), 2) if info.get("fiftyTwoWeekLow") is not None else "N/A",
        "Shares Outstanding": format_financial_value(info.get("sharesOutstanding")).replace("$", "")
    }

    executive_text = (
        f"**{summary['Company']}** operates in the **{summary['Sector']}** sector, "
        f"specializing in the **{summary['Industry'].lower()}** segment. "
        f"The firm maintains a market capitalization of **{summary['Market Cap']}** and "
        f"shows a Beta of **{summary['Beta']}**, defining its volatility baseline "
        f"relative to the broad market."
    )

    return summary, executive_text

# ---------------------------------------------------------
# STREAMLIT PAGE 4 — FUNDAMENTALS PANEL
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📘 Corporate Fundamentals & Executive Summary")
st.caption("SPC Version: 2026-08-08 — Fundamentals Module")

ticker = st.text_input("Enter a ticker symbol:", value="MSFT").strip().upper()

if ticker:
    with st.spinner(f"Querying fundamentals for {ticker}..."):
        summary, narrative = get_executive_summary(ticker)

    if summary is None:
        st.error(narrative)

    else:
        # ---------------------------------------------------------
        # EXECUTIVE SUMMARY
        # ---------------------------------------------------------
        st.markdown("### 🧠 Executive Summary")
        st.info(narrative)

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

        def score_valuation(forward_pe, trailing_pe):
            try:
                pe = forward_pe if forward_pe != "N/A" else trailing_pe
                if pe == "N/A":
                    return 50
                if pe < 10:
                    return 90
                elif pe < 20:
                    return 80
                elif pe < 30:
                    return 65
                elif pe < 40:
                    return 50
                else:
                    return 35
            except:
                return 50

        def score_growth(rev_growth, earn_growth):
            try:
                rg = float(rev_growth.replace("%","")) if rev_growth != "N/A" else 0
                eg = float(earn_growth.replace("%","")) if earn_growth != "N/A" else 0
                g = (rg + eg) / 2
                if g > 30:
                    return 95
                elif g > 20:
                    return 85
                elif g > 10:
                    return 70
                elif g > 5:
                    return 55
                else:
                    return 40
            except:
                return 50

        def score_profitability(margin):
            try:
                m = float(margin.replace("%",""))
                if m > 20:
                    return 95
                elif m > 10:
                    return 85
                elif m > 0:
                    return 70
                else:
                    return 40
            except:
                return 50

        def score_risk(beta, dte):
            try:
                b = float(beta)
                d = float(dte)

                beta_score = 90 if b < 0.8 else 75 if b < 1.0 else 60 if b < 1.2 else 45

                if d < 50:
                    dte_score = 90
                elif d < 100:
                    dte_score = 75
                elif d < 150:
                    dte_score = 60
                elif d < 200:
                    dte_score = 45
                else:
                    dte_score = 30

                return int((beta_score + dte_score) / 2)
            except:
                return 50

        valuation_score = score_valuation(summary["Forward PE"], summary["Trailing PE"])
        growth_score = score_growth(summary["Revenue Growth"], summary["Earnings Growth"])
        profit_score = score_profitability(summary["Profit Margin"])
        risk_score = score_risk(summary["Beta"], summary["Debt-to-Equity"])

        overall_score = int(
            valuation_score * 0.25 +
            growth_score * 0.30 +
            profit_score * 0.20 +
            risk_score * 0.25
        )

        st.markdown("### 📊 Fundamental Scores (0–100)")
        st.success(
            f"""
**Overall Fundamental Score:** {overall_score}

**Valuation Score:** {valuation_score}  
**Growth Score:** {growth_score}  
**Profitability Score:** {profit_score}  
**Risk Score:** {risk_score}
"""
        )

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
