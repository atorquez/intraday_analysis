import streamlit as st

if "trade_log" not in st.session_state:
    st.session_state["trade_log"] = []


st.set_page_config(layout="wide")
st.title("📘 Ticker Log — Trade Planner")

# Initialize log
if "trade_log" not in st.session_state:
    st.session_state["trade_log"] = []

st.markdown("### ➕ Add New Trade")

ticker = st.text_input("Ticker (e.g., AAPL)").upper()
buy_price = st.number_input("Buy Price", value=0.0, format="%.6f")
shares = st.number_input("Number of Shares", value=1, step=1)
expected_increase = st.number_input("Expected Price Increase (%)", value=2.0) / 100.0
rr_ratio = st.number_input("Risk/Reward Ratio (RR)", value=0.33)

if st.button("Add Trade"):
    if ticker and buy_price > 0:
        sell_price = buy_price * (1 + expected_increase)
        risk = expected_increase * rr_ratio
        exit_price = buy_price * (1 - risk)

        st.session_state["trade_log"].append({
            "ticker": ticker,
            "buy_price": buy_price,
            "shares": shares,
            "expected_increase": expected_increase,
            "rr": rr_ratio,
            "sell_price": sell_price,
            "exit_price": exit_price
        })

        st.success(f"Added {ticker} to trade log.")
    else:
        st.error("Ticker and Buy Price are required.")

st.markdown("---")
st.subheader("📊 Logged Trades")

if len(st.session_state["trade_log"]) == 0:
    st.info("No trades logged yet.")
else:
    for i, trade in enumerate(st.session_state["trade_log"]):
        st.markdown(f"### {trade['ticker']} — Scenario {i+1}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Buy Price:** {trade['buy_price']:.4f}")
            st.write(f"**Shares:** {trade['shares']}")
            st.write(f"**Expected Increase:** {trade['expected_increase']*100:.2f}%")

        with col2:
            st.write(f"**RR Ratio:** {trade['rr']:.2f}")
            st.write(f"**Sell Price:** {trade['sell_price']:.4f}")
            st.write(f"**Exit Price:** {trade['exit_price']:.4f}")

        with col3:
            pnl_target = (trade["sell_price"] - trade["buy_price"]) * trade["shares"]
            pnl_risk = (trade["buy_price"] - trade["exit_price"]) * trade["shares"]

            st.write(f"**Profit Target:** ${pnl_target:.2f}")
            st.write(f"**Max Risk:** ${pnl_risk:.2f}")

            if st.button(f"Remove Entry {i+1}", key=f"remove_{i}"):
                st.session_state["trade_log"].pop(i)
                st.experimental_rerun()

        st.markdown("---")
