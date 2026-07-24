import logging
import warnings
import yfinance as yf

# Silence yfinance internal warnings/debug BEFORE Streamlit loads anything else
try:
    yf.utils.disable_warnings()
    yf.utils.disable_debug_mode()
except Exception:
    pass

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


import streamlit as st

st.set_page_config(page_title="Wishing Well", layout="wide")

st.title("📊 Wishing Well")
st.write("Choose a module from the left sidebar to begin.")