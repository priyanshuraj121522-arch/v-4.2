
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from dcf import run_dcf
from upload_parsers import parse_uploaded_file

# Try safe import of data_fetchers
try:
    import data_fetchers as df
except Exception as e:
    st.error(f"Couldn't import data_fetchers: {e}")
    df = None

fetch_from_yfinance = getattr(df, "fetch_from_yfinance", None) if df else None
infer_net_debt_yf = getattr(df, "infer_net_debt_yf", None) if df else None
infer_shares_yf = getattr(df, "infer_shares_yf", None) if df else None
get_current_price_yf = getattr(df, "get_current_price_yf", None) if df else None
auto_wacc_best_effort = getattr(df, "auto_wacc_best_effort", None) if df else None

st.set_page_config(page_title="Indian Stocks DCF Fair Value", layout="wide")

st.title("📈 Indian Stocks Fair Value Calculator (DCF Method)")

st.sidebar.header("Input Options")

data_mode = st.sidebar.radio(
    "Choose Financial Data Source",
    ["Auto (Yahoo Finance)", "Manual Entry", "Upload File"]
)

wacc_mode = st.sidebar.radio(
    "Discount Rate (WACC)",
    ["Manual", "Semi-auto (CAPM)", "Full Auto"]
)

# Inputs
ticker = st.sidebar.text_input("Stock Ticker (e.g. RELIANCE.NS, TCS.NS)", "")

fcff = None
net_debt = None
shares = None
current_price = None

if data_mode == "Auto (Yahoo Finance)":
    if ticker and fetch_from_yfinance:
        try:
            fin = fetch_from_yfinance(ticker)
            fcff = fin.get("fcff")
            net_debt = infer_net_debt_yf(fin)
            shares = infer_shares_yf(fin)
            current_price = get_current_price_yf(ticker)
            st.success("Fetched financials from Yahoo Finance.")
        except Exception as e:
            st.error(f"Failed to fetch from Yahoo Finance: {e}")
    else:
        st.info("Enter a ticker to fetch financials.")
elif data_mode == "Manual Entry":
    fcff = st.sidebar.number_input("FCFF (₹ Cr)", value=0.0)
    net_debt = st.sidebar.number_input("Net Debt (₹ Cr)", value=0.0)
    shares = st.sidebar.number_input("Shares Outstanding (Crore)", value=0.0)
    current_price = st.sidebar.number_input("Current Market Price (₹)", value=0.0)
elif data_mode == "Upload File":
    uploaded = st.sidebar.file_uploader("Upload CSV/XLSX")
    if uploaded:
        try:
            fin = parse_uploaded_file(uploaded)
            fcff = fin.get("fcff")
            net_debt = fin.get("net_debt")
            shares = fin.get("shares")
            current_price = fin.get("current_price")
            st.success("Parsed uploaded financial file.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# WACC inputs
wacc = None
if wacc_mode == "Manual":
    wacc = st.sidebar.number_input("WACC (%)", value=10.0) / 100
elif wacc_mode == "Semi-auto (CAPM)":
    rf = st.sidebar.number_input("Risk-free rate (%)", value=7.0) / 100
    beta = st.sidebar.number_input("Beta", value=1.0)
    mrp = st.sidebar.number_input("Market Risk Premium (%)", value=6.0) / 100
    kd = st.sidebar.number_input("Cost of Debt (%)", value=8.5) / 100
    tax = st.sidebar.number_input("Tax Rate (%)", value=25.0) / 100
    e_weight = st.sidebar.slider("Equity Weight", 0.0, 1.0, 0.8)
    d_weight = 1 - e_weight
    ke = rf + beta * mrp
    wacc = e_weight * ke + d_weight * kd * (1 - tax)
elif wacc_mode == "Full Auto":
    if ticker and auto_wacc_best_effort:
        try:
            auto = auto_wacc_best_effort(ticker, fin if 'fin' in locals() else None)
            wacc = auto.get("wacc")
            with st.expander("Auto WACC Breakdown"):
                st.json(auto)
        except Exception as e:
            st.error(f"Auto WACC failed: {e}")
    else:
        st.info("Enter a ticker to run Full Auto WACC.")

run = st.button("Run Valuation")

if run:
    if fcff is None or net_debt is None or shares is None or wacc is None:
        st.error("Missing inputs. Please fill all required values.")
    else:
        try:
            result, df_proj = run_dcf(
                fcff=fcff,
                net_debt=net_debt,
                shares=shares,
                wacc=wacc
            )
            st.subheader("Results")
            st.write(result)
            st.subheader("Projection Table")
            st.dataframe(df_proj)
        except Exception as e:
            st.error(f"DCF calculation failed: {e}")
