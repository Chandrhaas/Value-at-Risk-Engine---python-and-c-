import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import datetime

# Get the absolute path to the src_python folder and the main project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Force Python's working directory to the project root so "data/..." saves correctly
os.chdir(project_root)

# Tell Python exactly where to find the C++ engine and your other Python scripts
sys.path.insert(0, os.path.join(project_root, 'build'))
sys.path.insert(0, current_dir)

#  Import Your Custom Modules
try:
    from fetch_data import fetch_data
    from analysis import calculate_parameters
except ImportError as e:
    st.error(f"🚨 Critical Error: Could not import your Python scripts. Details: {e}")
    st.stop()

# Connect the Pybind11 C++ Engine 
try:
    import riskengine
except ImportError as e:
    st.error(f"🚨 Critical Error: Could not find 'riskengine'. Did you compile it? Details: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Risk Engine", page_icon="📈", layout="wide")
st.title("Monte Carlo Risk Engine (Full Pipeline)")
st.markdown("Fetching -> Analysis -> C++ Simulation -> UI")

# --- UI Sidebar: User Inputs ---
st.sidebar.header("Portfolio Configuration")
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1000.0, value=10000.0, step=1000.0)
num_simulations = st.sidebar.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

default_tickers = "AAPL, MSFT, GOOGL, TSLA"
ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated)", value=default_tickers)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# --- Main Execution Pipeline ---
if len(tickers) < 2:
    st.warning("Please enter at least 2 valid tickers to calculate covariance.")
else:
    if st.button("Run Full Pipeline", type="primary"):
        
        with st.status("Executing Pipeline...", expanded=True) as status:
            try:
                # --- STEP 1: Execute fetch_data.py ---
                st.write("Fetching data...")
                end_date = datetime.date.today()
                start_date = end_date - datetime.timedelta(days=365)
                
                fetch_data(tickers, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
                
                # --- STEP 2: Execute analysis.py ---
                st.write("Analyzing data...")
                calculate_parameters()
                
                # --- STEP 3: Load the Processed Data ---
                st.write("Loading parameters for C++ Engine...")
                if not os.path.exists("data/means.csv") or not os.path.exists("data/covariance.csv"):
                    raise FileNotFoundError("Pipeline failed: CSV files were not generated in the 'data' folder.")

                means_df = pd.read_csv("data/means.csv", index_col=0)
                cov_df = pd.read_csv("data/covariance.csv", index_col=0)
                
                num_assets = len(tickers)
                weights = [1.0 / num_assets] * num_assets
                mean_returns_list = means_df.iloc[:, 0].tolist()
                cov_matrix_1d_list = cov_df.values.flatten().tolist()
                
                # --- STEP 4: Execute C++ Engine ---
                st.write("Executing C++ Engine...")
                engine = riskengine.MonteCarloEngine(num_assets, num_simulations)
                result = engine.runSimulation(initial_investment, weights, mean_returns_list, cov_matrix_1d_list)
                
                status.update(label="Pipeline Complete!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="Pipeline Failed!", state="error", expanded=True)
                st.error(f"An error occurred: {e}")
                st.stop()

        # --- STEP 5: Display Results ---
        col1, col2 ,col3= st.columns(3)
        with col1:
            st.metric(label="95% Value at Risk (VaR)", value=f"${result.var95:,.2f}")
        with col2:
            st.metric(label="99% Value at Risk (VaR)", value=f"${result.var99:,.2f}")
        with col3:
            st.metric(label="Expected Shortfall", value=f"${result.mean_loss:,.2f}")