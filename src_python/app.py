import streamlit as st
import requests

# Configure the page layout
st.set_page_config(page_title="Monte Carlo Risk Engine", layout="centered")

st.title("📈 Risk Engine v2.0")
st.write("Decoupled Microservice Architecture | C++ Pybind11 Backend")

st.sidebar.header("Portfolio Parameters")

# 1. Collect Inputs
portfolio_size = st.sidebar.number_input(
    "Total Portfolio Value ($)", 
    min_value=1000.0, 
    value=10000.0, 
    step=1000.0
)

years = st.sidebar.number_input(
    "Number of Years", 
    min_value=1, 
    value=1,
    max_value=5, 
    step=1
)

ticker_input = st.sidebar.text_input(
    "Tickers (comma-separated)", 
    value="AAPL, MSFT, GOOGL"
)

# Clean the string and convert it into a list of uppercase tickers
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

num_simulations = st.sidebar.slider(
    "Number of Simulations", 
    min_value=1000, 
    max_value=100000, 
    value=10000, 
    step=1000
)

# 2. The Execution Trigger
if st.button("Run Risk Analysis", type="primary"):
    if not tickers:
        st.warning("Please enter at least one valid ticker.")
    else:
        with st.spinner(f"Running {num_simulations:,} Monte Carlo paths via C++ Engine..."):
            
            # Package the exact JSON payload expected by your Pydantic Input Model
            payload = {
                "portfolio_size": portfolio_size,
                "years":years,
                "tickers": tickers,
                "num_simulations": num_simulations
            }
            
            try:
                # Shoot the payload to the local FastAPI server
                response = requests.post("http://127.0.0.1:8000/portfolio/var", json=payload)
                
                # Check if FastAPI rejected the data (e.g., 422 Validation Error)
                response.raise_for_status()
                
                # Parse the JSON response back into a Python dictionary
                metrics = response.json()
                
                # 3. Render the Results
                st.success(metrics.get("message", "Simulation completed successfully!"))
                
                st.subheader("Risk Metrics (252-Day Horizon)")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    var_95 = metrics['var_95']
                    if var_95 < 0:
                        st.metric("95% VaR", f"No loss expected", f"Min gain: ${abs(var_95):,.2f}")
                    else:
                        st.metric("95% VaR", f"${var_95:,.2f} max loss")
                with col2:
                    var_99 = metrics['var_99']
                    if var_99 < 0:
                        st.metric("99% VaR", f"No loss expected", f"Min gain: ${abs(var_99):,.2f}")
                    else:
                        st.metric("99% VaR", f"${var_99:,.2f} max loss")
                with col3:
                    st.metric(label="Expected Shortfall", value=f"${metrics['cvar']:,.2f}")
                    
            # 4. Error Handling
            except requests.exceptions.ConnectionError:
                st.error("🚨 API Connection Failed: Is your FastAPI server running on port 8000?")
            except requests.exceptions.HTTPError as e:
                try:
                    # Attempt to extract FastAPI's exact reason for rejection
                    error_detail = response.json().get("detail", str(e))
                    st.error(f"API Error: {error_detail}")
                except ValueError:
                    st.error(f"HTTP Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")