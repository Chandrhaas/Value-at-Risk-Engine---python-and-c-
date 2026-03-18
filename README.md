# Risk-Engine
I built a high-performance Monte Carlo Risk Engine. It is a quantitative finance tool that simulates thousands of potential future market paths to calculate critical portfolio risk metrics, specifically 95% Value at Risk (VaR) , 99% Value at Risk(VaR) and Expected Shortfall

# Technologies used
* c++
* python
    * Numpy
    * Pandas
    * yfinance
    * streamlit
    * pybind11

# Features
* Users can enter the stock tickers they want.
* Users can enter the number of simulations they want to run.
* Users can enter the size of the portfolio.

The Risk Engine applies Cholesky Decomposition to ensure simulated asset paths respect real-world market correlations. It also utilizes zero-copy memory transfers, allowing massive 2D matrices to pass from Python to C++ instantly without slowing down the system.

# The Process
I built this using a microservice-style architecture. I started by writing the data-fetching and statistical analysis modules in Python. Knowing Python is too slow for heavy Monte Carlo loops, I wrote the GBM (Geometric Brownian Motion) math natively in C++. To connect them, I engineered a Pybind11 bridge, allowing the Streamlit UI to hand off the heavy lifting to the compiled C++ binary, entirely bypassing Python's Global Interpreter Lock.

# Learnings
* Learned how to architect cross-language systems and safely manage memory handoffs between Python arrays and C++ vectors.
* Deepened my understanding of quantitative finance mathematics by implementing matrix transformations and random shock simulations entirely from scratch in C++, rather than relying on standard black-box libraries.
* pybind11 , i came to know about it during this project and i immediately shifted from ctypes to it because it's so clean as compared to ctypes

# How to run
* Clone the repository and install the Python dependencies from the requirements.txt file in a virtual environment.
* Run make clean and make in the terminal to compile the C++ engine for your specific machine.
* Run streamlit run app.py to launch the interactive web dashboard.

# Assumptions
## Mathematical 
* **Normally Distributed Returns:** GBM assumes that daily log returns follow a perfect Bell Curve (Normal Distribution).
* **Constant Volatility:** In our C++ loop, we calculated the volatility (variance) once and applied it evenly across all future paths.
* **Continuous Prices:** GBM assumes prices move smoothly. It cannot account for "gaps" in pricing, like a stock dropping 20% overnight due to a bad earnings report while the market was closed.

## Data 
* **Stationarity:** We fed the C++ engine the historical Mean Returns and Covariance Matrix from the last year, assuming those exact statistical properties will remain perfectly identical going forward.
* **252 days year**
* **Equally Weighted Portfolio**

## Market
* **Zero Transaction Costs:** We assumed we could hold this portfolio without paying any broker fees, bid-ask spreads, or taxes.
* **Perfect Liquidity:** We assumed that if our VaR is breached and we need to liquidate the portfolio, we can sell all our shares instantly at the exact market price without our own massive sell-off pushing the price down (slippage).

# Future Improvements
## Performance & Compute Optimizations
* **OpenMP Multi-threading:** Implementing #pragma omp parallel for in the C++ backend to distribute the Monte Carlo simulation paths across all available CPU cores simultaneously.

## Advanced Quantitative Models
* **Quasi-Random Numbers (Sobol Sequences):** Replacing standard pseudo-random number generators (PRNG) with Sobol sequences to ensure the simulated random shocks cover the distribution space more evenly, resulting in faster and more accurate mathematical convergence.
* **GARCH Volatility Modeling:** Upgrading from constant historical volatility to a dynamic GARCH(1,1) model to account for "volatility clustering"
* **Jump-Diffusion Models (Merton's Model):** Injecting random "jumps" into the Geometric Brownian Motion equation to simulate sudden market crashes or overnight gaps in pricing.
* **Fat-Tail Distributions:** Replacing the standard Normal Distribution with a Student's t-distribution to better model extreme, rare market events (Black Swans).

## System Architecture & Deployment
* **FastAPI Microservice:** Decoupling the Streamlit frontend from the Pybind11 bridge by wrapping the C++ engine in a high-speed Python REST API (FastAPI).
* **Docker Containerization:** Writing a Dockerfile to containerize the Linux environment, ensuring the C++ compiler, Pybind11, and Python dependencies run flawlessly on any cloud server without environment mismatch errors.


