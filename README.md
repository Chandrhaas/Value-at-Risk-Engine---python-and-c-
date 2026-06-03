# Quantitative Risk Engine v2.0

A high-performance portfolio risk assessment system built on a **decoupled microservice architecture** — a C++ Monte Carlo simulation core exposed via a FastAPI REST service, bridged through zero-copy pybind11 memory transfers.

Computes **95% VaR, 99% VaR, and Expected Shortfall (CVaR)** across configurable multi-asset portfolios, with support for dynamic ticker selection and historical training windows of 1–4 years.

![demo](https://github.com/user-attachments/assets/135048ca-9dc0-49ef-893c-63425492214a)

---

## Architecture

```
┌─────────────────────┐        HTTP/JSON        ┌──────────────────────────┐
│   Streamlit Frontend │ ──── POST /portfolio/var ──▶  FastAPI REST Service  │
│     (src_python/     │ ◀──── RiskMetricsOutput ────  (api/main.py)         │
│       app.py)        │                         └────────────┬─────────────┘
└─────────────────────┘                                       │
                                                    Pydantic validation
                                                    yfinance data fetch
                                                    Cholesky decomposition
                                                              │
                                                   zero-copy pybind11 bridge
                                                              │
                                                   ┌──────────▼─────────────┐
                                                   │   C++ Monte Carlo Core  │
                                                   │   (src_cpp/simulation)  │
                                                   │   GBM · Cholesky · VaR  │
                                                   └────────────────────────┘
```

The Streamlit frontend is fully decoupled — it knows nothing about the C++ engine. It sends a JSON payload to the FastAPI service and renders the response. Any other client (curl, Postman, another service) can consume the same API.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation Core | C++ (Monte Carlo, GBM, Cholesky) |
| Python–C++ Bridge | pybind11 (zero-copy memory transfer) |
| REST API | FastAPI + Pydantic |
| Data Fetching | yfinance |
| Statistical Analysis | NumPy, Pandas |
| Frontend | Streamlit |
| Build System | Make |

---

## How It Works

### Step 1 — Data Ingestion
`fetch_data.py` pulls historical adjusted closing prices from Yahoo Finance for user-specified tickers over a configurable window (1–4 years). Missing trading days are handled via forward-fill then backward-fill.

### Step 2 — Parameter Estimation
`analysis.py` computes annualised log-return means and the covariance matrix from historical data:

```
μ_annual = mean(log(P_t / P_{t-1})) × 252
Σ_annual = cov(log returns) × 252
```

### Step 3 — Cholesky Decomposition
The C++ engine factorises the covariance matrix `Σ = L Lᵀ` to generate **correlated** asset return paths. Applying `L` to a vector of independent standard normals produces shocks that respect the real-world correlation structure between assets.

### Step 4 — Monte Carlo Simulation (C++ Core)
For each of N simulations, the engine generates correlated random shocks and applies the Geometric Brownian Motion formula to project the portfolio value over 252 trading days:

```
S_T = S_0 × exp( (μ - ½σ²) + σZ )
```

where `Z ~ N(0,1)` is the Cholesky-correlated random shock. The result is a distribution of N terminal portfolio values.

### Step 5 — Risk Metric Extraction
Sorting the simulated terminal values in ascending order:

```
95% VaR  = Initial Value − P(5th percentile of terminal values)
99% VaR  = Initial Value − P(1st percentile of terminal values)
CVaR     = Initial Value − mean(worst 5% of terminal values)
```

A positive VaR means expected loss. A negative VaR means the portfolio is projected to gain even in adverse scenarios — a direct signal of model risk from over-optimistic training data.

---

## Key Finding — Model Risk

One of the most instructive outputs of this engine is what happens when you change the training window:

| Training Window | 95% VaR | 99% VaR | CVaR |
|---|---|---|---|
| 1 year (2024 bull market) | No loss expected | No loss expected | No loss expected |
| 4 years (includes 2022 bear market) | $1,817 loss | $3,044 loss | $2,551 loss |

**On a $10,000 portfolio of AAPL, MSFT, GOOGL.**

A 1-year window trained on the 2024 bull market tells you there is no meaningful downside risk. A 4-year window that includes the 2022 bear market gives you a completely different picture. This is a concrete demonstration of why short-window historical simulation is dangerous in practice — the model inherits the optimism of the period you train on.

---

## API Reference

### `POST /portfolio/var`
Runs the full simulation pipeline and returns risk metrics.

**Request body:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "portfolio_size": 10000.0,
  "num_simulations": 10000,
  "years": 4
}
```

**Constraints:** `portfolio_size` in (0, 1,000,000) · `num_simulations` in (999, 100,001) · `years` in (0, 5)

**Response:**
```json
{
  "var_95": 1816.92,
  "var_99": 3043.54,
  "cvar": 2551.42,
  "message": "Simulation completed successfully via C++ engine."
}
```

Positive values represent losses. Negative values indicate the portfolio gains even under the given adverse scenario.

---

## How to Run

### 1. Clone and install dependencies
```bash
git clone https://github.com/Chandrhaas/Value-at-Risk-Engine---python-and-cpp.git
cd Value-at-Risk-Engine---python-and-cpp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Compile the C++ engine
```bash
make clean && make
```
This compiles `src_cpp/simulation.cpp` and builds the `riskengine` pybind11 module into `build/`.

### 3. Start the FastAPI server
```bash
uvicorn api.main:app --reload
```
API will be live at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### 4. Launch the Streamlit frontend
In a separate terminal:
```bash
streamlit run src_python/app.py
```

---

## Mathematical Assumptions

### Model
- **Log-normal returns:** GBM assumes log returns are normally distributed. Real markets exhibit fat tails — extreme events occur more frequently than the normal distribution predicts.
- **Constant volatility:** Volatility is estimated once from historical data and held fixed across all simulation paths. Real volatility clusters (GARCH effects) are not captured.
- **Continuous prices:** GBM cannot model overnight gaps caused by earnings announcements or macro shocks.
- **Single-period projection:** The engine projects the full 252-day horizon in one step, not as 252 daily steps. This is equivalent mathematically for terminal value but does not capture path-dependent risk.

### Data
- **Stationarity:** Historical mean returns and covariance are assumed to be stable forward estimates. This assumption breaks down across market regime changes.
- **252 trading days per year**
- **Equal portfolio weights**

### Market
- **Zero transaction costs**
- **Perfect liquidity:** Full liquidation at market price is assumed, ignoring slippage from large position unwinds.

---

## Roadmap

### In Progress
- **ML-based VaR:** Replacing the parametric GBM model with a quantile regression neural network trained directly on historical return distributions — no distributional assumptions required.
- **Docker containerisation:** Single `docker-compose up` to run the full stack.

### Planned
- **GARCH(1,1) volatility:** Dynamic volatility that accounts for volatility clustering and mean-reversion.
- **Student's t-distribution:** Replacing the normal distribution to model fat-tailed return behaviour.
- **OpenMP multi-threading:** Parallelising the Monte Carlo loop across CPU cores.
- **Sobol sequences:** Quasi-random number generation for faster convergence.
- **Jump-diffusion (Merton's model):** Adding Poisson-distributed jump terms to capture overnight gap risk.

---

## License
MIT