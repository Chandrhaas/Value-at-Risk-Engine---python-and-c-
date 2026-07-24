# Quantitative Risk Engine v2.0

A high-performance portfolio risk assessment system that runs **two VaR estimation methods in parallel** — a C++ Monte Carlo simulation core and a PyTorch Quantile Regression Neural Network — exposed via a single FastAPI REST service and a custom HTML/CSS/JS dashboard.

Computes **95% VaR, 99% VaR, and Expected Shortfall (CVaR)** across configurable multi-asset portfolios, with per-ticker breakdowns and direct empirical comparison between parametric and non-parametric approaches.

---

## Demo

<img width="1144" height="703" alt="Image" src="https://github.com/user-attachments/assets/0356af9e-45a0-4a0e-bfc2-fc4621525883" />

---

## Architecture

```
┌──────────────────────────┐      HTTP/JSON       ┌────────────────────────────┐
│   HTML/CSS/JS Frontend   │ POST /portfolio/var ▶ │   FastAPI REST Service     │
│      (frontend/)         │ ◀ CombinedRiskMetrics │   (api/main.py)            │
└──────────────────────────┘                       └────────────┬───────────────┘
                                                                │
                                              ┌─────────────────┴──────────────────┐
                                              │                                     │
                                   ┌──────────▼──────────┐           ┌─────────────▼──────────┐
                                   │  C++ Monte Carlo     │           │  PyTorch QRNN          │
                                   │  (src_cpp/)          │           │  (src_python/ml_var.py)│
                                   │  GBM · Cholesky      │           │  Pinball Loss · AR/Vol │
                                   │  pybind11 bridge     │           │  features · Two heads  │
                                   └─────────────────────┘           └────────────────────────┘
```

The frontend is fully decoupled — it knows nothing about either engine. It sends one JSON payload and receives estimates from both methods in a single response.

---

## Two VaR Methods — Why Both?

| | Monte Carlo (GBM) | Quantile Regression NN |
|---|---|---|
| **Assumption** | Returns are normally distributed | No distributional assumption |
| **Volatility** | Constant (historical) | Implicitly captured via rolling vol features |
| **Correlation** | Full Cholesky decomposition | Conservative sum (assumes perfect correlation) |
| **Training data** | Mean + covariance matrix | 5 years of daily returns per asset |
| **Strength** | Well-calibrated in calm markets | Captures fat tails and volatility regimes |
| **Weakness** | Underestimates tail risk in crises | Conservative correlation assumption |

The comparison is the point — not which method wins. Real risk desks run multiple models and understand when each is appropriate.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation Core | C++ (Monte Carlo, GBM, Cholesky) |
| Python–C++ Bridge | pybind11 (zero-copy memory transfer) |
| ML Model | PyTorch (Quantile Regression NN) |
| Feature Engineering | NumPy, Pandas (AR lags, rolling volatility) |
| REST API | FastAPI + Pydantic |
| Data Fetching | yfinance |
| Frontend | HTML, CSS, JavaScript |
| Containerisation | Docker |
| Build System | Make |

---

## Method 1 — C++ Monte Carlo Engine

### Step 1 — Data Ingestion
`fetch_data.py` pulls historical adjusted closing prices from Yahoo Finance for user-specified tickers over a configurable window (1–5 years).

### Step 2 — Parameter Estimation
`analysis.py` computes annualised log-return means and the covariance matrix:

```
μ_annual = mean(log(P_t / P_{t-1})) × 252
Σ_annual = cov(log returns) × 252
```

### Step 3 — Cholesky Decomposition
The C++ engine factorises the covariance matrix `Σ = LLᵀ` to generate **correlated** asset return paths. Applying `L` to a vector of independent standard normals produces shocks that respect real-world correlations between assets.

### Step 4 — Monte Carlo Simulation
For each of N simulations, the engine applies Geometric Brownian Motion:

```
S_T = S_0 × exp( (μ - ½σ²) + σZ )
```

where `Z ~ N(0,1)` is the Cholesky-correlated random shock.

### Step 5 — Risk Metric Extraction

```
95% VaR  = Initial Value − P(5th percentile of terminal values)
99% VaR  = Initial Value − P(1st percentile of terminal values)
CVaR     = Initial Value − mean(worst 5% of terminal values)
```

---

## Method 2 — Quantile Regression Neural Network

### Architecture
A feedforward neural network with a shared trunk and two output heads — one per quantile. The monotonicity constraint (99% VaR ≥ 95% VaR) is enforced structurally: the network outputs q05 directly, and q01 as `q05 − softplus(gap)`, guaranteeing q01 ≤ q05 for every input.

```
Input (8 features) → [Linear→ReLU] × 2 → q05 head → q05
                                         → gap head (softplus) → q01 = q05 − gap
```

### Features (per asset, per day)
- **AR lags:** log returns at t−1, t−2, t−3, t−5 — captures short-term momentum
- **Rolling volatility:** standard deviation over 5, 10, 21, 63-day windows — captures volatility regime

### Training
- **Loss function:** Pinball loss (Koenker & Bassett, 1978)
  ```
  loss = q × (y − ŷ)      if y > ŷ   (under-prediction)
  loss = (q−1) × (ŷ − y)  if y ≤ ŷ   (over-prediction)
  ```
  At q=0.05, the model is penalised 19× more for underestimating than overestimating — this asymmetry mathematically forces convergence to the 5th percentile without assuming any distribution.

- **Training universe:** 16 liquid, sector-diverse tickers (tech, financials, healthcare, consumer, energy, industrials)
- **Held-out tickers:** 6 tickers never seen during training — used to verify the model generalises across assets
- **Train/test split:** Strict temporal 80/20 split per ticker. No shuffling. No data leakage.
- **Single-asset design:** The model takes one ticker's feature vector at a time, making it asset-agnostic at inference — it can predict VaR for any ticker, including ones not in the training universe.

### Portfolio VaR Combination
Per-ticker VaR estimates are summed across the portfolio — the most conservative possible combination, assuming perfect positive correlation. This is deliberately conservative and documented as such.

---

## Key Finding — Model Risk

| Training Window | 95% VaR | 99% VaR | CVaR |
|---|---|---|---|
| 1 year (2024 bull market) | No loss expected | No loss expected | No loss expected |
| 4 years (includes 2022 bear market) | $1,817 loss | $3,044 loss | $2,551 loss |

**On a $10,000 portfolio of AAPL, MSFT, GOOGL.**

A 1-year training window trained on the 2024 bull market tells you there is no meaningful downside risk. A 4-year window including the 2022 bear market gives a completely different picture. This demonstrates the regime sensitivity of historical simulation — the model inherits the optimism of the period it trains on.

---

## API Reference

### `POST /portfolio/var`

**Request:**
```json
{
  "tickers": ["AAPL", "MSFT"],
  "portfolio_size": 10000.0,
  "num_simulations": 10000,
  "years": 4,
  "weights": [0.6, 0.4]
}
```

`weights` is optional — omit for equal weighting.

**Response:**
```json
{
  "monte_carlo": {
    "var_95": 1816.92,
    "var_99": 3043.54,
    "cvar": 2551.42,
    "message": "Simulation completed successfully via C++ engine."
  },
  "neural_network": {
    "var_95": 1923.41,
    "var_99": 3198.76,
    "per_ticker": {
      "AAPL": {
        "as_of_date": "2026-07-18",
        "q05_return": -0.034,
        "q01_return": -0.061,
        "var_95": 1153.82,
        "var_99": 1966.11
      }
    },
    "message": "Conservative estimate: assumes perfect positive correlation across tickers."
  },
  "comparison_note": "MC uses full Cholesky correlation; NN sums per-ticker VaR (worst-case correlation assumption)."
}
```

---

## How to Run

### Option 1 — Docker (recommended)
```bash
git clone https://github.com/Chandrhaas/Value-at-Risk-Engine---python-and-cpp.git
cd Value-at-Risk-Engine---python-and-cpp
docker compose up --build
```
Open `http://localhost:8000`. The model weights in `models/` are included — no training required.

### Option 2 — Local

**1. Clone and install:**
```bash
git clone https://github.com/Chandrhaas/Value-at-Risk-Engine---python-and-cpp.git
cd Value-at-Risk-Engine---python-and-cpp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Compile the C++ engine:**
```bash
make clean && make
```

**3. (Optional) Retrain the QRNN:**
```bash
python src_python/prepare_data.py   # build feature dataset
python src_python/train_var_model.py  # train and save to models/
```

**4. Start the API:**
```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000`.

---

## Project Structure

```
├── src_cpp/
│   ├── simulation.cpp        C++ Monte Carlo engine
│   └── simulation.h
├── src_python/
│   ├── fetch_data.py         yfinance data fetching
│   ├── analysis.py           mean returns + covariance matrix
│   ├── prepare_data.py       feature engineering pipeline
│   ├── train_var_model.py    QRNN definition + training loop
│   └── ml_var.py             QRNN inference module
├── api/
│   ├── main.py               FastAPI routes
│   └── model.py              Pydantic schemas
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── models/
│   ├── var_model.pth         Trained QRNN weights
│   └── feature_scaler.pkl    Fitted StandardScaler
├── data/                     Generated datasets (not tracked)
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## Mathematical Assumptions

### Monte Carlo
- Log-normal returns (GBM). Real markets have fat tails.
- Constant volatility. Real volatility clusters (GARCH effects not captured).
- No overnight gaps or jumps.
- Equal or user-specified weights; zero transaction costs; perfect liquidity.

### Neural Network
- Single-asset inference with conservative portfolio aggregation (sum of per-ticker VaR).
- Trained on 5 years of data — performance degrades in market regimes not seen during training.
- Features are stationary (log returns, rolling vol) but the model is not explicitly regime-aware.

---

## Roadmap
- **GARCH(1,1) volatility:** Dynamic volatility modelling for the Monte Carlo engine
- **Student's t-distribution:** Fat-tail distribution for GBM
- **OpenMP multi-threading:** Parallelise the C++ simulation loop across CPU cores
- **Sobol sequences:** Quasi-random number generation for faster Monte Carlo convergence
- **Jump-diffusion (Merton's model):** Poisson jump terms for overnight gap risk
- **Sentiment integration:** Feed NLP Alpha Engine signals as exogenous features into the QRNN

---

## License
MIT