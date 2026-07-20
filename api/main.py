import os, sys
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from api.model import (
    PortfolioInput,
    RiskMetricsOutput,
    NeuralNetResult,
    NeuralNetTickerBreakdown,
    CombinedRiskMetricsOutput,
)
from src_python.fetch_data import fetch_data
from src_python.analysis import calculate_parameters
from src_python import ml_var
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')))
import riskengine

app = FastAPI()

# Load the NN model + scaler ONCE at startup, not per request, reused
# run train_var_model.py first.
NN_MODEL, NN_SCALER, NN_FEATURE_COLUMNS, NN_QUANTILES = ml_var.load_model_and_scaler()


@app.post("/portfolio/var", response_model=CombinedRiskMetricsOutput)
def calculate_var(req: PortfolioInput):
    try:
        print(f"Processing portfolio of size: {req.portfolio_size}")
        print(f"Analyzing tickers: {req.tickers}")

        num_assets = len(req.tickers)

        if req.weights is not None:
            if len(req.weights) != num_assets:
                raise HTTPException(status_code=400, detail=f"weights length ({len(req.weights)}) must match tickers length ({num_assets})")
            weight_sum = sum(req.weights)
            if not (0.99 <= weight_sum <= 1.01):
                raise HTTPException(status_code=400, detail=f"weights must sum to 1.0 (got {weight_sum:.4f})")
            weights = req.weights
        else:
            weights = [1.0 / num_assets] * num_assets

        # Monte Carlo 
        data = fetch_data(req.tickers, req.years)
        mean_returns, cov_matrix = calculate_parameters(data)
        mean_flat = mean_returns.tolist()
        cov_flat = cov_matrix.flatten().tolist()

        engine = riskengine.MonteCarloEngine(num_assets, req.num_simulations)

        cpp_result = engine.runSimulation(
            req.portfolio_size, 
            weights, 
            mean_flat, 
            cov_flat
        )
        mc_result = RiskMetricsOutput(
            var_95=cpp_result.var95,
            var_99=cpp_result.var99,
            cvar=cpp_result.mean_loss,
            message="Simulation completed successfully via C++ engine."
        )

        # Neural network 
        # Deliberately NOT passing req.years through here: the NN only
        # needs ~1 year of history for its rolling-feature warm-up
        nn_raw = ml_var.predict_portfolio_var(
            tickers=req.tickers,
            weights=weights,
            portfolio_value=req.portfolio_size,
            model=NN_MODEL,
            scaler=NN_SCALER,
            feature_columns=NN_FEATURE_COLUMNS,
        )
        nn_message = (
            "NN quantile-regression estimate. Portfolio total assumes worst-case "
            "(perfectly correlated) combination across tickers, it does NOT "
            "model cross-asset correlation the way the Monte Carlo engine's "
            "covariance matrix does."
        )
        nn_result = NeuralNetResult(
            var_95=nn_raw["var_95"],
            var_99=nn_raw["var_99"],
            per_ticker={
                ticker: NeuralNetTickerBreakdown(**info)
                for ticker, info in nn_raw["per_ticker"].items()
            },
            message=nn_message,
        )

        comparison_note = (
            "Monte Carlo models cross-asset correlation via a covariance matrix; "
            "the neural network does not (see neural_network.message). A gap "
            "between the two, especially a lower Monte Carlo estimate, mainly "
            "reflects diversification credit the NN's conservative summed estimate doesn't give."
        )

        return CombinedRiskMetricsOutput(
            monte_carlo=mc_result,
            neural_network=nn_result,
            comparison_note=comparison_note,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mounted LAST, after /portfolio/var 
app.mount(
    "/",
    StaticFiles(
        directory=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")),
        html=True,
    ),
    name="frontend",
)

