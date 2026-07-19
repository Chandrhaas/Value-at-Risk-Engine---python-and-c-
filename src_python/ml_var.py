"""
Inference module for the trained quantile regression VaR model. Loads
models/var_model.pth + models/feature_scaler.pkl (produced by
train_var_model.py) and produces VaR estimates from live/recent market data,
for comparison against the Monte Carlo engine's estimates.

"""

import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from src_python.fetch_data import fetch_data  
from src_python.prepare_data import compute_log_returns, engineer_features_single_asset  
from src_python.train_var_model import QuantileVaRNet 

MODEL_DIR: str = os.path.join(root, "models")
INFERENCE_YEARS: int = 1  # only need ~64 trading days of warm-up; 1y is a comfortable buffer

def load_model_and_scaler(model_dir: str = MODEL_DIR, device: torch.device = torch.device("cpu")
                          ) -> Tuple[QuantileVaRNet, object, List[str], List[float]]:
    """
    Load the trained model + scaler saved by train_var_model.py.

    Returns
    -------
    (model, scaler, feature_columns, quantiles) 
    model.eval() already called ready for inference.
    """
    model_path = os.path.join(model_dir, "var_model.pth")
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Model/scaler not found in {model_dir}. Run train_var_model.py first."
        )

    # weights_only=False: this checkpoint is our own trusted file (not
    # something downloaded from the internet), and it deliberately stores
    # plain-Python metadata (hidden_dims, feature_columns, quantiles)
    # alongside the tensor weights, not just tensors. Newer torch versions
    # default torch.load to weights_only=True (a hardening measure aimed
    # at untrusted files), which would reject that metadata. Safe to
    # disable here since we wrote this file ourselves.
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = QuantileVaRNet(
        n_features=checkpoint["n_features"],
        hidden_dims=checkpoint["hidden_dims"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler, checkpoint["feature_columns"], checkpoint["quantiles"]


def build_latest_feature_row(ticker: str, years: int = INFERENCE_YEARS) -> pd.DataFrame:
    """
    Fetch recent price history for ONE ticker and build its most recent
    complete feature row.

    Reuses prepare_data.py's own compute_log_returns and
    engineer_features_single_asset, so inference features are built by
    the exact same code path as training features.

    Raises
    ------
    ValueError if the ticker doesn't have enough history (needs ~64+
    trading days for the vol_63 warm-up) to produce even one valid row e.g. a very recent IPO.
    """
    prices = fetch_data([ticker], period=years, align_dates=False)
    returns = compute_log_returns(prices)
    feature_table = engineer_features_single_asset(returns[ticker], ticker)
    feature_table = feature_table.dropna()

    if feature_table.empty:
        raise ValueError(
            f"Not enough trading history for {ticker} to build a feature row "
            f"(needs ~64+ trading days; try a longer-listed ticker)."
        )

    return feature_table.iloc[[-1]]  # most recent row, kept as a 1-row DataFrame


def predict_quantiles_for_ticker(
    ticker: str,
    model: QuantileVaRNet,
    scaler: object,
    feature_columns: List[str],
    years: int = INFERENCE_YEARS,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """
    Predict this ticker's next-day return quantiles (q05, q01) from its
    most recent trading history.

    Returns
    -------
    dict with keys: 'ticker', 'as_of_date', 'q05_return', 'q01_return'
    Returns are plain log returns (e.g. -0.02 means "predicted 5%/1% chance tomorrow is -2% or worse").
    """
    latest_row = build_latest_feature_row(ticker, years=years)
    X = scaler.transform(latest_row[feature_columns].values)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()[0]  # shape (2,) -> [q05, q01]

    return {
        "ticker": ticker,
        "as_of_date": str(latest_row.index[-1].date()),
        "q05_return": float(preds[0]),
        "q01_return": float(preds[1]),
    }


def predict_portfolio_var(
    tickers: List[str],
    weights: List[float],
    portfolio_value: float,
    model: QuantileVaRNet = None,
    scaler: object = None,
    feature_columns: List[str] = None,
    model_dir: str = MODEL_DIR,
    years: int = INFERENCE_YEARS,
) -> Dict[str, object]:
    """
    NN-based portfolio VaR estimate, for side by side comparison against
    the Monte Carlo engine's RiskMetricsOutput.

    The model predicts each ticker's OWN return quantiles independently. 
    The only honest way to combine independent per-ticker VaR figures into one
    portfolio number is to SUM each ticker's dollar VaR which implicitly
    assumes every asset moves against you simultaneously (perfect positive
    correlation). That's the most conservative combination possible, and it
    is NOT equivalent to what the Monte Carlo engine computes.

    Returns
    -------
    dict with keys:
        'var_95', 'var_99' : float, portfolio-level dollar VaR (positive = dollar loss magnitude, same sign convention
                                as the C++ engine's SimResult.var95/var99)
        'per_ticker'        : dict[ticker -> per-ticker breakdown], for inspecting which names are driving the total
    """
    if len(tickers) != len(weights):
        raise ValueError(f"tickers ({len(tickers)}) and weights ({len(weights)}) must be the same length.")

    weight_sum = sum(weights)
    if not (0.99 <= weight_sum <= 1.01):
        raise ValueError(f"weights must sum to 1.0 (got {weight_sum:.4f}).")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative (no short positions supported).")
 
    if model is None or scaler is None or feature_columns is None:
        model, scaler, feature_columns, _ = load_model_and_scaler(model_dir)

    per_ticker: Dict[str, Dict[str, object]] = {}
    var_95_total, var_99_total = 0.0, 0.0

    for ticker, weight in zip(tickers, weights):
        pred = predict_quantiles_for_ticker(ticker, model, scaler, feature_columns, years=years)
        ticker_value = portfolio_value * weight

        # Dollar VaR, positive-loss convention (matches C++ engine's
        # initial_value - final_value_at_quantile).
        var_95_ticker = ticker_value * -pred["q05_return"]
        var_99_ticker = ticker_value * -pred["q01_return"]
        
        if var_95_ticker < 0 or var_99_ticker < 0:
            print(f"[predict_portfolio_var] Note: {ticker} implies a NEGATIVE "
                    f"standalone VaR (var_95=${var_95_ticker:,.2f}, var_99=${var_99_ticker:,.2f}) "
                    f"model reads this ticker as low-risk/bullish enough that even "
                    f"its tail case is a gain.")

        per_ticker[ticker] = {
            **pred,
            "weight": weight,
            "dollar_value": ticker_value,
            "var_95": var_95_ticker,
            "var_99": var_99_ticker,
        }
        var_95_total += var_95_ticker
        var_99_total += var_99_ticker

    return {
        "var_95": var_95_total,
        "var_99": var_99_total,
        "per_ticker": per_ticker,
    }


if __name__ == "__main__":
    example_tickers = ["AAPL", "MSFT", "JPM"]
    example_weights = [1 / 3, 1 / 3, 1 / 3]
    example_portfolio_value = 1_000_000.0

    print(f"[main] Predicting NN VaR for {example_tickers}, "
          f"equal-weighted, ${example_portfolio_value:,.0f} portfolio")

    result = predict_portfolio_var(example_tickers, example_weights, example_portfolio_value)

    print("\n=== Per-ticker breakdown ===")
    for ticker, info in result["per_ticker"].items():
        print(f"{ticker} (as of {info['as_of_date']}): "
              f"q05={info['q05_return']:+.4f}  q01={info['q01_return']:+.4f}  "
              f"var_95=${info['var_95']:,.2f}  var_99=${info['var_99']:,.2f}")

    print("\n=== Portfolio total (NN, conservative/summed) ===")
    print(f"var_95: ${result['var_95']:,.2f}")
    print(f"var_99: ${result['var_99']:,.2f}")