"""
This script combines multiple tickers into a single, anonymous training dataset. 

While the script groups the data by ticker initially to calculate accurate 
historical features (like 5-day lags and rolling volatility), it strips away 
the ticker names and dates before passing the numbers to the model.

Because the final dataset contains no ticker identities, the neural network 
is forced to learn universal market behaviors rather than memorizing specific 
stock names. A row of data looks exactly the same to the model whether it 
originated from AAPL or XOM. This guarantees the model can accurately predict 
risk for brand new tickers it has never seen before.

Output shapes
--------------
X_train : (n_train_samples, 8) float64, standardized feature matrix
y_train : (n_train_samples, 1) float64, next-realized log return
X_test : (n_test_samples, 8) float64, standardized feature matrix
y_test : (n_test_samples, 1) float64, next-realized log return
n_samples = sum over tickers of (trading_days_available - warm_up_rows_dropped)
8 features = 4 AR lags (lag_1, lag_2, lag_3, lag_5)+ 4 rolling volatility windows (vol_5, vol_10, vol_21, vol_63)
"""

import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from src_python.fetch_data import fetch_data 


LAG_DAYS: List[int] = [1, 2, 3, 5] #for short term momentum
VOL_WINDOWS: List[int] = [5, 10, 21, 63] # for volatility
FEATURE_COLUMNS: List[str] = ["lag_1", "lag_2", "lag_3", "lag_5","vol_5", "vol_10", "vol_21", "vol_63"]
TRAIN_FRACTION: float = 0.8
DEFAULT_YEARS: int = 5 


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    -----------
    prices : pd.DataFrame
    Adjusted close prices, as returned by fetch_data.

    Returns
    -------
    pd.DataFrame
    Same columns as `prices`, first row dropped (structurally NaN for every ticker since there's no prior price on the very first date).
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.iloc[1:]
    print(f"[compute_log_returns] Log return matrix shape: {log_returns.shape}")
    return log_returns


def engineer_features_single_asset(returns: pd.Series, ticker: str) -> pd.DataFrame:
    """
    Build the feature/target table for ONE ticker's log-return series.

    Parameters
    ----------
    returns : pd.Series
    Log returns for a single ticker, indexed by date.
    ticker : str

    Returns
    ---------
    pd.DataFrame
    Columns: ['target', 'lag_1', 'lag_2', 'lag_3', 'lag_5','vol_5', 'vol_10', 'vol_21', 'vol_63', 'ticker']
    """
    
    df = pd.DataFrame(index=returns.index)
    df["target"] = returns
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = returns.shift(lag)

    # shift(1) first so no rolling window can see day t's own return
    lagged_returns = returns.shift(1)
    for win in VOL_WINDOWS:
        df[f"vol_{win}"] = lagged_returns.rolling(window=win).std()
    df["ticker"] = ticker

    return df


def build_asset_tables(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply engineer_features_single_asset to every ticker and stack the results into one long-format table, then drop NaN warm-up rows.

    Parameters
    ----------
    returns_df : pd.DataFrame
    Log returns, one column per ticker (output of compute_log_returns).

    Returns
    -------
    pd.DataFrame
    Long-format table: one row per (ticker, date) 
    columns =['target'] + FEATURE_COLUMNS + ['ticker']. 
    NaN rows dropped.
    """
    tables = [engineer_features_single_asset(returns_df[ticker], ticker) for ticker in returns_df.columns]
    combined = pd.concat(tables, axis=0)
    n_before = len(combined)
    combined = combined.dropna()
    n_after = len(combined)

    print(f"[build_asset_tables] Stacked {len(tables)} tickers, {n_before} raw rows. "
          f"Dropped {n_before - n_after} NaN warm-up rows (lags/rolling windows). "
          f"Remaining: {n_after}")
    
    return combined



def temporal_train_test_split(feature_table: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> "tuple[pd.DataFrame, pd.DataFrame]":
   
    """
    Chronological 80/20 split, applied independently within each ticker's own timeline, then recombined.
    For every ticker: the earliest `train_fraction` of its rows go to train, the most recent (1 - train_fraction) go to test.
    Strict temporal order is preserved within each ticker's series.

    Parameters
    ----------

    feature_table : pd.DataFrame
    Output of build_asset_tables (must contain a 'ticker' column).
    train_fraction : float
    Fraction of each ticker's history used for training. Default 0.8.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
    """
    train_parts, test_parts = [], []
    for ticker, group in feature_table.groupby("ticker", sort=False):
        group = group.sort_index()
        split_idx = int(len(group) * train_fraction)
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])

    train_df = pd.concat(train_parts).sort_index()
    test_df = pd.concat(test_parts).sort_index()

    print(f"[temporal_train_test_split] Train rows: {len(train_df)} "
          f"({train_fraction:.0%} target) | Test rows: {len(test_df)}")
    return train_df, test_df
    


def fit_scaler_and_transform(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit a StandardScaler on TRAINING features only, then transform both
    train and test. This prevents test-period statistics (mean/std) from
    leaking into the scaling applied to the training set.

    Parameters
    -----------
    train_df, test_df : pd.DataFrame (Outputs of temporal_train_test_split.)

    Returns
    ---------

    X_train : np.ndarray, shape (n_train, 8)
    y_train : np.ndarray, shape (n_train, 1)
    X_test : np.ndarray, shape (n_test, 8)
    y_test : np.ndarray, shape (n_test, 1)
    scaler : fitted sklearn.preprocessing.StandardScaler
    """
   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)
    y_train = train_df["target"].values.reshape(-1, 1)
    y_test = test_df["target"].values.reshape(-1, 1)

    print(f"[fit_scaler_and_transform] X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"[fit_scaler_and_transform] X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    return X_train, y_train, X_test, y_test, scaler


def transform_with_scaler(feature_table: pd.DataFrame, scaler: StandardScaler):
    """
    Unseen data is scaled using training universe's scaler.

    Parameters
    -----------
    feature_table : pd.DataFrame
    Output of build_asset_tables for tickers NOT used in training.
    scaler : StandardScaler
    The scaler fit on the training universe (from fit_scaler_and_transform).

    Returns
    -------
    X : np.ndarray, shape (n_rows, 8)
    y : np.ndarray, shape (n_rows, 1)

    """
    
    X = scaler.transform(feature_table[FEATURE_COLUMNS].values)
    y = feature_table["target"].values.reshape(-1, 1)
    print(f"[transform_with_scaler] X: {X.shape}  y: {y.shape}")
    return X, y


def build_feature_table(tickers: List[str], years: int = DEFAULT_YEARS) -> pd.DataFrame:
    """
    Fetch prices and build the engineered, unscaled, unsplit feature/target table for a given set of tickers.
    Reused for both the training universe(which then goes through temporal_train_test_split + fit_scaler_and_transform)
    and any held-out validation universe (which goes straight to transform_with_scaler).

    Parameters
    ----------
    tickers : List[str]
    years : int

    Returns
    -------
    pd.DataFrame
    Long-format table: one row per (ticker, date), columns =['target'] + FEATURE_COLUMNS + ['ticker'].

    """
    
    print(f"[build_feature_table] Fetching {years}y of data for {len(tickers)} "
          f"tickers via src_python.fetch_data.fetch_data(align_dates=False)")
    prices = fetch_data(tickers, period=years, align_dates=False)
    print(f"[build_feature_table] Price matrix shape: {prices.shape}")
    returns = compute_log_returns(prices)
    feature_table = build_asset_tables(returns)
    return feature_table


def prepare_var_dataset(tickers: List[str],years: int = DEFAULT_YEARS,train_fraction: float = TRAIN_FRACTION,) -> Dict[str, object]:
    """
    Run the full pipeline end to end for a TRAINING ticker universe:
    fetch -> returns -> features -> temporal split -> fit scaler.

    Returns
    -------
    dict with keys: 'X_train', 'y_train', 'X_test', 'y_test' : np.ndarray tensors
    'scaler' : fitted StandardScaler (reuse for any held-out ticker validation set, and at inference time)
    'feature_columns' : list[str], column order matching X_* columns
    'train_df', 'test_df' : the unscaled long-format tables, kept around for debugging / EDA (includes the 'ticker' column dropped from X).
    """
    
    feature_table = build_feature_table(tickers, years=years)
    train_df, test_df = temporal_train_test_split(feature_table, train_fraction)
    X_train, y_train, X_test, y_test, scaler = fit_scaler_and_transform(train_df, test_df)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "train_df": train_df,
        "test_df": test_df,
    }


def save_dataset(dataset: Dict[str, object], output_dir: str = "data") -> None:
    """
Writes into the repo's existing data/ folder:
{output_dir}/var_dataset.npz -> X_train, y_train, X_test, y_test
{output_dir}/feature_scaler.pkl -> fitted StandardScaler (joblib)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "var_dataset.npz")
    np.savez(
        npz_path,
        X_train=dataset["X_train"],
        y_train=dataset["y_train"],
        X_test=dataset["X_test"],
        y_test=dataset["y_test"],
    )
    scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
    joblib.dump(dataset["scaler"], scaler_path)

    print(f"[save_dataset] Saved tensors -> {npz_path}")
    print(f"[save_dataset] Saved scaler  -> {scaler_path}")


if __name__ == "__main__":

    TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "PG", "KO"]

    dataset = prepare_var_dataset(TICKERS, years=DEFAULT_YEARS)
    save_dataset(dataset, output_dir=os.path.join(root, "data"))

    print("\n=== Final Output Shapes ===")
    print(f"X_train: {dataset['X_train'].shape}   y_train: {dataset['y_train'].shape}")
    print(f"X_test:  {dataset['X_test'].shape}   y_test:  {dataset['y_test'].shape}")
    print(f"Features ({len(dataset['feature_columns'])}): {dataset['feature_columns']}")