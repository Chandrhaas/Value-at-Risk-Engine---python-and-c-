import pandas as pd
import numpy as np
import os

def calculate_parameters(input="data/stock_prices.csv"):

    if not os.path.exists(input):
        print("Error input file not found. Please run fetch_data.py")
        return
    
    #dates will be treated as index
    prices = pd.read_csv(input,index_col=0,parse_dates=True)
    print(f"Loaded prices for{list(prices.columns)}")

    #returns = log(a t/ a t-1) , we take log beacause it is addidtive over time and addition is faster than multiplication
    returns = np.log(prices/prices.shift(1)).dropna()

    trading_days=252

    mean= returns.mean()*trading_days

    covs=returns.cov()*trading_days

    mean.to_csv("data/means.csv")
    covs.to_csv("data/covariance.csv")

    print("\nCalculated Means (Daily):")
    print(mean)
    print("\nCalculated Covariance Matrix:")
    print(covs)
    print("\nSuccess! Parameters saved to data/means.csv and data/cov_matrix.csv")

if __name__ == "__main__":
    calculate_parameters()
