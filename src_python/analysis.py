"""
This script calculates the daily returns, means of returns and covariance matrix.
"""

import pandas as pd
import numpy as np
import os

def calculate_parameters(prices: pd.DataFrame):

    #returns = log(a t/ a t-1) , we take log beacause it is addidtive over time and addition is faster than multiplication
    returns = np.log(prices/prices.shift(1)).dropna()

    mean_s= returns.mean()

    cov_s=returns.cov()

    return mean_s.to_numpy(), cov_s.to_numpy()