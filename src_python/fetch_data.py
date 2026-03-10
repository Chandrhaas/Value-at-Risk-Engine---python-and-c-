import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date, output_path="data/stock_prices.csv"):

    print("Downloading data for : {tickers}..")

    #auto adjust = false makes sure data is real and not automatically accounts for splits/diviends etc
    # raw data has 5 columns open , high, low, close, adj close, volume
    raw_data=yf.download(tickers,start_date,end_date, auto_adjust=False)
    print(f"Columns: {raw_data.columns}")

    try:
        data = raw_data['Adj Close']
    except KeyError:
        print("Adj close not found , using close ")
        data = raw_data['Adj Close']
    
    if data.empty:
        print("No data found")
        return
    
    #handling missing values by forward and backward shift , we use forward shift first because on some day we know the previous day's value
    # but not the next day's , only for those days where we do not have a previous day's value we use the backward shift.
    data = data.ffill().bfill()

    # if data dir DNE then it makes it otherwsie leaves it alone 
    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    data.to_csv(output_path)

if __name__== "__main__":
    tickers=["AAPL","MSFT","GOOG","AMZN"]
    fetch_data(tickers,start_date="2022-01-01", end_date="2024-01-01")