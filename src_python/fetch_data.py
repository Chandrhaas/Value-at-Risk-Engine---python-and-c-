import yfinance as yf
import pandas as pd
from typing import List

def fetch_data(tickers:List[str],period:int=1):

    per=str(period)+'y'

    print("Downloading data for : {tickers}..")
    try:
        tickers_str=" ".join(tickers)
        #auto adjust = false makes sure data is real and not automatically accounts for splits/diviends etc
        # raw data has 5 columns open , high, low, close, adj close, volume
        raw_data=yf.download(tickers_str,period=per,auto_adjust=False,progress=False)
        
        try:
            data = raw_data['Adj Close']
        except KeyError:
            print("Adj close not found , using close ")
            data = raw_data['Close']

        #yfinance gives different output format for single vs multiple tickers 
        #for consistency we are forcing a single ticker intput into a 2D table
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
    
        if data.empty:
            raise ValueError("Yahoo Finance returned empty data. Check your tickers.")
        
        
        
        #handling missing values by forward and backward shift , we use forward shift first because on some day we know the previous day's value
        # but not the next day's , only for those days where we do not have a previous day's value we use the backward shift.
        data = data.ffill().bfill()

        #Drop any days where a stock didn't trade
        data.dropna(inplace=True)

        print(f"Successfully loaded {len(data)} trading days into memory.")
        return data
    
    except Exception as e:
        # If this fails,trigger a 500 error safely
        raise ValueError(f"Data fetch failed: {str(e)}")
