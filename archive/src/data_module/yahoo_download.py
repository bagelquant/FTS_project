"""
Download Yahoo Finance data for a given ticker and date range.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def download_yahoo_data(ticker: str, 
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
    """
    Download Yahoo Finance data for a given ticker and date range.
    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    :param start_date: Start date for the data download (datetime object).
    :param end_date: End date for the data download (datetime object).
    :return: DataFrame containing the downloaded data.

    output format:
    index name: trade_date
    columns: ["open", "high", "low", "close", "vol", "pct_change", "vwap"]
    
    """
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    # Change column to single index and rename columns
    data = data.droplevel(level=1, axis=1)  # type: ignore
    # Rename columns to match the desired format
    data.columns = ["open", "high", "low", "close", "vol", "pct_change", "vwap"]
    # rename index to trade_date
    data.index.name = "trade_date"

    # calculate pct_change
    data["pct_change"] = data["close"].pct_change()
    # calculate daily volume-weighted average price
    
    
    return data

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    start_date = datetime(2004, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    data = download_yahoo_data(ticker, start_date, end_date)
    print(data.head()) 
