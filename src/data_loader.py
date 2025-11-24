import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
from config import CRYPTOCURRENCIES, VS_CURRENCY, DAYS_LOOKBACK, DATA_DIR
from tqdm import tqdm

class CryptoDataLoader:
    """Load cryptocurrency data from various sources"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_from_yfinance(self, symbol, period='1y', interval='1d'):
        """
        Fetch cryptocurrency data from Yahoo Finance
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            print(f"Fetching {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df = df.reset_index()
            df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            df['symbol'] = symbol.replace('-USD', '')
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def load_from_csv(self, filepath):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def fetch_multiple_cryptos(self, symbols=None):
        """Fetch data for multiple cryptocurrencies"""
        if symbols is None:
            symbols = [f"{crypto}-USD" for crypto in CRYPTOCURRENCIES]
        
        all_data = []
        for symbol in symbols:
            df = self.fetch_from_yfinance(symbol)
            if df is not None:
                all_data.append(df)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('date').reset_index(drop=True)
            return combined_data
        return None
    
    def save_data(self, df, filename='crypto_data.csv'):
        """Save data to CSV"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def get_data(self, use_cached=True, filename='crypto_data.csv'):
        """
        Get crypto data with caching option
        
        Args:
            use_cached: Use cached data if available
            filename: Filename for cached data
            
        Returns:
            DataFrame with crypto data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if use_cached and os.path.exists(filepath):
            print(f"Loading cached data from {filepath}")
            return self.load_from_csv(filepath)
        
        df = self.fetch_multiple_cryptos()
        if df is not None:
            self.save_data(df, filename)
        return df


if __name__ == '__main__':
    loader = CryptoDataLoader()
    data = loader.get_data(use_cached=False)
    print(data.head())
    print(f"Data shape: {data.shape}")