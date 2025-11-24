import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import FORECAST_DAYS, TRAIN_TEST_SPLIT

class DataPreprocessor:
    """Preprocess cryptocurrency data for ML models"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_price = MinMaxScaler(feature_range=(0, 1))
    
    def create_features(self, df, lookback=60):
        """
        Create technical indicators and features
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of days for rolling calculations
            
        Returns:
            DataFrame with features
        
        Raises:
            ValueError: If insufficient data is provided
        """
        df = df.copy()
        
        # Validate input data
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot create features.")
        
        if len(df) < 30:
            raise ValueError(
                f"Insufficient data: {len(df)} rows provided, but at least 30 rows "
                f"are required for feature engineering (SMA_30 requires 30 days of data)."
            )
        
        # Simple Moving Averages
        df['SMA_7'] = df['close'].rolling(window=7).mean()
        df['SMA_14'] = df['close'].rolling(window=14).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume features
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / (df['Volume_MA'] + 1)
        
        # Price change percentage
        df['Daily_Return'] = df['close'].pct_change()
        df['Price_Range'] = (df['high'] - df['low']) / df['close']
        
        # Volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        # Validate processed data
        if df.empty:
            raise ValueError(
                "All rows were dropped after feature engineering. "
                "This typically happens when there is insufficient data. "
                "Please provide more historical data."
            )
        
        return df
    
    def prepare_data_for_models(self, df, lookback=60, split_ratio=0.8):
        """
        Prepare data for training and testing
        
        Args:
            df: Preprocessed DataFrame
            lookback: Number of historical days to use
            split_ratio: Train/test split ratio
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
            
        Raises:
            ValueError: If insufficient data is provided
        """
        df = df.copy()
        
        # Validate input data
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot prepare data for models.")
        
        min_required = lookback + FORECAST_DAYS + 1
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} rows provided, but at least {min_required} rows "
                f"are required (lookback={lookback} + forecast_days={FORECAST_DAYS} + 1)."
            )
        
        # Extract close prices
        close_prices = df['close'].values.reshape(-1, 1)
        
        # Scale close prices
        scaled_prices = self.scaler_price.fit_transform(close_prices)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - lookback - FORECAST_DAYS):
            X.append(scaled_prices[i:i+lookback])
            y.append(scaled_prices[i+lookback+FORECAST_DAYS-1:i+lookback+FORECAST_DAYS])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split data
        split_idx = int(len(X) * split_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test, self.scaler_price
    
    def prepare_data_with_features(self, df, lookback=60, split_ratio=0.8):
        """
        Prepare data with technical features for tree-based models
        
        Args:
            df: Preprocessed DataFrame with features
            lookback: Number of historical days to use
            split_ratio: Train/test split ratio
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
            
        Raises:
            ValueError: If insufficient data is provided
        """
        df = df.copy()
        
        # Validate input data
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot prepare data for models.")
        
        min_required = lookback + FORECAST_DAYS + 1
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} rows provided, but at least {min_required} rows "
                f"are required (lookback={lookback} + forecast_days={FORECAST_DAYS} + 1)."
            )
        
        # Feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                       'SMA_7', 'SMA_14', 'SMA_30', 'EMA_12', 'EMA_26',
                       'MACD', 'RSI', 'BB_Upper', 'BB_Lower', 'Volume_Ratio',
                       'Daily_Return', 'Price_Range', 'Volatility']
        
        X = df[feature_cols].values
        y = df['close'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - lookback - FORECAST_DAYS):
            X_seq.append(X_scaled[i:i+lookback].flatten())
            y_seq.append(y[i+lookback+FORECAST_DAYS-1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split data
        split_idx = int(len(X_seq) * split_ratio)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        return X_train, X_test, y_train, y_test, feature_cols * lookback, self.scaler
    
    def inverse_transform(self, scaled_data):
        """Inverse transform scaled data back to original scale"""
        return self.scaler_price.inverse_transform(scaled_data)


if __name__ == '__main__':
    from src.data_loader import CryptoDataLoader
    
    loader = CryptoDataLoader()
    df = loader.get_data()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.create_features(df)
    
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data_for_models(df_processed)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")