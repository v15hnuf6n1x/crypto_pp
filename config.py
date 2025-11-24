import os
from dotenv import load_dotenv

load_dotenv()

# Cryptocurrency Configuration
CRYPTOCURRENCIES = ['BTC', 'ETH', 'XRP', 'LTC']
VS_CURRENCY = 'usd'
DAYS_LOOKBACK = 365
TRAIN_TEST_SPLIT = 0.8

# Model Configuration
FORECAST_DAYS = 30
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# LSTM Configuration
LSTM_UNITS = 50
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32

# XGBoost Configuration
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}

# Model Paths
MODEL_DIR = 'models/saved'
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# API Keys (optional for live data)
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')