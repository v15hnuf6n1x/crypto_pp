import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import os

from src.models import ARIMAModel, LSTMModel, XGBoostModel, ProphetModel
from src.data_preprocessor import DataPreprocessor
from config import MODEL_DIR, FORECAST_DAYS

class ModelTrainer:
    """Train multiple ML models for price prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.preprocessor = DataPreprocessor()
    
    def train_arima(self, data, order=(5, 1, 2)):
        """Train ARIMA model"""
        print("\n" + "="*50)
        print("Training ARIMA Model")
        print("="*50)
        
        model = ARIMAModel(order=order)
        if model.train(data):
            self.models['ARIMA'] = model
            return model
        return None
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=100):
        """Train LSTM model"""
        print("\n" + "="*50)
        print("Training LSTM Model")
        print("="*50)
        
        input_shape = (X_train.shape[1], X_train.shape[2]) if len(X_train.shape) > 2 else (X_train.shape[1], 1)
        model = LSTMModel(input_shape=input_shape)
        
        # Reshape if needed
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model.train(X_train, y_train, epochs=epochs)
        self.models['LSTM'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        model = XGBoostModel()
        model.train(X_train, y_train)
        self.models['XGBoost'] = model
        return model
    
    def train_prophet(self, df):
        """Train Prophet model"""
        print("\n" + "="*50)
        print("Training Prophet Model")
        print("="*50)
        
        # Prepare data in Prophet format
        prophet_df = df[['date', 'close']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        model = ProphetModel()
        if model.train(prophet_df):
            self.models['Prophet'] = model
            return model
        return None
    
    def train_all_models(self, df):
        """
        Train all models
        
        Args:
            df: Preprocessed DataFrame with features
        """
        print("\n" + "="*70)
        print("STARTING MULTI-MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Extract close prices for ARIMA and Prophet
        close_prices = df['close'].values
        
        # Train ARIMA
        self.train_arima(close_prices)
        
        # Train Prophet
        self.train_prophet(df)
        
        # Prepare data for neural networks and tree models
        X_train, X_test, y_train, y_test, scaler = self.preprocessor.prepare_data_for_models(df)
        
        # Train LSTM
        self.train_lstm(X_train, y_train, X_test, y_test)
        
        # Prepare data for XGBoost
        X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, features, scaler_xgb = \
            self.preprocessor.prepare_data_with_features(df)
        
        # Train XGBoost
        self.train_xgboost(X_train_xgb, y_train_xgb)
        
        print("\n" + "="*70)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_xgb': X_train_xgb,
            'X_test_xgb': X_test_xgb,
            'y_train_xgb': y_train_xgb,
            'y_test_xgb': y_test_xgb,
            'scaler': scaler
        }
    
    def save_models(self, directory=MODEL_DIR):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_model.pkl')
            try:
                if name == 'LSTM':
                    model.save(filepath.replace('.pkl', '.h5'))
                elif name == 'XGBoost':
                    model.save(filepath.replace('.pkl', '.model'))
                elif name == 'Prophet':
                    model.save(filepath)
                else:
                    joblib.dump(model, filepath)
                print(f"Saved {name} model to {filepath}")
            except Exception as e:
                print(f"Error saving {name} model: {e}")
    
    def load_models(self, directory=MODEL_DIR):
        """Load trained models"""
        for name in ['ARIMA', 'LSTM', 'XGBoost', 'Prophet']:
            filepath = os.path.join(directory, f'{name}_model.pkl')
            try:
                if name == 'LSTM':
                    filepath = filepath.replace('.pkl', '.h5')
                    model = LSTMModel((60, 1))
                    model.load(filepath)
                    self.models[name] = model
                elif name == 'XGBoost':
                    filepath = filepath.replace('.pkl', '.model')
                    model = XGBoostModel()
                    model.load(filepath)
                    self.models[name] = model
                elif name == 'Prophet':
                    model = ProphetModel()
                    model.load(filepath)
                    self.models[name] = model
                else:
                    self.models[name] = joblib.load(filepath)
                print(f"Loaded {name} model from {filepath}")
            except Exception as e:
                print(f"Error loading {name} model: {e}")