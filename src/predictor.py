import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class Predictor:
    """Make predictions using trained models"""
    
    def __init__(self, models, scaler=None):
        """
        Initialize predictor
        
        Args:
            models: Dictionary of trained models
            scaler: Data scaler for inverse transformation
        """
        self.models = models
        self.scaler = scaler
    
    def predict_arima(self, steps=30):
        """Make ARIMA predictions"""
        try:
            if 'ARIMA' not in self.models:
                return None
            
            forecast = self.models['ARIMA'].predict(steps=steps)
            return forecast
        except Exception as e:
            print(f"Error in ARIMA prediction: {e}")
            return None
    
    def predict_lstm(self, X_test, scaler=None):
        """Make LSTM predictions"""
        try:
            if 'LSTM' not in self.models:
                return None
            
            predictions = self.models['LSTM'].predict(X_test)
            
            if scaler:
                predictions = scaler.inverse_transform(predictions)
            
            return predictions.flatten()
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return None
    
    def predict_xgboost(self, X_test):
        """Make XGBoost predictions"""
        try:
            if 'XGBoost' not in self.models:
                return None
            
            predictions = self.models['XGBoost'].predict(X_test)
            return predictions
        except Exception as e:
            print(f"Error in XGBoost prediction: {e}")
            return None
    
    def predict_prophet(self, steps=30):
        """Make Prophet predictions"""
        try:
            if 'Prophet' not in self.models:
                return None
            
            forecast = self.models['Prophet'].predict(periods=steps)
            return forecast['yhat'].values
        except Exception as e:
            print(f"Error in Prophet prediction: {e}")
            return None
    
    def ensemble_predict(self, X_test, steps=30, weights=None):
        """
        Make ensemble prediction combining all models
        
        Args:
            X_test: Test data for neural network models
            steps: Number of steps for time series models
            weights: Weights for each model (default: equal)
            
        Returns:
            Ensemble prediction
        """
        predictions = {}
        
        # Get predictions from each model
        if 'LSTM' in self.models:
            lstm_pred = self.predict_lstm(X_test, self.scaler)
            if lstm_pred is not None:
                predictions['LSTM'] = lstm_pred[:steps]
        
        if 'XGBoost' in self.models:
            xgb_pred = self.predict_xgboost(X_test)
            if xgb_pred is not None:
                predictions['XGBoost'] = xgb_pred[:steps]
        
        if 'Prophet' in self.models:
            prophet_pred = self.predict_prophet(steps=steps)
            if prophet_pred is not None:
                predictions['Prophet'] = prophet_pred
        
        if 'ARIMA' in self.models:
            arima_pred = self.predict_arima(steps=steps)
            if arima_pred is not None:
                predictions['ARIMA'] = arima_pred
        
        if not predictions:
            return None
        
        # Set default equal weights
        if weights is None:
            weights = {model: 1/len(predictions) for model in predictions}
        
        # Calculate weighted average
        ensemble_pred = np.zeros(steps)
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            # Handle different prediction lengths
            min_len = min(len(pred), steps)
            ensemble_pred[:min_len] += pred[:min_len] * weight
        
        return ensemble_pred
    
    def get_forecast_dataframe(self, current_price, ensemble_pred, 
                               current_date=None, interval='D'):
        """
        Create forecast dataframe
        
        Args:
            current_price: Current price
            ensemble_pred: Ensemble predictions
            current_date: Current date (default: today)
            interval: Time interval (D for daily, H for hourly)
            
        Returns:
            DataFrame with forecast
        """
        if current_date is None:
            current_date = datetime.now()
        
        dates = []
        prices = []
        
        if interval == 'D':
            delta = timedelta(days=1)
        elif interval == 'H':
            delta = timedelta(hours=1)
        else:
            delta = timedelta(days=1)
        
        for i, price in enumerate(ensemble_pred):
            date = current_date + delta * (i + 1)
            dates.append(date)
            prices.append(price)
        
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecast_price': prices,
            'change_from_current': prices - current_price,
            'pct_change': ((prices - current_price) / current_price) * 100
        })
        
        return forecast_df
    
    def print_forecast(self, forecast_df):
        """Print forecast nicely"""
        print("\n" + "="*80)
        print("PRICE FORECAST")
        print("="*80)
        print(forecast_df.to_string(index=False))
        print("="*80 + "\n")