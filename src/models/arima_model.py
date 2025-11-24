import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    """ARIMA model for time series forecasting"""
    
    def __init__(self, order=(5, 1, 2)):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) tuple for ARIMA parameters
        """
        self.order = order
        self.model = None
        self.results = None
    
    def train(self, data):
        """
        Train ARIMA model
        
        Args:
            data: Time series data (1D array or Series)
        """
        try:
            print(f"Training ARIMA{self.order}...")
            self.model = ARIMA(data, order=self.order)
            self.results = self.model.fit()
            print("ARIMA model trained successfully!")
            return True
        except Exception as e:
            print(f"Error training ARIMA: {e}")
            return False
    
    def predict(self, steps=30):
        """
        Make predictions
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecast values
        """
        try:
            forecast = self.results.get_forecast(steps=steps)
            forecast_values = forecast.predicted_mean.values
            return forecast_values
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, test_data):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test time series data
            
        Returns:
            Dictionary with metrics
        """
        try:
            # In-sample predictions (for training data)
            train_predictions = self.results.fittedvalues
            
            # Adjust test data to match predictions length
            if len(train_predictions) < len(test_data):
                test_data_adj = test_data[:len(train_predictions)]
            else:
                test_data_adj = test_data
            
            # Make new predictions for test period
            forecast = self.predict(steps=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_data[:len(forecast)], forecast)
            r2 = r2_score(test_data[:len(forecast)], forecast)
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_name': 'ARIMA'
            }
        except Exception as e:
            print(f"Error evaluating ARIMA: {e}")
            return None
    
    def get_summary(self):
        """Get model summary"""
        if self.results:
            return self.results.summary()
        return None