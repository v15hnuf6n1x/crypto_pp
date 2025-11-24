import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ProphetModel:
    """Facebook Prophet model for time series forecasting"""
    
    def __init__(self, interval_width=0.95, yearly_seasonality=True, 
                 weekly_seasonality=True, daily_seasonality=False):
        """
        Initialize Prophet model
        
        Args:
            interval_width: Width of prediction intervals
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
        """
        self.interval_width = interval_width
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
    
    def train(self, df):
        """
        Train Prophet model
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
        """
        try:
            print("Training Prophet model...")
            self.model = Prophet(
                interval_width=self.interval_width,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            self.model.fit(df)
            print("Prophet model trained successfully!")
            return True
        except Exception as e:
            print(f"Error training Prophet: {e}")
            return False
    
    def predict(self, periods=30):
        """
        Make predictions
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Forecast DataFrame
        """
        try:
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def evaluate(self, df_test):
        """
        Evaluate model
        
        Args:
            df_test: Test DataFrame with 'ds' and 'y' columns
            
        Returns:
            Dictionary with metrics
        """
        try:
            # Make predictions for test period
            test_future = df_test[['ds']].copy()
            forecast = self.model.predict(test_future)
            
            # Calculate metrics
            mse = mean_squared_error(df_test['y'], forecast['yhat'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(df_test['y'], forecast['yhat'])
            r2 = r2_score(df_test['y'], forecast['yhat'])
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_name': 'Prophet'
            }
        except Exception as e:
            print(f"Error evaluating Prophet: {e}")
            return None
    
    def save(self, filepath):
        """Save model"""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        import joblib
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")