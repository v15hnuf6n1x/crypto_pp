import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import XGBOOST_PARAMS

class XGBoostModel:
    """XGBoost model for price prediction"""
    
    def __init__(self, params=None):
        """
        Initialize XGBoost model
        
        Args:
            params: XGBoost parameters
        """
        self.params = params or XGBOOST_PARAMS
        self.model = None
    
    def train(self, X_train, y_train, num_boost_round=100):
        """
        Train XGBoost model
        
        Args:
            X_train: Training input data
            y_train: Training target data
            num_boost_round: Number of boosting rounds
        """
        print("Training XGBoost model...")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False
        )
        print("XGBoost model trained successfully!")
    
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test: Test input data
            
        Returns:
            Predictions
        """
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model_name': 'XGBoost'
        }
    
    def feature_importance(self, top_n=10):
        """Get feature importance"""
        importance = self.model.get_score(importance_type='weight')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_importance[:top_n]
    
    def save(self, filepath):
        """Save model"""
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")