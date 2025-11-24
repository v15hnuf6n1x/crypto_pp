import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from config import LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE

class LSTMModel:
    """LSTM neural network for time series prediction"""
    
    def __init__(self, input_shape, units=LSTM_UNITS, dropout=LSTM_DROPOUT):
        """
        Initialize LSTM model
        
        Args:
            input_shape: Shape of input data (lookback, features)
            units: Number of LSTM units
            dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(self.units, activation='relu', input_shape=self.input_shape, return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units, activation='relu', return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units//2, activation='relu'),
            Dropout(self.dropout),
            Dense(self.units//4, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def train(self, X_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, 
              validation_split=0.1, verbose=1):
        """
        Train LSTM model
        
        Args:
            X_train: Training input data
            y_train: Training target data
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            verbose: Verbosity level
        """
        print("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=False
        )
        print("LSTM model trained successfully!")
        return history
    
    def predict(self, X_test):
        """
        Make predictions
        
        Args:
            X_test: Test input data
            
        Returns:
            Predictions
        """
        return self.model.predict(X_test, verbose=0)
    
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
            'model_name': 'LSTM'
        }
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")