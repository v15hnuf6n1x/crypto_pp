import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelEvaluator:
    """Evaluate and compare multiple ML models"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
    
    def evaluate_arima(self, model, test_data):
        """Evaluate ARIMA model"""
        try:
            print("Evaluating ARIMA...")
            metrics = model.evaluate(test_data)
            self.results['ARIMA'] = metrics
            return metrics
        except Exception as e:
            print(f"Error evaluating ARIMA: {e}")
            return None
    
    def evaluate_lstm(self, model, X_test, y_test):
        """Evaluate LSTM model"""
        try:
            print("Evaluating LSTM...")
            metrics = model.evaluate(X_test, y_test)
            self.results['LSTM'] = metrics
            self.predictions['LSTM'] = model.predict(X_test)
            return metrics
        except Exception as e:
            print(f"Error evaluating LSTM: {e}")
            return None
    
    def evaluate_xgboost(self, model, X_test, y_test):
        """Evaluate XGBoost model"""
        try:
            print("Evaluating XGBoost...")
            metrics = model.evaluate(X_test, y_test)
            self.results['XGBoost'] = metrics
            self.predictions['XGBoost'] = model.predict(X_test)
            return metrics
        except Exception as e:
            print(f"Error evaluating XGBoost: {e}")
            return None
    
    def evaluate_prophet(self, model, test_df):
        """Evaluate Prophet model"""
        try:
            print("Evaluating Prophet...")
            metrics = model.evaluate(test_df)
            self.results['Prophet'] = metrics
            return metrics
        except Exception as e:
            print(f"Error evaluating Prophet: {e}")
            return None
    
    def get_best_model(self, metric='rmse'):
        """Get best model based on metric"""
        if not self.results:
            return None
        
        best_model = min(self.results.items(), 
                        key=lambda x: x[1].get(metric, float('inf')))
        return best_model[0]
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        summary_df = pd.DataFrame(self.results).T
        summary_df = summary_df.round(4)
        
        print("\nMetrics Comparison:")
        print(summary_df)
        
        print("\n" + "-"*80)
        print(f"Best Model (RMSE): {self.get_best_model('rmse')}")
        print(f"Best Model (MAE): {self.get_best_model('mae')}")
        print(f"Best Model (R2): {self.get_best_model('r2')}")
        print("="*80 + "\n")
        
        return summary_df
    
    def plot_comparison(self, save_path=None):
        """Plot model comparison"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        
        # RMSE
        rmse_vals = [self.results[m].get('rmse', 0) for m in models]
        axes[0, 0].bar(models, rmse_vals, color='steelblue')
        axes[0, 0].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 0].set_ylabel('RMSE')
        
        # MAE
        mae_vals = [self.results[m].get('mae', 0) for m in models]
        axes[0, 1].bar(models, mae_vals, color='coral')
        axes[0, 1].set_title('Mean Absolute Error (MAE)')
        axes[0, 1].set_ylabel('MAE')
        
        # MSE
        mse_vals = [self.results[m].get('mse', 0) for m in models]
        axes[1, 0].bar(models, mse_vals, color='lightgreen')
        axes[1, 0].set_title('Mean Squared Error (MSE)')
        axes[1, 0].set_ylabel('MSE')
        
        # R2 Score
        r2_vals = [self.results[m].get('r2', 0) for m in models]
        axes[1, 1].bar(models, r2_vals, color='orchid')
        axes[1, 1].set_title('R² Score (Higher is Better)')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filepath='results/model_evaluation.csv'):
        """Export results to CSV"""
        summary_df = pd.DataFrame(self.results).T
        summary_df.to_csv(filepath)
        print(f"Results exported to {filepath}")