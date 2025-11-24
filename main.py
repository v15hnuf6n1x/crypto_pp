"""
Cryptocurrency Price Prediction - Multi-Model ML Pipeline
Main execution script
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_loader import CryptoDataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.predictor import Predictor
from config import RESULTS_DIR, MODEL_DIR

def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("CRYPTOCURRENCY PRICE PREDICTION - MULTI-MODEL ML PIPELINE")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[Step 1] Loading Cryptocurrency Data...")
    loader = CryptoDataLoader()
    df = loader.get_data(use_cached=True)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    print(f"✓ Data loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Cryptocurrencies: {df['symbol'].unique().tolist()}")
    
    # Step 2: Preprocess Data
    print("\n[Step 2] Preprocessing Data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.create_features(df)
    print(f"✓ Data preprocessed successfully!")
    print(f"  Features created: {df_processed.shape[1]}")
    
    # Step 3: Train Models
    print("\n[Step 3] Training Multiple Models...")
    trainer = ModelTrainer()
    data_splits = trainer.train_all_models(df_processed)
    print("✓ All models trained successfully!")
    
    # Step 4: Evaluate Models
    print("\n[Step 4] Evaluating Models...")
    evaluator = ModelEvaluator()
    
    # Evaluate each model
    close_prices = df_processed['close'].values
    test_size = len(close_prices) // 5
    
    if 'ARIMA' in trainer.models:
        evaluator.evaluate_arima(trainer.models['ARIMA'], 
                                close_prices[-test_size:])
    
    if 'LSTM' in trainer.models:
        evaluator.evaluate_lstm(trainer.models['LSTM'],
                               data_splits['X_test'],
                               data_splits['y_test'])
    
    if 'XGBoost' in trainer.models:
        evaluator.evaluate_xgboost(trainer.models['XGBoost'],
                                  data_splits['X_test_xgb'],
                                  data_splits['y_test_xgb'])
    
    if 'Prophet' in trainer.models:
        prophet_df = df_processed[['date', 'close']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        test_df = prophet_df.iloc[-test_size:].reset_index(drop=True)
        evaluator.evaluate_prophet(trainer.models['Prophet'], test_df)
    
    print("✓ Models evaluated successfully!")
    
    # Step 5: Print Summary
    print("\n[Step 5] Model Evaluation Summary")
    summary_df = evaluator.print_summary()
    
    # Step 6: Save Results
    print("\n[Step 6] Saving Models and Results...")
    trainer.save_models()
    
    # Save evaluation results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'model_evaluation.csv'))
    
    # Save plots
    plot_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
    evaluator.plot_comparison(save_path=plot_path)
    
    print("✓ Models and results saved!")
    
    # Step 7: Make Predictions
    print("\n[Step 7] Making Predictions...")
    predictor = Predictor(trainer.models, scaler=data_splits['scaler'])
    
    # Get current price
    current_price = df_processed['close'].iloc[-1]
    
    # Make ensemble prediction
    ensemble_pred = predictor.ensemble_predict(
        data_splits['X_test'][:30],
        steps=30,
        weights={'LSTM': 0.3, 'XGBoost': 0.3, 'Prophet': 0.2, 'ARIMA': 0.2}
    )
    
    if ensemble_pred is not None:
        forecast_df = predictor.get_forecast_dataframe(
            current_price,
            ensemble_pred,
            current_date=df_processed['date'].max()
        )
        predictor.print_forecast(forecast_df)
        
        # Save forecast
        forecast_df.to_csv(os.path.join(RESULTS_DIR, 'price_forecast.csv'), 
                          index=False)
        print(f"✓ Forecast saved to {os.path.join(RESULTS_DIR, 'price_forecast.csv')}")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print("\n")

if __name__ == '__main__':
    main()