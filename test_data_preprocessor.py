"""
Test cases for data_preprocessor.py to validate empty DataFrame handling
"""

import pandas as pd
import numpy as np
import sys
from src.data_preprocessor import DataPreprocessor
from config import FORECAST_DAYS


def test_empty_dataframe():
    """Test that empty DataFrame raises appropriate error"""
    print("Test 1: Empty DataFrame")
    try:
        df_empty = pd.DataFrame()
        preprocessor = DataPreprocessor()
        result = preprocessor.create_features(df_empty)
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        expected_msg = "Input DataFrame is empty"
        if expected_msg in str(e):
            print(f"  ✓ PASS: Raised ValueError with message: '{e}'")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: '{e}'")
            return False


def test_insufficient_data():
    """Test that insufficient data raises appropriate error"""
    print("\nTest 2: Insufficient data (5 rows)")
    try:
        df_small = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'open': [100]*5,
            'high': [110]*5,
            'low': [90]*5,
            'close': [105]*5,
            'volume': [1000]*5
        })
        preprocessor = DataPreprocessor()
        result = preprocessor.create_features(df_small)
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        expected_msg = "at least 30 rows are required"
        if expected_msg in str(e):
            print(f"  ✓ PASS: Raised ValueError with message: '{e}'")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: '{e}'")
            return False


def test_sufficient_data():
    """Test that sufficient data processes successfully"""
    print("\nTest 3: Sufficient data (50 rows)")
    try:
        dates = pd.date_range('2024-01-01', periods=50)
        df_good = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1000, 2000, 50)
        })
        preprocessor = DataPreprocessor()
        result = preprocessor.create_features(df_good)
        if result.empty:
            print(f"  ✗ FAIL: Result DataFrame is empty")
            return False
        print(f"  ✓ PASS: Created features successfully, result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: Unexpected error: {e}")
        return False


def test_prepare_data_empty():
    """Test that prepare_data_for_models handles empty DataFrame"""
    print("\nTest 4: prepare_data_for_models with empty DataFrame")
    try:
        df_empty = pd.DataFrame()
        preprocessor = DataPreprocessor()
        result = preprocessor.prepare_data_for_models(df_empty)
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        expected_msg = "Input DataFrame is empty"
        if expected_msg in str(e):
            print(f"  ✓ PASS: Raised ValueError with message: '{e}'")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: '{e}'")
            return False


def test_prepare_data_insufficient():
    """Test that prepare_data_for_models handles insufficient data"""
    print("\nTest 5: prepare_data_for_models with insufficient data")
    min_required = 60 + FORECAST_DAYS + 1
    try:
        dates = pd.date_range('2024-01-01', periods=50)
        df_small = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(100, 110, 50),
        })
        preprocessor = DataPreprocessor()
        result = preprocessor.prepare_data_for_models(df_small, lookback=60)
        print("  ✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        expected_msg = f"at least {min_required} rows are required"
        if expected_msg in str(e):
            print(f"  ✓ PASS: Raised ValueError with message: '{e}'")
            return True
        else:
            print(f"  ✗ FAIL: Wrong error message: '{e}'")
            return False


def test_prepare_data_sufficient():
    """Test that prepare_data_for_models works with sufficient data"""
    print("\nTest 6: prepare_data_for_models with sufficient data")
    try:
        dates = pd.date_range('2024-01-01', periods=100)
        df_good = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(100, 110, 100),
        })
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data_for_models(
            df_good, lookback=60
        )
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"  ✗ FAIL: Empty train or test data")
            return False
        print(f"  ✓ PASS: Data prepared successfully")
        print(f"    X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: Unexpected error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("Testing DataPreprocessor empty DataFrame handling")
    print("="*70)
    
    tests = [
        test_empty_dataframe,
        test_insufficient_data,
        test_sufficient_data,
        test_prepare_data_empty,
        test_prepare_data_insufficient,
        test_prepare_data_sufficient,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*70)
    
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
