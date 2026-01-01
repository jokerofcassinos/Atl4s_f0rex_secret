import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.fifth_eye import FifthEye

def test_fifth_eye():
    print("--- Testing Quinto Olho (Fifth Eye) ---")
    oracle = FifthEye()
    
    # Mock Data Map
    dates = pd.date_range(end=pd.Timestamp.now(), periods=150, freq='4h')
    prices = 2000 + np.cumsum(np.random.normal(0, 5, 150))
    df_h4 = pd.DataFrame({'close': prices, 'open': prices, 'high': prices+1, 'low': prices-1}, index=dates)
    
    dates_d1 = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    prices_d1 = 2000 + np.cumsum(np.random.normal(0, 20, 100))
    df_d1 = pd.DataFrame({'close': prices_d1, 'open': prices_d1, 'high': prices_d1+10, 'low': prices_d1-10}, index=dates_d1)
    
    data_map = {
        'H4': df_h4,
        'D1': df_d1,
        'W1': df_d1.resample('W').last()
    }
    
    # 1. Test Structure
    score, details = oracle.analyze_structure(data_map)
    print(f"Structural Score: {score} | Details: {details}")
    
    # 2. Test Cycles
    period = oracle.detect_cycles(df_d1)
    print(f"Dominant Cycle Period (Days): {period:.2f}")
    
    # 3. Test ADR Levels
    levels = oracle.calculate_adr_levels(df_d1)
    print(f"ADR Levels: {levels}")
    
    # 4. Test Integration
    # Mocking intermarket for speed (skip sync)
    oracle.intermarket_data = {
        'DXY': pd.DataFrame({'close': [100, 101]}, index=[0, 1]),
        'US10Y': pd.DataFrame({'close': [4.0, 4.1]}, index=[0, 1])
    }
    oracle.last_sync = pd.Timestamp.now()
    
    res = oracle.deliberate(data_map)
    print(f"\nDeliberation Result: {res['decision']} (Score: {res['score']})")

if __name__ == "__main__":
    test_fifth_eye()
