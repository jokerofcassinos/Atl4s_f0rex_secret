import pandas as pd
import numpy as np
import logging
from analysis.seventh_eye import SeventhEye
from src.data_loader import DataLoader

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify-SeventhEye")

def test_seventh_eye():
    print("\n" + "="*50)
    print("TESTING SEVENTH EYE (THE OVERLORD)")
    print("="*50 + "\n")
    
    overlord = SeventhEye()
    
    # Generate Synthetic Bullish Data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
    # Upward trend + low noise
    close = np.linspace(2000, 2010, 100) + np.random.normal(0, 0.2, 100)
    df_bullish = pd.DataFrame({
        'open': close - 0.1,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    print("Checking Bullish Synthetic Data...")
    res = overlord.deliberate({'M5': df_bullish})
    print(f"Decision: {res['decision']}")
    print(f"Score: {res['score']:.2f}")
    print(f"Reasoning: {res['reason']}")
    print(f"Metrics: {res['metrics']}")
    
    # Test Curvature Impact
    # Add a jump to simulate phase transition
    df_jump = df_bullish.copy()
    df_jump.iloc[-10:, df_jump.columns.get_loc('close')] += 5.0 
    
    print("\nChecking Phase Transition (Price Jump)...")
    res_jump = overlord.deliberate({'M5': df_jump})
    print(f"New Curvature: {res_jump['metrics']['curvature']:.4f}")
    print(f"New Score: {res_jump['score']:.2f}")
    
    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_seventh_eye()
