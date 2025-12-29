import pandas as pd
import numpy as np
import logging
from analysis.eighth_eye import EighthEye

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify-Sovereign")

def generate_aligned_data(direction=1):
    dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
    base_price = 2000
    
    # Generate M5
    m5_noise = np.random.normal(0, 0.1, 100)
    m5_trend = np.linspace(0, 5 * direction, 100)
    m5_close = base_price + m5_trend + m5_noise
    
    df_m5 = pd.DataFrame({'close': m5_close}, index=dates)
    
    # Generate H1 (32 bars)
    h1_close = base_price + np.linspace(0, 10 * direction, 32) + np.random.normal(0, 0.5, 32)
    df_h1 = pd.DataFrame({'close': h1_close})
    
    # Generate H4
    h4_close = base_price + np.linspace(0, 20 * direction, 32) + np.random.normal(0, 1.0, 32)
    df_h4 = pd.DataFrame({'close': h4_close})
    
    # Generate D1
    d1_close = base_price + np.linspace(0, 40 * direction, 32) + np.random.normal(0, 2.0, 32)
    df_d1 = pd.DataFrame({'close': d1_close})
    
    return {
        'M5': df_m5,
        'H1': df_h1,
        'H4': df_h4,
        'D1': df_d1
    }

def test_sovereign():
    print("\n" + "="*50)
    print("TESTING EIGHTH EYE (THE SOVEREIGN)")
    print("="*50 + "\n")
    
    sov = EighthEye()
    
    # 1. Total Alignment (Bullish)
    print("Scenario 1: Full Bullish Alignment (M5, H1, H4, D1)")
    data_aligned = generate_aligned_data(direction=1)
    res = sov.deliberate(data_aligned)
    print(f"Alignment: {res['alignment']}")
    print(f"Decision: {res['decision']}")
    print(f"Coherences: {res['coherences']}")
    
    # 2. Conflict
    print("\nScenario 2: Market Conflict (D1 Bearish)")
    data_aligned['D1']['close'] = 2050 - np.linspace(0, 40, 32)
    res_conflict = sov.deliberate(data_aligned)
    print(f"Alignment: {res_conflict['alignment']}")
    print(f"Decision: {res_conflict['decision']}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_sovereign()
