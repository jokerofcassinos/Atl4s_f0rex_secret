import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.sixth_eye import SixthEye

def test_sixth_eye():
    print("--- Testing Sexto Olho (Sixth Eye) ---")
    council = SixthEye()
    
    # 1. Mock MN/W1 Data Map
    dates_mn = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='MS')
    prices_mn = 2000 + np.cumsum(np.random.normal(0, 50, 24))
    df_mn = pd.DataFrame({'close': prices_mn, 'open': prices_mn, 'high': prices_mn+20, 'low': prices_mn-20}, index=dates_mn)
    
    dates_w1 = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='W')
    prices_w1 = 2000 + np.cumsum(np.random.normal(0, 30, 60))
    df_w1 = pd.DataFrame({'close': prices_w1, 'open': prices_w1, 'high': prices_w1+15, 'low': prices_w1-15, 'volume': np.random.randint(1000, 5000, 60)}, index=dates_w1)
    
    data_map = {
        'MN': df_mn,
        'W1': df_w1
    }
    
    # 2. Test Secular Trend
    score, details = council.analyze_secular_trend(data_map)
    print(f"Secular Trend Score: {score} | Details: {details}")
    
    # 3. Test Implicit Real Rates (Mocked Context)
    council.macro_context = {
        'TIP': pd.DataFrame({'close': [110, 111, 112, 113, 114]}, index=[0, 1, 2, 3, 4]),
        'Yield': pd.DataFrame({'close': [4.0, 3.9, 3.8, 3.7, 3.6]}, index=[0, 1, 2, 3, 4])
    }
    council.last_macro_sync = pd.Timestamp.now()
    real_rate_bias = council.calculate_implicit_real_rates()
    print(f"Real Rate Bias: {real_rate_bias} (Expected positive for rising TIP and falling Yield)")
    
    # 4. Test COT Proxy
    # Set rising price and rising volume
    df_w1.iloc[-4:, df_w1.columns.get_loc('close')] = df_w1['close'].iloc[-4:] * 1.1
    df_w1.iloc[-4:, df_w1.columns.get_loc('volume')] = df_w1['volume'].iloc[-4:] * 1.5
    cot_bias = council.cot_proxy_analysis(df_w1)
    print(f"COT Proxy Bias: {cot_bias} (Expected positive for breakout on volume)")
    
    # 5. Full Deliberation
    res = council.deliberate(data_map)
    print(f"\nPosition Anchor Recommendation: {res['anchor']} (Score: {res['score']})")

if __name__ == "__main__":
    test_sixth_eye()
