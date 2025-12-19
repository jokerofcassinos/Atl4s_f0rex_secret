import pandas as pd
import numpy as np
import logging

# Mock logging
logging.basicConfig(level=logging.INFO)

# Import new modules
try:
    from analysis.smart_money import SmartMoneyEngine
    from analysis.deep_cognition import DeepCognition
    from analysis.hyper_dimension import HyperDimension
    from src.notifications import NotificationManager
    print("[PASS] Imports Successful")
except ImportError as e:
    print(f"[FAIL] Import Error: {e}")
    exit(1)

def create_mock_data():
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
    data = {
        'open': np.random.uniform(2000, 2010, 100),
        'high': np.random.uniform(2010, 2020, 100),
        'low': np.random.uniform(1990, 2000, 100),
        'close': np.random.uniform(1995, 2015, 100),
        'volume': np.random.randint(100, 1000, 100),
        'RSI': np.random.uniform(30, 70, 100), # Mock features
        'ATR': np.ones(100)
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_engines():
    df = create_mock_data()
    
    # 1. Smart Money
    smc = SmartMoneyEngine()
    score_smc = smc.analyze(df)
    print(f"[SmartMoney] Score: {score_smc}, FVGs: {len(smc.fvgs)}, OBs: {len(smc.order_blocks)}")
    
    # 2. Hyper Dimension
    hd = HyperDimension()
    score_hd, state_hd = hd.analyze_reality(df)
    print(f"[HyperDimension] Score: {score_hd}, State: {state_hd}")
    
    # 3. Deep Cognition
    dc = DeepCognition()
    
    # Mock Live Tick for Microstructure
    tick = {'time': 1678888000000, 'last': 2005.0, 'bid': 2004.9, 'ask': 2005.1, 'volume': 100}
    
    # Mock inputs
    decision, phy_state, future_prob = dc.consult_subconscious(
        trend_score=50, 
        volatility_score=20, 
        pattern_score=score_hd, 
        smc_score=score_smc,
        df_m5=df,
        live_tick=tick
    )
    print(f"[DeepCognition] Decision: {decision:.2f} | State: {phy_state} | Future Prob: {future_prob:.2f}")

    # 4. Notifications (Dry Run)
    nm = NotificationManager(cooldown_minutes=0)
    # We won't actually send to avoid spamming user during test, unless we mock subprocess
    # But init check is good enough.
    print(f"[NotificationManager] Initialized with cooldown {nm.cooldown}s")

if __name__ == "__main__":
    try:
        test_engines()
        print("\n>>> SYSTEM INTEGRITY CHECK PASSED <<<")
    except Exception as e:
        print(f"\n>>> SYSTEM FAIL: {e} <<<")
        import traceback
        traceback.print_exc()
