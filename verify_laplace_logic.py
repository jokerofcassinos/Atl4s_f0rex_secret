import sys
import os
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

# Add project root
sys.path.append(os.getcwd())

from core.laplace_demon import LaplaceDemonCore

def create_mock_data():
    """Create realistic OHLCV data for testing."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='5min')
    df = pd.DataFrame(index=dates)
    
    # Generate random walk with trend
    df['close'] = np.cumsum(np.random.randn(500)) + 100
    df['open'] = df['close'].shift(1)
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(500))
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(500))
    df['volume'] = np.random.randint(100, 10000, 500)
    df.dropna(inplace=True)
    
    return df

try:
    print("--- STARTING LAPLACE DEMON VERIFICATION ---")
    
    # 1. Initialize
    print("Initializing LaplaceDemonCore...")
    demon = LaplaceDemonCore("GBPUSD")
    print("Optimization: Initialization passed.")
    
    # 2. Prepare Data
    # Mock data map
    # M1 Data (random walk)
    df_m1 = create_mock_data() 
    df_m5 = df_m1.resample('5min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
    
    data_map = {
        'M5': df_m5,
        'H1': df_m5.resample('1h').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
    }
    
    # 3. Time setup
    import datetime
    current_time = datetime.datetime.now()
    current_price = df_m5['close'].iloc[-1]
    
    print(f"Running Analysis at Price {current_price:.4f}...")
    
    # 4. Run Analysis (Async wrapper if needed, but analyze is async? No, check signature)
    # The analyze method in laplace_demon.py is async?
    # Let's check the code or assume. My previous view didn't show 'async def analyze', but 'def analyze'.
    # Wait, SwarmOrchestrator.process_tick is async.
    # LaplaceDemonCore.analyze calls it.
    # If LaplaceDemonCore.analyze is NOT async, but calls async code, it must use asyncio.run or loop.
    
    # Let's check if analyze is async.
    # In my edit (Step 52/53): "async def analyze(...)" was the signature?
    # Let's inspect the file signature to be sure.
    
    # I'll try running it as async first.
    import asyncio
    
    async def run_test():
        prediction = await demon.analyze(
            df_m1=df_m1,
            df_m5=df_m5,
            df_h1=data_map['H1'],
            df_h4=None,
            df_d1=None,
            current_time=current_time,
            current_price=current_price
        )
        return prediction

    # Check if 'analyze' is a coroutine function
    if asyncio.iscoroutinefunction(demon.analyze):
         print("Method 'analyze' is ASYNC.")
         prediction = asyncio.run(run_test())
    else:
         print("Method 'analyze' is SYNC.")
         prediction = demon.analyze(
            df_m1=None,
            df_m5=df_m5,
            df_h1=data_map['H1'],
            df_h4=None,
            df_d1=None,
            current_time=current_time,
            current_price=current_price
        )
         
    print("\n--- PREDICTION RESULT ---")
    print(f"Execute: {prediction.execute}")
    print(f"Direction: {prediction.direction}")
    print(f"Confidence: {prediction.confidence}")
    print(f"Primary Signal: {prediction.primary_signal}")
    print("Reasons:")
    for r in prediction.reasons:
        print(f" - {r}")
        
    print("\nSUCCESS: Verification Completed.")

except Exception as e:
    print(f"\nFAILURE: {e}")
    import traceback
    traceback.print_exc()
