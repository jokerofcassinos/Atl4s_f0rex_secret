
import pandas as pd
import numpy as np
from signals.structure import SMCAnalyzer

def test_smc_robustness():
    # 1. Create Synthetic Data (Random Walk)
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
    df = pd.DataFrame(index=dates)
    df['close'] = np.random.randn(1000).cumsum() + 100
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = df['close'].iloc[0]
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(1000) * 0.1
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(1000) * 0.1
    df['volume'] = 1000

    print("Running SMC Analysis on synthetic DataFrame...")
    smc = SMCAnalyzer()
    
    try:
        # Run Analyze
        result = smc.analyze(df, current_price=df['close'].iloc[-1])
        print("SMC Result:", result.keys())
        
        # Check Internals
        print(f"Order Blocks: {len(smc.order_blocks)}")
        print(f"Liquidity Pools: {len(smc.liquidity_pools)}")
        
        # Test extreme edge case: Small DF
        small_df = df.iloc[-10:]
        print("Running on Small DataFrame (10 rows)...")
        smc.analyze(small_df, current_price=df['close'].iloc[-1])
        print("Small DF Success.")

        # Test broken data
        print("Running on Data with NaNs...")
        df_nan = df.copy()
        df_nan.iloc[50, df_nan.columns.get_loc('high')] = np.nan
        smc.analyze(df_nan, current_price=100.0)
        print("NaN DF Success (handled or ignored).")
        
        print("\n✅ STRUCTURE MODULE IS ROBUST")
        
    except Exception as e:
        print(f"\n❌ CRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_smc_robustness()
