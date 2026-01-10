
import pandas as pd
import numpy as np
from signals.momentum import MomentumAnalyzer

def test_momentum():
    print("Testing Momentum Analyzer...")
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame(index=dates)
    df['close'] = np.random.randn(100).cumsum() + 100
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = df['close'].iloc[0]
    df['high'] = df[['open', 'close']].max(axis=1) + 0.1
    df['low'] = df[['open', 'close']].min(axis=1) - 0.1
    df['volume'] = 1000
    
    mom = MomentumAnalyzer()
    try:
        res = mom.analyze(df)
        print("Momentum Result Keys:", res.keys())
        if 'flow' in res:
            print("Flow Keys:", res['flow'].keys())
        else:
            print("❌ FLOW MISSING!")
            
        print("Composite:", res['composite'])
        print("✅ MOMENTUM OK")
    except Exception as e:
        print(f"❌ CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_momentum()
