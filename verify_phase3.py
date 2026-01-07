import pandas as pd
import numpy as np
from core.agi.profiler import AGIProfiler
from core.execution_engine import ExecutionEngine # Correct path
from core.swarm_orchestrator import SwarmOrchestrator

# Mock Classes
class MockDataLoader:
    def get_data(self):
        # Create Dummy Data
        # D1 for ATR (High Volatility)
        dates = pd.date_range(start="2024-01-01", periods=20)
        # Create moves of $50 per day (Very High Vol for XAUUSD)
        close = np.linspace(2000, 3000, 20) 
        high = close + 25
        low = close - 25
        df_d1 = pd.DataFrame({'close': close, 'high': high, 'low': low}, index=dates)
        
        # H1 for Entropy (High Chaos)
        # Random noise
        dates_h1 = pd.date_range(start="2024-01-01", periods=100, freq='h')
        close_h1 = np.random.normal(2000, 50, 100)
        df_h1 = pd.DataFrame({'close': close_h1}, index=dates_h1)
        
        return {'D1': df_d1, 'H1': df_h1}

print("=== VERIFYING PHASE 3: DYNAMIC INTELLIGENCE ===")

# 1. Test Profiler with High Volatility Data
loader = MockDataLoader()
profiler = AGIProfiler(loader)
rec = profiler.analyze_market_conditions()

print(f"\n[AGI PROFILER RESULT]")
print(f"Metrics: {rec['metrics']}")
print(f"Recommendation: {rec['reason']}")

atr = rec['metrics'].get('atr', 0)
entropy = rec['metrics'].get('entropy', 0)
vol_score = rec['metrics'].get('vol_score', 0)

if atr > 10 and entropy > 0.6:
    print("[PASS] Profiler correctly detected High Volatility/Chaos.")
else:
    print(f"[FAIL] Profiler metrics too low? ATR={atr} Ent={entropy}")

# 2. Test Dynamic Slots (Reduce in Chaos)
# Use the detected entropy/volatility
slots = SwarmOrchestrator(None,None,None,None,None).calculate_dynamic_slots(
    volatility=vol_score,
    trend_strength=50, # Weak trend
    mode="SNIPER"
)

print(f"\n[DYNAMIC SLOTS]")
print(f"Calculated Slots for Chaos: {slots}")

if slots <= 6:
    print("[PASS] Slots correctly reduced in Chaos.")
else:
    print(f"[FAIL] Slots too high for chaos ({slots}).")

# 3. Test Physical Stops (Widen in Volatility)
exec_engine = ExecutionEngine()
# Standard case ($2000 price)
tp, sl = exec_engine.calculate_smart_exits(
    price=2000, equity=1000, used_slots=1, max_slots=10, volatility=50,
    atr_value=atr # PASSING THE ATR
)

print(f"\n[PHYSICAL AWARENESS]")
print(f"ATR Input: {atr:.2f}")
print(f"Calculated SL Desc: {sl:.2f}")

# Standard SL is usually 1.5. ATR-based should be 1.5 * ATR (~50 * 1.5 = 75, clamped to 20 max)
if sl >= 10.0:
    print(f"[PASS] Stop Loss widened appropriately to {sl:.2f} (Max Cap Hit).")
elif sl > 2.0:
    print(f"[PASS] Stop Loss widened to {sl:.2f}.")
else:
    print(f"[FAIL] Stop Loss remained static/tight: {sl:.2f}")
