"""
Quick verification script for Laplace Demon modules.
"""
import sys
print("Starting verification...", flush=True)

# Test imports
try:
    print("1. Testing signals.timing...", flush=True)
    from signals.timing import QuarterlyTheory, M8FibonacciSystem
    print("   ✅ timing OK", flush=True)
except Exception as e:
    print(f"   ❌ timing FAILED: {e}", flush=True)

try:
    print("2. Testing signals.structure...", flush=True)
    from signals.structure import SMCAnalyzer, BlackRockPatterns
    print("   ✅ structure OK", flush=True)
except Exception as e:
    print(f"   ❌ structure FAILED: {e}", flush=True)

try:
    print("3. Testing signals.correlation...", flush=True)
    from signals.correlation import SMTDivergence, AMDPowerOfThree
    print("   ✅ correlation OK", flush=True)
except Exception as e:
    print(f"   ❌ correlation FAILED: {e}", flush=True)

try:
    print("4. Testing signals.momentum...", flush=True)
    from signals.momentum import MomentumAnalyzer
    print("   ✅ momentum OK", flush=True)
except Exception as e:
    print(f"   ❌ momentum FAILED: {e}", flush=True)

try:
    print("5. Testing signals.volatility...", flush=True)
    from signals.volatility import VolatilityAnalyzer
    print("   ✅ volatility OK", flush=True)
except Exception as e:
    print(f"   ❌ volatility FAILED: {e}", flush=True)

try:
    print("6. Testing core.laplace_demon...", flush=True)
    from core.laplace_demon import LaplaceDemonCore
    print("   ✅ laplace_demon OK", flush=True)
except Exception as e:
    print(f"   ❌ laplace_demon FAILED: {e}", flush=True)

try:
    print("7. Testing backtest.engine...", flush=True)
    from backtest.engine import BacktestEngine, BacktestConfig
    print("   ✅ backtest OK", flush=True)
except Exception as e:
    print(f"   ❌ backtest FAILED: {e}", flush=True)

print("\n" + "="*50, flush=True)
print("Verification complete!", flush=True)
