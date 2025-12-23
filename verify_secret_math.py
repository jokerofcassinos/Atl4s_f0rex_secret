import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantum_math import QuantumMath

def test_fisher_curvature():
    print("Testing Fisher Information Curvature...")
    # Case 1: Constant Price (Low Curiosity/Curvature)
    prices_flat = pd.Series([100.0] * 50)
    curv_flat = QuantumMath.fisher_information_curvature(prices_flat).iloc[-1]
    print(f"Flat Price Curvature: {curv_flat:.4f} (Expected: ~0)")

    # Case 2: Regime Shift (Mean shift)
    prices_shift = pd.Series([100.0] * 10 + [105.0] * 10)
    curv_shift = QuantumMath.fisher_information_curvature(prices_shift).iloc[-1]
    print(f"Regime Shift Curvature: {curv_shift:.4f} (Expected: > 0)")

    # Case 3: Volatility Explosion
    prices_vol = pd.Series([100.0 + np.random.normal(0, 0.1) for _ in range(10)] + 
                          [100.0 + np.random.normal(0, 2.0) for _ in range(10)])
    curv_vol = QuantumMath.fisher_information_curvature(prices_vol).iloc[-1]
    print(f"Volatility Explosion Curvature: {curv_vol:.4f} (Expected: > 0)")

def test_robust_hurst():
    print("\nTesting Robust Hurst Exponent...")
    # Case 1: Mean Reverting (Rough) - H < 0.5
    # Ornstein-Uhlenbeck like
    vals = [100.0]
    for _ in range(100):
        vals.append(vals[-1] + 0.5 * (100.0 - vals[-1]) + np.random.normal(0, 0.1))
    h_rev = QuantumMath.calculate_hurst_exponent(pd.Series(vals)).iloc[-1]
    print(f"Mean Reverting Hurst: {h_rev:.4f} (Expected: < 0.5)")

    # Case 2: Trending - H > 0.5
    vals = [100.0]
    for i in range(100):
        vals.append(vals[-1] + 0.1 + np.random.normal(0, 0.01))
    h_trend = QuantumMath.calculate_hurst_exponent(pd.Series(vals)).iloc[-1]
    print(f"Trending Hurst: {h_trend:.4f} (Expected: > 0.5)")

if __name__ == "__main__":
    test_fisher_curvature()
    test_robust_hurst()
