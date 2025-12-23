import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.macro_math import MacroMath

def test_macro_math():
    print("--- Testing Advanced Macro Math Engines ---")
    
    # 1. GARCH(1,1) Test
    # Generate data with volatility clustering
    np.random.seed(42)
    n = 100
    vols = np.zeros(n)
    vols[0] = 0.01
    for t in range(1, n):
        # Volatility clustering: omega=0.0001, alpha=0.1, beta=0.8
        vols[t] = np.sqrt(0.0001 + 0.1 * (np.random.normal(0, vols[t-1])**2) + 0.8 * vols[t-1]**2)
    
    returns = np.random.normal(0, vols)
    garch_std = MacroMath.garch_11_forecast(returns)
    print(f"GARCH(1,1) Forecasted Vol: {garch_std:.6f} (Standard Std: {np.std(returns):.6f})")

    # 2. Wavelet Haar MRA Test
    # Two sine waves with high coherence at low freq, but noisy at high freq
    t = np.linspace(0, 10, 64)
    s1 = np.sin(t) + np.random.normal(0, 0.1, 64)
    s2 = np.sin(t) + np.random.normal(0, 0.1, 64)
    
    w_res = MacroMath.wavelet_haar_mra(s1, s2)
    print(f"Wavelet Coherence (Haar): Trend={w_res['trend_coherence']:.2f}, Noise={w_res['noise_coherence']:.2f}")

    # 3. Cointegration Test
    # Generate two cointegrated series
    x = np.cumsum(np.random.normal(0, 1, 100))
    y = 2.0 * x + np.random.normal(0, 0.5, 100) # Cointegrated with beta=2
    
    coint_res = MacroMath.calculate_cointegration(y, x)
    print(f"Cointegration Test: Beta={coint_res['beta']:.2f}, StatScore={coint_res['stat_score']:.2f}, Z-Score={coint_res['z_score']:.2f}")

    # 4. Bayesian Regime Test (Recursive)
    # Generate secular trend
    drift_up = 0.005
    series = 100 + np.cumsum(np.random.normal(drift_up, 0.01, 100))
    prob = 0.5
    for i in range(20, 100):
        prob = MacroMath.bayesian_regime_detect(series[:i], prev_prob=prob)
    print(f"Bayesian Expansion Prob (Recursive +Drift): {prob:.4f}")
    
    drift_down = -0.005
    series_down = 100 + np.cumsum(np.random.normal(drift_down, 0.01, 100))
    prob_down = 0.5
    for i in range(20, 100):
        prob_down = MacroMath.bayesian_regime_detect(series_down[:i], prev_prob=prob_down)
    print(f"Bayesian Expansion Prob (Recursive -Drift): {prob_down:.4f}")

if __name__ == "__main__":
    test_macro_math()
