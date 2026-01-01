import pandas as pd
import numpy as np
import logging
from analysis.ninth_eye import NinthEye

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verify-Singularity")

def test_singularity():
    print("\n" + "="*50)
    print("TESTING NINTH EYE (THE SINGULARITY)")
    print("="*50 + "\n")
    
    eye = NinthEye()
    
    # 1. Random Data (No Singularity)
    print("Scenario 1: Random Gaussian Noise (No Singularity)")
    dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
    df_noise = pd.DataFrame({
        'close': 2000 + np.random.normal(0, 1, 100),
        'volume': np.random.randint(100, 200, 100)
    }, index=dates)
    res = eye.deliberate({'M5': df_noise})
    print(f"Decision: {res['decision']} | Score: {res['score']:.2f} | Density: {res['density']:.2f}")
    
    # 2. Concentrated Periodic Data (Manifold folding + Density Collapse)
    print("\nScenario 2: Low-Variance Sine Wave (Singularity approaching)")
    # A very smooth sine wave is periodic, creating a clean phase-space loop
    x = np.linspace(0, 10 * np.pi, 100)
    prices = 2000 + 2 * np.sin(x)
    # Concentration: Most volume at the peak/bottom of sine
    volumes = 1.0 / (np.abs(np.sin(x)) + 0.1) 
    df_sync = pd.DataFrame({
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    res = eye.deliberate({'M5': df_sync})
    print(f"Decision: {res['decision']} | Score: {res['score']:.2f} | Geometry: {res['geometry']:.2f}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_singularity()
