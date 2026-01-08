import logging
import pandas as pd
import numpy as np
from core.agi.augur import Augur

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPhase7")

def test_augur_perception():
    print("\n--- TEST: AUGUR PERCEPTION (LORENTZIAN + SMC) ---")
    
    augur = Augur()
    
    # 1. Test Lorentzian Distance
    # Vector A: [1, 2, 3]
    # Vector B: [1.1, 2.1, 3.1]
    # Diff: [0.1, 0.1, 0.1]
    # Lorentz Metric: sum(ln(1 + 0.1)) = 3 * ln(1.1) approx 3 * 0.095 = 0.285
    
    vec_a = np.array([1.0, 2.0, 3.0])
    vec_b = np.array([1.1, 2.1, 3.1])
    
    dist = augur.lorentzian_distance(vec_a, vec_b)
    print(f"Lorentzian Distance: {dist:.4f}")
    
    if 0.28 < dist < 0.29:
        print("PASS: Lorentzian Math Check.")
    else:
        print(f"FAIL: Lorentzian Math is off. Got {dist}")

    # 2. Test Smart Money (FVG Detection)
    # Create a Bullish FVG Pattern
    # Candle 0: High=10, Low=5
    # Candle 1: High=12, Low=8 (Big Move)
    # Candle 2: High=15, Low=13 
    # Gap: Candle 2 Low (13) > Candle 0 High (10). FVG = (13+10)/2 = 11.5
    
    df = pd.DataFrame({
        'open': [5, 8, 13, 14, 15],
        'high': [10, 12, 15, 16, 17],
        'low':  [5, 8, 13, 13, 14],
        'close':[9, 11, 14, 15, 16]
    })
    
    print("\nDetecting FVG in synthetic data...")
    smc = augur.detect_smart_money(df)
    
    print(f"SMC Result: {smc}")
    
    if len(smc['fvg_bull']) > 0:
        gap = smc['fvg_bull'][0]
        print(f"Found Bullish FVG at level: {gap}")
        if gap == 11.5:
            print("PASS: FVG Detection Accurate.")
        else:
            print(f"FAIL: FVG Level Wrong. Expected 11.5, got {gap}")
    else:
        print("FAIL: No FVG Detected.")

if __name__ == "__main__":
    test_augur_perception()
