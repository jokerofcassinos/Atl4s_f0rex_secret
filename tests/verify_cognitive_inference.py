
import unittest
import pandas as pd
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agi.inference_engine import inference_engine

class TestCognitiveInference(unittest.TestCase):
    def test_trend_deduction(self):
        print("\n--- VERIFYING COGNITIVE INFERENCE (REAL THOUGHTS) ---")
        
        # Scenario 1: Price > EMA but Falling (Divergence, Positive Decay)
        # We need Price > EMA (102 > 100), but Price < Past Price (Negative Delta)
        # Past Price (index -10) needs to be HIGHER than current (102.5)
        # Let's set the base to 110, falling to 102.5
        closes = [110.0] * 50 + [109, 108, 107, 106, 104, 102.5]
        emas = [100.0] * 50 + [100, 100, 100, 100, 101, 101.5]
        
        data = {
            'close': closes,
            'ema_50': emas
        }
        df = pd.DataFrame(data)
        context = {'df_m1': df, 'domain': 'TrendArchitect'}
        
        thought = inference_engine.deduce("Is the trend healthy?", "TrendArchitect", context)
        print(f"Scenario 1 (Divergence): {thought}")
        
        self.assertIn("DECAYING", thought)
        
    def test_sniper_deduction(self):
         # Scenario 2: Hammer Candle (Wick Rejection)
         # Open 10, Close 11, Low 5, High 11. Body=1. Lower Wick=5. Ratio=5x.
         tick = {}
         data = {
             'open': [10], 'close': [11], 'low': [5], 'high': [11]
         }
         df = pd.DataFrame(data)
         context = {'df_m1': df, 'tick': tick, 'domain': 'SniperSwarm'}
         
         thought = inference_engine.deduce("Is there a liquidity grab?", "SniperSwarm", context)
         print(f"Scenario 2 (Sniper Hammer): {thought}")
         
         self.assertIn("High Probability Rejection", thought)

if __name__ == '__main__':
    unittest.main()
