
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agi.perception import perception

class TestPerception(unittest.TestCase):
    def test_vector_encoding(self):
        print("\n--- VERIFYING PERCEPTION ENGINE (VECTOR VISION) ---")
        
        # 1. Generate Dummy Data (ZigZag)
        closes = [100.0 + i for i in range(50)] # Steady uptrend
        opens = closes
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        
        data = {
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'tick_volume': [1000]*50,
            'rsi': [60]*50,
            'ema_50': [c-2 for c in closes] # Price always above EMA
        }
        df = pd.DataFrame(data)
        
        # 2. Mock Context with Swarm
        context = {
            'df_m1': df,
            'tick': {'bid': 150.0, 'ask': 150.1},
            'swarm_votes': {
                'TrendArchitect': {'signal': 'BUY', 'conf': 0.9},
                'SniperSwarm': {'signal': 'WAIT', 'conf': 0.0}
            }
        }
        
        # 3. Encode
        state_vec = perception.encode_state(context)
        
        # 4. Verify Geometry
        print(f"Vector Shape: {state_vec.shape}")
        print(f"Vector Type: {state_vec.dtype}")
        print(f"Sample Values: {state_vec[:10]}") # Log Returns
        
        self.assertEqual(state_vec.shape, (1024,))
        self.assertEqual(state_vec.dtype, np.float32)
        
        # Verify Price Normalization (Log returns of steady line should be near const)
        # Log ret of 100->101 is ~0.01. Std dev might make it varying.
        # Just check bounds.
        self.assertTrue(np.max(state_vec) <= 3.0) # check clipping
        self.assertTrue(np.min(state_vec) >= -3.0)
        
        # Verify Swarm Embedding
        # Trend Architect hash
        mod_hash = abs(hash('TrendArchitect')) % 100
        # The offset is 30 (returns) + 3 (technicals) = 33. So 33 + hash.
        # Actually exact index doesn't matter as long as it's non-zero
        swarm_segment = state_vec[33:133] 
        self.assertTrue(np.any(swarm_segment != 0), "Swarm votes not encoded!")
        
        print("Perception Engine Validated.")

if __name__ == '__main__':
    unittest.main()
