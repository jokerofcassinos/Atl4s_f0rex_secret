
import unittest
import numpy as np
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agi.infinite_memory import holographic_memory
from core.agi.recursive_engine import recursive_engine

class TestInfiniteRecursion(unittest.TestCase):
    def test_holographic_recall(self):
        print("\n--- VERIFYING HOLOGRAPHIC MEMORY ---")
        # 1. Store a distinct event (Bullish Impulse)
        vec_a = np.zeros(1024, dtype=np.float32)
        vec_a[0] = 1.0 # Momentum High
        holographic_memory.store_event(vec_a, action=1, outcome=100.0)
        
        # 2. Store a distinct event (Bearish Crash)
        vec_b = np.zeros(1024, dtype=np.float32)
        vec_b[0] = -1.0 # Momentum Low
        holographic_memory.store_event(vec_b, action=2, outcome=-50.0)
        
        # 3. Query with similar Bullish vector
        query = np.zeros(1024, dtype=np.float32)
        query[0] = 0.9 # Similar to A
        
        results = holographic_memory.recall_patterns(query, k=1)
        self.assertTrue(len(results) > 0)
        best_match = results[0][0]
        score = results[0][1]
        
        print(f"Query Similarity Score: {score:.4f}")
        self.assertEqual(best_match['action'], 1) # Should match Bullish Event
        self.assertTrue(score > 0.8) # Should be highly similar

    def test_recursive_branching(self):
        print("\n--- VERIFYING INFINITE RECURSION ---")
        state = np.zeros(1024, dtype=np.float32)
        
        # Execute Deep Scan
        root = recursive_engine.deep_scan_recursive(state)
        
        # Verify Tree Structure
        print(f"Root Risk Score: {root['risk_score']}")
        print(f"Branch Count (Depth 1): {len(root['branches'])}")
        
        self.assertTrue(len(root['branches']) > 0)
        self.assertTrue(root['branches'][0]['depth'] == 1)
        
        # Check sub-branch (Depth 2)
        sub_branch = root['branches'][0]
        self.assertTrue(len(sub_branch['branches']) > 0)
        self.assertTrue(sub_branch['branches'][0]['depth'] == 2)
        
        print("Recursion Tree Validated (Depth 0 -> 1 -> 2)")

if __name__ == '__main__':
    unittest.main()
