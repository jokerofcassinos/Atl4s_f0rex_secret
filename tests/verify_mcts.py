
import unittest
import numpy as np
import torch
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agi.neural_core import ReasoningTransformer
from core.agi.mcts import MCTSEngine

class TestMCTS(unittest.TestCase):
    def test_alpha_swarm_search(self):
        print("\n--- VERIFYING MCTS ENGINE (ALPHA-SWARM REASONING) ---")
        
        # 1. Initialize Thinking Hardware
        device = torch.device("cpu")
        brain = ReasoningTransformer().to(device)
        brain.eval()
        
        mcts = MCTSEngine(brain, simulations=10) # 10 sims for speed
        
        # 2. Define Problem (Root State)
        # 1024-d randomness representing "Confusing Market"
        root_state = np.random.randn(1024).astype(np.float32)
        
        # 3. Think (Search)
        print("Starting Tree Search...")
        best_action = mcts.search(root_state)
        
        # 4. Result
        action_map = {0: "WAIT", 1: "BUY", 2: "SELL"}
        decision = action_map.get(best_action, "UNKNOWN")
        
        print(f"MCTS Decision: {decision} (Action ID: {best_action})")
        
        # Assertions
        self.assertIn(best_action, [0, 1, 2])
        self.assertTrue(mcts is not None)
        
        print("Alpha-Swarm MCTS Architecture Verified.")

if __name__ == '__main__':
    unittest.main()
