
import unittest
import torch
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agi.neural_core import ReasoningTransformer

class TestNeuralValue(unittest.TestCase):
    def test_vector_vision_forward(self):
        print("\n--- VERIFYING NEURAL VALUE FUNCTION (ALPHA-SWARM) ---")
        
        # 1. Initialize Model
        device = torch.device("cpu") # Test on CPU for speed
        model = ReasoningTransformer().to(device)
        model.eval()
        
        # 2. Simulate Perception Output (Batch=2, Dim=1024)
        # Random noise simulating distinct market states
        state_batch = torch.randn(2, 1024, dtype=torch.float32).to(device)
        
        # 3. Forward Pass (Intuition)
        with torch.no_grad():
            value_pred = model(state_batch)
            
        # 4. output Checks
        print(f"Input Shape: {state_batch.shape}")
        print(f"Output Shape: {value_pred.shape}")
        print(f"Predicted Values (Win Prob): {value_pred.flatten().tolist()}")
        
        self.assertEqual(value_pred.shape, (2, 1))
        
        # Sigmoid validation
        self.assertTrue(torch.all(value_pred >= 0.0))
        self.assertTrue(torch.all(value_pred <= 1.0))
        
        # 5. Check Text Mode Compatibility (Regression Test)
        text_batch = torch.randint(0, 32000, (2, 20)).to(device) # Batch=2, Seq=20 tokens
        with torch.no_grad():
            text_pred = model(text_batch)
        self.assertEqual(text_pred.shape, (2, 1))
        
        print("Neural Core verified for both Vision (Vectors) and Language (Tokens).")

if __name__ == '__main__':
    unittest.main()
