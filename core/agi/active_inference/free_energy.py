
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("FreeEnergy")

class FreeEnergyMinimizer:
    """
    Sistema D-9: Free Energy Minimizer.
    Calculates 'Surprise' (Prediction Error).
    To survive, the agent must minimize Free Energy (F).
    F = Surprise + Complexity (ignored for now)
    """
    def __init__(self):
        self.surprise_history = []
        
    def calculate_surprise(self, prediction: Dict[str, float], reality: Dict[str, Any]) -> float:
        """
        Measures the mismatch between Dream and Reality.
        Surprise = -ln P(Observation | Model)
        Simplified: Z-Score Squared (Mahalanobis distance)
        """
        pred_price = prediction.get('bid', 0.0)
        real_price = reality.get('bid', 0.0)
        
        sigma = prediction.get('volatility', 0.001) + 1e-9
        
        # Error
        error = real_price - pred_price
        
        # Precision-weighted Error (Surprise)
        # Low sigma (high confidence) + High Error = HUGE Surprise
        surprise = (error ** 2) / (sigma ** 2)
        
        # Cap for numerical stability
        surprise = min(100.0, surprise)
        
        self.surprise_history.append(surprise)
        return float(surprise)
        
    def select_best_policy(self, policies: list, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates policies and returns the one with Min Expected Free Energy (G).
        """
        scores = {}
        for p in policies:
            # G = Risk + Ambiguity
            # Placeholder: assign random "Energy" based on policy type
            # In real Active Inference, this uses the Generative Model to predict future states
            
            # Bias: 'HOLD' is usually lower energy (safer) unless signal is strong
            base_energy = 0.5
            if p == "HOLD": base_energy = 0.2
            
            # Simulated calculation
            g_value = base_energy + np.random.uniform(0, 0.5)
            scores[p] = g_value
            
        best_policy = min(scores, key=scores.get)
        return {
            "selected_policy": best_policy,
            "best_G": scores[best_policy],
            "all_scores": scores
        }

    def minimize(self, action_space: list) -> str:
        """
        Legacy wrapper.
        """
        res = self.select_best_policy(action_space, {})
        return res['selected_policy']
