
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
        # Real Active Inference Calculation
        # G = Risk (Divergence from Goal) + Ambiguity (Uncertainty)
        
        target_decision = context.get('consensus_decision', 'WAIT')
        confidence = context.get('consensus_confidence', 0.0) / 100.0
        entropy = context.get('entropy', 0.5)
        
        scores = {}
        for p in policies:
            g_value = 1.0 # Base Energy
            
            # Map Policy to Decision
            p_action = "WAIT" if p == "HOLD" else p
            
            # 1. Risk (Alignment with Swarm Intelligence)
            # If we act against the Collective Wisdom, Risk is High.
            if p_action == target_decision:
                # Aligned: Energy reduces as Confidence increases
                # If Conf=0.9, Cost=0.1
                g_value -= confidence 
            else:
                # Opposed: Energy increases with Confidence
                # If Conf=0.9, Cost=1.9
                g_value += confidence
                
            # 2. Ambiguity (Entropy Context)
            # In High Entropy (Chaos), Action is expensive. Holding is cheap.
            if p == "HOLD":
                # High Entropy favors HOLD -> Reduces G
                g_value -= (entropy * 0.8)
            else:
                # High Entropy penalizes Action -> Increases G
                g_value += (entropy * 0.5)
                
            # 3. Volatility Cost (Spread/Slippage Risk)
            # Volatility makes Action costlier
            vol = context.get('volatility', 0.0)
            if p != "HOLD":
                g_value += (vol * 100.0) # Penalize action in high vol
                
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
