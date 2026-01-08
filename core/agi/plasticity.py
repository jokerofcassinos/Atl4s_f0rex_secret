
import logging
from typing import Dict, Any, List

logger = logging.getLogger("Plasticity")

class SelfModificationHeuristic:
    """
    System 10: Self-Modification Heuristic.
    Enables the AGI to rewrite its own operating parameters in real-time
    based on performance feedback.
    """
    def __init__(self):
        self.modifications_log = []
        
    def evaluate_and_adapt(self, metrics: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes performance and returns a dictionary of 'config_overrides'.
        metrics: {'win_rate': 0.45, 'drawdown': 0.02, ...}
        """
        modifications = {}
        
        win_rate = metrics.get('win_rate', 0.5)
        loss_streak = metrics.get('loss_streak', 0)
        drawdown = metrics.get('drawdown', 0.0)
        
        # 1. Protection Circuit: High Drawdown -> Slash Risk
        if drawdown > 0.05: # 5% Drawdown
            if current_config.get('risk_per_trade', 0.01) > 0.005:
                modifications['risk_per_trade'] = 0.005
                self._log_change("Reduced Risk to 0.5% due to Drawdown > 5%")
        
        # 2. Aggression Circuit: High Win Rate -> Boost Confidence Limit
        if win_rate > 0.7 and metrics.get('trades_count', 0) > 10:
             # Maybe we are too picky? Lower threshold slightly to trade more?
             # Or increase position size? 
             # Let's keep it safe: Increase position sizing (if allowed) or maintain.
             pass
             
        # 3. Plasticity: Adjusting the 'Alpha Threshold' (Confidence needed to trade)
        # If we are losing, maybe we are trading low quality setups? Increase threshold.
        current_threshold = current_config.get('alpha_threshold', 0.8)
        if win_rate < 0.4 and metrics.get('trades_count', 0) > 5:
            new_threshold = min(0.95, current_threshold + 0.05)
            if new_threshold != current_threshold:
                modifications['alpha_threshold'] = new_threshold
                self._log_change(f"Increased Alpha Threshold to {new_threshold:.2f} due to low Win Rate")
                
        # 4. Recovery: If winning again, relax threshold
        if win_rate > 0.6 and current_threshold > 0.8:
            new_threshold = max(0.8, current_threshold - 0.02)
            if new_threshold != current_threshold:
                 modifications['alpha_threshold'] = new_threshold
                 self._log_change(f"Relaxed Alpha Threshold to {new_threshold:.2f} due to recovery")

        return modifications

    def _log_change(self, message: str):
        logger.warning(f"PLASTICITY: {message}")
        self.modifications_log.append(message)
