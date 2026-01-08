
import logging
from typing import Dict, Any, List

logger = logging.getLogger("NeuroPlasticityV2")

class NeuroPlasticityV2:
    """
    Sistema D-4: Real-Time Neuroplasticity (V2)
    Capacidade de reescrever heurísticas e podar conexões sinápticas em tempo real.
    """
    def __init__(self):
        self.synaptic_weights = {
            "trend_following": 1.0,
            "mean_reversion": 1.0,
            "breakout": 1.0,
            "scalping": 1.0
        }
        self.learning_rate = 0.05
        self.pruning_threshold = 0.2
        
    def adapt(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Reescreve pesos baseado na performance recente.
        """
        changes = {}
        
        # Heuristic Rewriter Logic
        if performance_metrics.get('win_rate', 0.5) > 0.6:
            # Strengthen successful paths
            strategy = performance_metrics.get('last_strategy', 'trend_following')
            if strategy in self.synaptic_weights:
                old_w = self.synaptic_weights[strategy]
                self.synaptic_weights[strategy] += self.learning_rate
                changes[strategy] = f"{old_w:.2f} -> {self.synaptic_weights[strategy]:.2f}"
        
        elif performance_metrics.get('drawdown', 0) > 0.02:
            # Weaken failing paths
            strategy = performance_metrics.get('last_strategy', 'trend_following')
            if strategy in self.synaptic_weights:
                old_w = self.synaptic_weights[strategy]
                self.synaptic_weights[strategy] -= self.learning_rate
                changes[strategy] = f"{old_w:.2f} -> {self.synaptic_weights[strategy]:.2f}"
                
        # Synaptic Pruning
        pruned = self._prune_synapses()
        
        return {
            "weight_adjustments": changes,
            "pruned_synapses": pruned,
            "current_weights": self.synaptic_weights.copy()
        }
        
    def _prune_synapses(self) -> List[str]:
        pruned = []
        # Create list to avoid runtime modification error
        keys = list(self.synaptic_weights.keys())
        for k in keys:
            if self.synaptic_weights[k] < self.pruning_threshold:
                pruned.append(k)
                # Don't actually delete keys to prevent crashing logic that depends on them
                # Just reset to baseline minimum or flag as inactive
                self.synaptic_weights[k] = 0.1 # Soft prune (dormant)
        return pruned
