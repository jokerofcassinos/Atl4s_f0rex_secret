"""
Neural Plasticity Core - Real-time Weight Adaptation.

Implements neural plasticity for adapting model weights in real-time
based on market regime changes and performance feedback.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("NeuralPlasticity")


@dataclass
class PlasticityState:
    """Current plasticity system state."""
    learning_rate: float
    regime: str
    adaptation_count: int
    weight_delta_magnitude: float
    stability_score: float


class NeuralPlasticityCore:
    """
    The Adaptive Brain.
    
    Implements real-time neural plasticity through:
    - Hebbian-inspired weight updates
    - Regime-dependent learning rates
    - Stability-plasticity balance
    - Continuous weight adaptation
    """
    
    def __init__(self, base_learning_rate: float = 0.1):
        self.base_lr = base_learning_rate
        self.current_lr = base_learning_rate
        self.regime = 'UNKNOWN'
        
        # Adaptive weights for different market conditions
        self.weights = {
            'trend_following': 0.5,
            'mean_reversion': 0.5,
            'momentum': 0.5,
            'volatility_sensitivity': 0.5,
            'session_awareness': 0.5,
            'liquidity_weight': 0.5,
        }
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=100)
        self.adaptation_count = 0
        
        # Stability control
        self.stability_threshold = 0.3
        self.recent_changes: deque = deque(maxlen=20)
        
        logger.info(f"NeuralPlasticityCore initialized with LR={base_learning_rate}")
    
    def adapt(self, feedback: Dict) -> PlasticityState:
        """
        Adapt weights based on performance feedback.
        
        Args:
            feedback: Dict with 'success', 'pnl', 'regime', 'factors_used'
            
        Returns:
            Current plasticity state.
        """
        success = feedback.get('success', False)
        pnl = feedback.get('pnl', 0.0)
        new_regime = feedback.get('regime', 'UNKNOWN')
        factors = feedback.get('factors_used', [])
        
        # Regime change detection
        if new_regime != self.regime:
            self._handle_regime_change(new_regime)
        
        # Calculate learning rate based on stability
        self.current_lr = self._calculate_adaptive_lr()
        
        # Update weights
        delta_magnitude = self._update_weights(success, pnl, factors)
        
        # Track performance
        self.performance_history.append({
            'time': datetime.now(timezone.utc),
            'success': success,
            'pnl': pnl,
            'regime': new_regime
        })
        
        self.adaptation_count += 1
        stability = self._calculate_stability()
        
        return PlasticityState(
            learning_rate=self.current_lr,
            regime=self.regime,
            adaptation_count=self.adaptation_count,
            weight_delta_magnitude=delta_magnitude,
            stability_score=stability
        )
    
    def _handle_regime_change(self, new_regime: str):
        """Handle detected regime change."""
        logger.info(f"REGIME CHANGE: {self.regime} -> {new_regime}")
        self.regime = new_regime
        
        # Increase learning rate for faster adaptation
        self.current_lr = min(0.3, self.base_lr * 2)
        
        # Adjust base weights for new regime
        if new_regime == 'TRENDING':
            self.weights['trend_following'] = min(0.8, self.weights['trend_following'] + 0.2)
            self.weights['mean_reversion'] = max(0.2, self.weights['mean_reversion'] - 0.1)
        elif new_regime == 'RANGING':
            self.weights['mean_reversion'] = min(0.8, self.weights['mean_reversion'] + 0.2)
            self.weights['trend_following'] = max(0.2, self.weights['trend_following'] - 0.1)
        elif new_regime == 'VOLATILE':
            self.weights['volatility_sensitivity'] = min(0.9, self.weights['volatility_sensitivity'] + 0.3)
    
    def _calculate_adaptive_lr(self) -> float:
        """Calculate learning rate based on stability and performance."""
        if len(self.performance_history) < 10:
            return self.base_lr
        
        recent = list(self.performance_history)[-10:]
        win_rate = sum(1 for r in recent if r['success']) / len(recent)
        
        # Lower LR when performing well (stability), higher when struggling (exploration)
        if win_rate > 0.6:
            return self.base_lr * 0.5  # Stable, reduce learning
        elif win_rate < 0.4:
            return min(0.3, self.base_lr * 1.5)  # Struggling, increase learning
        
        return self.base_lr
    
    def _update_weights(self, success: bool, pnl: float, factors: List[str]) -> float:
        """Update weights using Hebbian-like learning."""
        if not factors:
            return 0.0
        
        reward = 1.0 if success else -1.0
        if pnl != 0:
            reward *= min(2.0, abs(pnl) / 10)  # Scale by PnL magnitude
        
        total_delta = 0.0
        
        for factor in factors:
            if factor in self.weights:
                # Hebbian update: strengthen connections that lead to success
                delta = self.current_lr * reward
                self.weights[factor] = np.clip(
                    self.weights[factor] + delta,
                    0.1, 0.9
                )
                total_delta += abs(delta)
        
        self.recent_changes.append(total_delta)
        return total_delta
    
    def _calculate_stability(self) -> float:
        """Calculate weight stability score."""
        if len(self.recent_changes) < 5:
            return 0.5
        
        changes = list(self.recent_changes)
        avg_change = np.mean(changes)
        
        # Lower average change = higher stability
        stability = 1.0 - min(1.0, avg_change / 0.2)
        return float(stability)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weight values."""
        return self.weights.copy()
    
    def get_weight(self, factor: str) -> float:
        """Get specific weight value."""
        return self.weights.get(factor, 0.5)
    
    def manual_adjust(self, factor: str, adjustment: float):
        """Manually adjust a weight."""
        if factor in self.weights:
            self.weights[factor] = np.clip(
                self.weights[factor] + adjustment,
                0.1, 0.9
            )
            logger.info(f"MANUAL ADJUST: {factor} -> {self.weights[factor]:.2f}")
