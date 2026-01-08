"""
Spread Entropy Analyzer - Entropy-Based Spread Prediction.

Analyzes spread entropy for optimal execution timing
and spread condition prediction.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("SpreadEntropy")


@dataclass
class SpreadState:
    """Current spread state analysis."""
    current_spread: float
    entropy: float  # 0-1, higher = more unpredictable
    regime: str  # 'TIGHT', 'NORMAL', 'WIDE', 'VOLATILE'
    predicted_direction: str  # 'WIDENING', 'TIGHTENING', 'STABLE'
    optimal_execution_window: bool
    confidence: float


class SpreadEntropyAnalyzer:
    """
    The Spread Oracle.
    
    Analyzes spread entropy through:
    - Statistical entropy calculation
    - Regime detection
    - Spread direction prediction
    - Optimal execution window identification
    """
    
    def __init__(self):
        self.spread_history: deque = deque(maxlen=500)
        self.entropy_history: deque = deque(maxlen=100)
        
        logger.info("SpreadEntropyAnalyzer initialized")
    
    def analyze(self, bid: float, ask: float) -> SpreadState:
        """Analyze current spread state."""
        spread = ask - bid
        self.spread_history.append(spread)
        
        # Calculate entropy
        entropy = self._calculate_entropy()
        self.entropy_history.append(entropy)
        
        # Determine regime
        regime = self._determine_regime(spread)
        
        # Predict direction
        direction = self._predict_direction()
        
        # Check optimal window
        optimal = self._is_optimal_window(spread, entropy)
        
        # Confidence
        confidence = 0.8 if len(self.spread_history) > 50 else 0.5
        
        return SpreadState(
            current_spread=spread,
            entropy=entropy,
            regime=regime,
            predicted_direction=direction,
            optimal_execution_window=optimal,
            confidence=confidence
        )
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of spread distribution."""
        if len(self.spread_history) < 20:
            return 0.5
        
        spreads = np.array(list(self.spread_history))
        
        # Normalize to probabilities
        hist, _ = np.histogram(spreads, bins=10, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / np.sum(hist)
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize to 0-1
        max_entropy = np.log2(10)
        normalized = entropy / max_entropy
        
        return float(np.clip(normalized, 0, 1))
    
    def _determine_regime(self, current_spread: float) -> str:
        """Determine current spread regime."""
        if len(self.spread_history) < 20:
            return 'NORMAL'
        
        spreads = np.array(list(self.spread_history))
        mean = np.mean(spreads)
        std = np.std(spreads)
        
        if current_spread < mean - std:
            return 'TIGHT'
        elif current_spread > mean + 2 * std:
            return 'VOLATILE'
        elif current_spread > mean + std:
            return 'WIDE'
        return 'NORMAL'
    
    def _predict_direction(self) -> str:
        """Predict spread direction."""
        if len(self.spread_history) < 10:
            return 'STABLE'
        
        recent = list(self.spread_history)[-10:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.00001:
            return 'WIDENING'
        elif trend < -0.00001:
            return 'TIGHTENING'
        return 'STABLE'
    
    def _is_optimal_window(self, spread: float, entropy: float) -> bool:
        """Check if current conditions are optimal for execution."""
        if len(self.spread_history) < 20:
            return True
        
        spreads = np.array(list(self.spread_history))
        percentile = np.percentile(spreads, 30)
        
        # Optimal: low spread + low entropy
        return spread <= percentile and entropy < 0.6
    
    def get_spread_score(self) -> float:
        """Get execution favorability score (0-1, higher = better)."""
        state = self.analyze(0, 0.001)  # Dummy call for state
        
        if state.regime == 'TIGHT':
            base = 0.9
        elif state.regime == 'NORMAL':
            base = 0.7
        elif state.regime == 'WIDE':
            base = 0.4
        else:
            base = 0.2
        
        # Adjust by entropy
        score = base * (1 - state.entropy * 0.3)
        
        return float(np.clip(score, 0, 1))
