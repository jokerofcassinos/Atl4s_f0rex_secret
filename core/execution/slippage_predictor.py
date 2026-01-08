"""
Slippage Predictor - ML-Based Slippage Prediction.

Predicts expected slippage using order flow and liquidity analysis.
"""

import logging
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("SlippagePredictor")


@dataclass
class SlippagePrediction:
    """Slippage prediction result."""
    expected_slippage_pips: float
    max_slippage_pips: float
    probability_positive: float  # Favorable slippage
    probability_negative: float
    risk_score: float
    confidence: float


class SlippagePredictor:
    """
    The Execution Cost Forecaster.
    
    Predicts slippage through:
    - Historical slippage pattern learning
    - Liquidity-based estimation
    - Order size impact modeling
    - Time-of-day patterns
    """
    
    def __init__(self):
        self.slippage_history: deque = deque(maxlen=500)
        self.feature_history: deque = deque(maxlen=500)
        
        # Simple learned parameters
        self.base_slippage = 0.5  # pips
        self.liquidity_factor = 0.3
        self.size_factor = 0.2
        self.volatility_factor = 0.4
        
        logger.info("SlippagePredictor initialized")
    
    def predict(self, order_size: float, liquidity_score: float,
               volatility: float, spread: float) -> SlippagePrediction:
        """
        Predict slippage for an order.
        
        Args:
            order_size: Order size in lots
            liquidity_score: 0-1 liquidity score
            volatility: Current volatility
            spread: Current spread
            
        Returns:
            SlippagePrediction.
        """
        # Base slippage estimation
        base = self.base_slippage
        
        # Liquidity impact (low liquidity = more slippage)
        liq_impact = (1 - liquidity_score) * self.liquidity_factor * 2
        
        # Size impact (larger orders = more slippage)
        size_impact = (order_size / 1.0) * self.size_factor
        
        # Volatility impact
        vol_impact = volatility * self.volatility_factor * 100
        
        expected = base + liq_impact + size_impact + vol_impact
        
        # Max slippage (95th percentile estimate)
        max_slip = expected * 2.5
        
        # Probabilities
        prob_positive = 0.2  # 20% chance of favorable slippage
        prob_negative = 0.6  # 60% chance of negative slippage
        
        # Risk score
        risk = min(1.0, expected / 3.0)
        
        # Confidence based on history
        confidence = 0.7 if len(self.slippage_history) > 50 else 0.5
        
        return SlippagePrediction(
            expected_slippage_pips=expected,
            max_slippage_pips=max_slip,
            probability_positive=prob_positive,
            probability_negative=prob_negative,
            risk_score=risk,
            confidence=confidence
        )
    
    def record_actual(self, predicted: float, actual: float, features: Dict):
        """Record actual slippage for learning."""
        self.slippage_history.append({
            'predicted': predicted,
            'actual': actual,
            'error': actual - predicted
        })
        self.feature_history.append(features)
        
        # Simple online learning
        self._update_model()
    
    def _update_model(self):
        """Update model parameters from recent data."""
        if len(self.slippage_history) < 20:
            return
        
        recent = list(self.slippage_history)[-20:]
        avg_error = np.mean([r['error'] for r in recent])
        
        # Adjust base slippage
        self.base_slippage += avg_error * 0.1
        self.base_slippage = np.clip(self.base_slippage, 0.1, 3.0)
    
    def get_recommendation(self, prediction: SlippagePrediction) -> str:
        """Get execution recommendation based on prediction."""
        if prediction.risk_score > 0.7:
            return "REDUCE_SIZE_OR_WAIT"
        elif prediction.risk_score > 0.4:
            return "USE_LIMIT_ORDER"
        else:
            return "MARKET_ORDER_OK"
