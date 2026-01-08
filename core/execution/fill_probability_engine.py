"""
Fill Probability Engine - Real-Time Fill Probability Estimation.

Estimates fill probability at each price level for limit orders.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("FillProbability")


@dataclass
class FillEstimate:
    """Fill probability estimate."""
    price: float
    fill_probability: float
    expected_wait_seconds: int
    partial_fill_probability: float
    recommended_action: str


class FillProbabilityEngine:
    """
    The Fill Oracle.
    
    Estimates fill probability through:
    - Price level touch frequency analysis
    - Time-to-fill modeling
    - Partial fill likelihood estimation
    - Queue position modeling
    """
    
    def __init__(self):
        self.price_touches: Dict[float, List[float]] = {}  # price -> timestamps
        self.fill_outcomes: deque = deque(maxlen=200)
        self.price_granularity = 0.0001
        
        logger.info("FillProbabilityEngine initialized")
    
    def estimate(self, target_price: float, current_price: float,
                direction: str, volatility: float) -> FillEstimate:
        """
        Estimate fill probability for a limit order.
        
        Args:
            target_price: Target limit price
            current_price: Current market price
            direction: 'BUY' or 'SELL'
            volatility: Current volatility
            
        Returns:
            FillEstimate.
        """
        distance = abs(target_price - current_price)
        relative_distance = distance / current_price
        
        # Base probability from distance
        if relative_distance < 0.0005:  # Within 5 pips
            base_prob = 0.85
        elif relative_distance < 0.001:  # Within 10 pips
            base_prob = 0.6
        elif relative_distance < 0.002:  # Within 20 pips
            base_prob = 0.35
        else:
            base_prob = 0.15
        
        # Adjust by historical touches
        touch_adj = self._get_touch_adjustment(target_price)
        
        # Volatility adjustment (high vol = higher chance to reach level)
        vol_adj = min(0.2, volatility * 100)
        
        fill_prob = min(0.95, base_prob + touch_adj + vol_adj)
        
        # Expected wait time
        if fill_prob > 0.7:
            wait = 60
        elif fill_prob > 0.4:
            wait = 300
        else:
            wait = 900
        
        # Partial fill probability
        partial = 0.3 if relative_distance > 0.001 else 0.1
        
        # Recommendation
        if fill_prob < 0.3:
            action = "ADJUST_PRICE"
        elif fill_prob < 0.5:
            action = "MONITOR_CLOSELY"
        else:
            action = "LEAVE_ORDER"
        
        return FillEstimate(
            price=target_price,
            fill_probability=fill_prob,
            expected_wait_seconds=wait,
            partial_fill_probability=partial,
            recommended_action=action
        )
    
    def _get_touch_adjustment(self, price: float) -> float:
        """Get probability adjustment based on historical touches."""
        quantized = round(price / self.price_granularity) * self.price_granularity
        
        if quantized in self.price_touches:
            touch_count = len(self.price_touches[quantized])
            return min(0.2, touch_count * 0.02)
        
        return 0.0
    
    def record_price(self, price: float, timestamp: float):
        """Record a price touch."""
        quantized = round(price / self.price_granularity) * self.price_granularity
        
        if quantized not in self.price_touches:
            self.price_touches[quantized] = []
        
        self.price_touches[quantized].append(timestamp)
        
        # Keep only recent touches
        if len(self.price_touches[quantized]) > 100:
            self.price_touches[quantized] = self.price_touches[quantized][-50:]
    
    def record_fill_outcome(self, target: float, filled: bool, 
                           wait_time: float, fill_amount: float):
        """Record fill outcome for learning."""
        self.fill_outcomes.append({
            'target': target,
            'filled': filled,
            'wait_time': wait_time,
            'fill_amount': fill_amount
        })
    
    def get_optimal_limit_price(self, current: float, direction: str,
                               min_fill_prob: float = 0.6) -> float:
        """Calculate optimal limit price for desired fill probability."""
        if direction == 'BUY':
            # For buying, want lower price
            offsets = [0, -0.0002, -0.0005, -0.001, -0.002]
        else:
            # For selling, want higher price
            offsets = [0, 0.0002, 0.0005, 0.001, 0.002]
        
        for offset in offsets:
            test_price = current + offset
            estimate = self.estimate(test_price, current, direction, 0.01)
            
            if estimate.fill_probability >= min_fill_prob:
                return test_price
        
        return current  # Fall back to market price
