
import logging
from typing import Dict, Any, List
from enum import Enum
import numpy as np

logger = logging.getLogger("RegimeDetector")

class MarketRegime(Enum):
    OPTIMAL = "OPTIMAL"         # Balanced volatility, trend presence
    CHOPPY = "CHOPPY"           # Low vol, no trend, side-ways
    VOLATILE = "VOLATILE"       # High vol, extreme movement
    CRITICAL = "CRITICAL"       # System underperformance or Black Swan
    
class RegimeDetector:
    """
    detects the 'Global Market State' and adjusts system risk appetite.
    Acts as the 'Meta-Governor' for the Swarm.
    """
    
    def __init__(self):
        self.current_regime = MarketRegime.OPTIMAL
        self.last_adjustment = 0.0
        
    def detect_regime(self, market_metrics: Dict[str, float], performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Determines the current regime based on Market Physics and Bot Performance.
        
        Args:
            market_metrics: {'atr': float, 'entropy': float, 'trend_strength': float}
            performance_metrics: {'win_rate_10': float, 'drawdown': float}
            
        Returns:
            Dict containing 'regime', 'threshold_modifier', 'reason'
        """
        
        # 1. Extract Metrics
        atr = market_metrics.get('atr', 0.0010)
        entropy = market_metrics.get('entropy', 0.5)
        trend = market_metrics.get('trend_strength', 50.0)
        
        win_rate = performance_metrics.get('win_rate_10', 0.5) # Last 10 trades
        drawdown = performance_metrics.get('drawdown', 0.0)    # Current DD %
        
        # 2. Logic Tree
        
        # A. SAFETY FIRST: Critical Failure Check
        if drawdown > 0.08 or win_rate < 0.25:
            # If we are losing badly, we tighten EVERYTHING.
            return self._set_regime(MarketRegime.CRITICAL, modifier=15.0, reason=f"Drawdown {drawdown:.1%} / WR {win_rate:.0%}")
            
        # B. Volatility Check
        if entropy > 0.8:
            # Extreme Chaos
            return self._set_regime(MarketRegime.VOLATILE, modifier=5.0, reason=f"High Entropy {entropy:.2f}")
            
        # C. Chop Check
        if trend < 30.0 and entropy < 0.4:
            # Dead Market
            return self._set_regime(MarketRegime.CHOPPY, modifier=10.0, reason=f"Dead Market (Trend {trend:.0f})")
            
        # D. Optimal Conditions
        if trend > 60.0 and 0.4 <= entropy <= 0.7:
             # Let it bleed! Loosen thresholds.
             return self._set_regime(MarketRegime.OPTIMAL, modifier=-5.0, reason="Strong Trend - Aggressive Mode")
             
        # Default
        return self._set_regime(MarketRegime.OPTIMAL, modifier=0.0, reason="Balanced Market")

    def _set_regime(self, regime: MarketRegime, modifier: float, reason: str) -> Dict[str, Any]:
        self.current_regime = regime
        self.last_adjustment = modifier
        
        if regime != MarketRegime.OPTIMAL:
             logger.info(f"REGIME CHANGE: {regime.value} ({reason}). Adjusting Thresholds by {modifier:+.1f}")
             
        return {
            'regime': regime.value,
            'threshold_modifier': modifier,
            'description': reason
        }
