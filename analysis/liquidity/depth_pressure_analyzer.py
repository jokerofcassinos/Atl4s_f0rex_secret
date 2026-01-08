"""
Depth Pressure Analyzer - Order Book Imbalance and Absorption Analysis.

Analyzes bid/ask pressure through simulated order book reconstruction
and absorption pattern detection.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("DepthPressure")


@dataclass
class PressureLevel:
    """Pressure at a price level."""
    price: float
    bid_pressure: float
    ask_pressure: float
    net_pressure: float  # Positive = buying pressure
    absorption_rate: float
    stability: float  # How stable is this level


@dataclass
class DepthAnalysis:
    """Order book depth analysis result."""
    levels: List[PressureLevel]
    total_bid_pressure: float
    total_ask_pressure: float
    imbalance_ratio: float  # >1 = more bids
    
    # Key levels
    strongest_support: Optional[float]
    strongest_resistance: Optional[float]
    
    # Predictions
    breakout_direction: str
    breakout_probability: float
    absorption_detected: bool


class DepthPressureAnalyzer:
    """
    The Pressure Reader.
    
    Analyzes order book pressure through:
    - Simulated order book reconstruction from price action
    - Bid/ask imbalance tracking
    - Absorption pattern detection
    - Breakout prediction from depth analysis
    """
    
    def __init__(self, levels: int = 10):
        self.num_levels = levels
        self.pressure_memory: deque = deque(maxlen=100)
        self.imbalance_history: deque = deque(maxlen=50)
        
        # Simulated order book state
        self.simulated_bids: Dict[float, float] = {}
        self.simulated_asks: Dict[float, float] = {}
        
        # Parameters
        self.level_spacing = 0.0001  # 1 pip for Forex
        self.decay_rate = 0.95
        
        logger.info("DepthPressureAnalyzer initialized")
    
    def analyze(self, bid: float, ask: float, volume: float,
               last_price: Optional[float] = None) -> DepthAnalysis:
        """
        Analyze order book depth and pressure.
        
        Args:
            bid: Current bid price
            ask: Current ask price
            volume: Current tick volume
            last_price: Previous close price
            
        Returns:
            DepthAnalysis with pressure levels and predictions.
        """
        mid_price = (bid + ask) / 2
        
        # Update simulated order book
        self._update_simulated_book(bid, ask, volume, last_price)
        
        # Calculate pressure at each level
        levels = self._calculate_pressure_levels(mid_price)
        
        # Aggregate metrics
        total_bid = sum(l.bid_pressure for l in levels if l.bid_pressure > 0)
        total_ask = sum(l.ask_pressure for l in levels if l.ask_pressure > 0)
        
        imbalance = total_bid / max(total_ask, 0.01)
        self.imbalance_history.append(imbalance)
        
        # Find strongest levels
        supports = [l for l in levels if l.net_pressure > 0]
        supports.sort(key=lambda l: l.bid_pressure, reverse=True)
        strongest_support = supports[0].price if supports else None
        
        resistances = [l for l in levels if l.net_pressure < 0]
        resistances.sort(key=lambda l: l.ask_pressure, reverse=True)
        strongest_resistance = resistances[0].price if resistances else None
        
        # Detect absorption
        absorption = self._detect_absorption(levels)
        
        # Predict breakout
        direction, probability = self._predict_breakout(imbalance, levels)
        
        return DepthAnalysis(
            levels=levels,
            total_bid_pressure=total_bid,
            total_ask_pressure=total_ask,
            imbalance_ratio=imbalance,
            strongest_support=strongest_support,
            strongest_resistance=strongest_resistance,
            breakout_direction=direction,
            breakout_probability=probability,
            absorption_detected=absorption
        )
    
    def _update_simulated_book(self, bid: float, ask: float, volume: float,
                               last_price: Optional[float]):
        """Update simulated order book based on price action."""
        # Decay existing orders
        for price in list(self.simulated_bids.keys()):
            self.simulated_bids[price] *= self.decay_rate
            if self.simulated_bids[price] < 0.01:
                del self.simulated_bids[price]
        
        for price in list(self.simulated_asks.keys()):
            self.simulated_asks[price] *= self.decay_rate
            if self.simulated_asks[price] < 0.01:
                del self.simulated_asks[price]
        
        # Add new orders based on price movement
        if last_price:
            price_change = (bid + ask) / 2 - last_price
            
            if price_change > 0:
                # Price went up - aggressive buyers hit asks
                # Estimate remaining bid support
                support_level = bid - self.level_spacing
                self.simulated_bids[support_level] = volume * 0.7
            else:
                # Price went down - aggressive sellers hit bids
                # Estimate remaining ask resistance
                resistance_level = ask + self.level_spacing
                self.simulated_asks[resistance_level] = volume * 0.7
        
        # Always add some base liquidity at current levels
        self.simulated_bids[bid] = self.simulated_bids.get(bid, 0) + volume * 0.3
        self.simulated_asks[ask] = self.simulated_asks.get(ask, 0) + volume * 0.3
    
    def _calculate_pressure_levels(self, mid_price: float) -> List[PressureLevel]:
        """Calculate pressure at each level around mid price."""
        levels = []
        
        for i in range(-self.num_levels, self.num_levels + 1):
            level_price = mid_price + i * self.level_spacing * 5  # 5 pip spacing
            
            # Find nearby simulated orders
            bid_pressure = sum(
                v for p, v in self.simulated_bids.items()
                if abs(p - level_price) <= self.level_spacing * 2.5
            )
            
            ask_pressure = sum(
                v for p, v in self.simulated_asks.items()
                if abs(p - level_price) <= self.level_spacing * 2.5
            )
            
            net = bid_pressure - ask_pressure
            
            # Calculate absorption rate (how much volume is absorbed at this level)
            total = bid_pressure + ask_pressure
            absorption = min(bid_pressure, ask_pressure) / max(total, 0.01)
            
            # Stability based on historical touches
            stability = self._calculate_level_stability(level_price)
            
            levels.append(PressureLevel(
                price=level_price,
                bid_pressure=bid_pressure,
                ask_pressure=ask_pressure,
                net_pressure=net,
                absorption_rate=absorption,
                stability=stability
            ))
        
        return levels
    
    def _calculate_level_stability(self, price: float) -> float:
        """Calculate how stable a price level is."""
        # Based on how often price has touched this level
        touches = sum(
            1 for entry in self.pressure_memory
            if 'mid_price' in entry and abs(entry['mid_price'] - price) < self.level_spacing * 2
        )
        
        return min(1.0, touches / 10)
    
    def _detect_absorption(self, levels: List[PressureLevel]) -> bool:
        """Detect if absorption is occurring."""
        high_absorption = [l for l in levels if l.absorption_rate > 0.4]
        return len(high_absorption) >= 2
    
    def _predict_breakout(self, imbalance: float, 
                         levels: List[PressureLevel]) -> Tuple[str, float]:
        """Predict breakout direction and probability."""
        # Use imbalance history trend
        if len(self.imbalance_history) < 5:
            return 'NEUTRAL', 0.5
        
        imbalance_trend = np.polyfit(
            range(5), 
            list(self.imbalance_history)[-5:], 
            1
        )[0]
        
        # Strong positive trend = bullish breakout
        if imbalance_trend > 0.1 and imbalance > 1.2:
            return 'BULLISH', min(0.8, 0.5 + imbalance_trend)
        elif imbalance_trend < -0.1 and imbalance < 0.8:
            return 'BEARISH', min(0.8, 0.5 + abs(imbalance_trend))
        else:
            return 'NEUTRAL', 0.5
    
    def update_from_tick(self, bid: float, ask: float, volume: float):
        """Simple update without full analysis."""
        mid = (bid + ask) / 2
        self.pressure_memory.append({
            'mid_price': mid,
            'volume': volume,
            'spread': ask - bid,
            'time': datetime.now(timezone.utc)
        })
    
    def get_quick_imbalance(self) -> float:
        """Get current bid/ask imbalance quickly."""
        if not self.imbalance_history:
            return 1.0
        return self.imbalance_history[-1]
    
    def get_pressure_at_price(self, price: float) -> Tuple[float, float]:
        """Get bid and ask pressure at a specific price."""
        bid_p = sum(
            v for p, v in self.simulated_bids.items()
            if abs(p - price) <= self.level_spacing * 2
        )
        ask_p = sum(
            v for p, v in self.simulated_asks.items()
            if abs(p - price) <= self.level_spacing * 2
        )
        return bid_p, ask_p
