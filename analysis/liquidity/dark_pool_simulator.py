"""
Dark Pool Simulator - Hidden Liquidity Pool Simulation.

Models institutional dark pool behavior based on historical
fill patterns and order flow analysis.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("DarkPoolSimulator")


@dataclass
class DarkPoolSignature:
    """Detected dark pool activity signature."""
    price_level: float
    estimated_size: float  # In lots
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    detection_time: datetime
    fill_pattern: str  # 'ICEBERG', 'TWAP', 'VWAP', 'BLOCK'


@dataclass
class DarkPoolReading:
    """Current dark pool analysis."""
    active_pools: List[DarkPoolSignature]
    hidden_bid_volume: float
    hidden_ask_volume: float
    net_hidden_pressure: float  # Positive = buying pressure
    institutional_presence: float  # 0-1
    
    # Predictions
    expected_support_levels: List[float]
    expected_resistance_levels: List[float]
    breakout_probability: float
    direction_bias: str


class DarkPoolSimulator:
    """
    The Hidden Depth Reader.
    
    Simulates dark pool activity through:
    - Fill pattern recognition (TWAP, VWAP, Iceberg)
    - Hidden volume estimation
    - Institutional footprint detection
    - Predictive modeling of hidden orders
    """
    
    def __init__(self):
        self.price_history: deque = deque(maxlen=1000)
        self.volume_history: deque = deque(maxlen=1000)
        self.detected_pools: List[DarkPoolSignature] = []
        
        # Detection thresholds (evolved through learning)
        self.thresholds = {
            'volume_spike': 2.5,  # Std devs above mean
            'price_absorption': 0.8,  # % of volume absorbed
            'repetitive_fill': 3,  # Min repeats for TWAP
            'iceberg_ratio': 0.3,  # Visible vs total
        }
        
        # Pattern templates for matching
        self.fill_patterns = {
            'TWAP': self._match_twap_pattern,
            'VWAP': self._match_vwap_pattern,
            'ICEBERG': self._match_iceberg_pattern,
            'BLOCK': self._match_block_pattern,
        }
        
        logger.info("DarkPoolSimulator initialized")
    
    def analyze(self, price: float, volume: float, bid: float, ask: float,
               tick_time: Optional[datetime] = None) -> DarkPoolReading:
        """
        Analyze current tick for dark pool activity.
        
        Returns:
            DarkPoolReading with detected pools and predictions.
        """
        if tick_time is None:
            tick_time = datetime.now(timezone.utc)
        
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Detect new pools
        new_pools = self._detect_pools(price, volume, tick_time)
        self.detected_pools.extend(new_pools)
        
        # Clean old pools
        self._clean_old_pools(tick_time)
        
        # Calculate hidden volumes
        hidden_bid, hidden_ask = self._estimate_hidden_volume(price)
        net_pressure = hidden_bid - hidden_ask
        
        # Institutional presence
        presence = self._calculate_institutional_presence()
        
        # Predict levels
        supports = self._predict_support_levels(price)
        resistances = self._predict_resistance_levels(price)
        
        # Breakout analysis
        breakout_prob, direction = self._analyze_breakout_potential(
            net_pressure, supports, resistances, price
        )
        
        return DarkPoolReading(
            active_pools=self.detected_pools[-10:],  # Last 10
            hidden_bid_volume=hidden_bid,
            hidden_ask_volume=hidden_ask,
            net_hidden_pressure=net_pressure,
            institutional_presence=presence,
            expected_support_levels=supports,
            expected_resistance_levels=resistances,
            breakout_probability=breakout_prob,
            direction_bias=direction
        )
    
    def _detect_pools(self, price: float, volume: float,
                     tick_time: datetime) -> List[DarkPoolSignature]:
        """Detect new dark pool activity."""
        detected = []
        
        if len(self.volume_history) < 50:
            return detected
        
        # Check for volume anomalies
        vol_array = np.array(list(self.volume_history))
        mean_vol = np.mean(vol_array[-50:])
        std_vol = np.std(vol_array[-50:])
        
        # Volume spike detection
        if volume > mean_vol + std_vol * self.thresholds['volume_spike']:
            # Determine direction based on price movement
            if len(self.price_history) > 1:
                price_change = price - self.price_history[-2]
                direction = 'BUY' if price_change > 0 else 'SELL'
            else:
                direction = 'NEUTRAL'
            
            # Match fill patterns
            for pattern_name, matcher in self.fill_patterns.items():
                confidence = matcher(price, volume)
                if confidence > 0.6:
                    detected.append(DarkPoolSignature(
                        price_level=price,
                        estimated_size=volume * 2,  # Hidden typically 2x visible
                        direction=direction,
                        confidence=confidence,
                        detection_time=tick_time,
                        fill_pattern=pattern_name
                    ))
                    break
        
        return detected
    
    def _match_twap_pattern(self, price: float, volume: float) -> float:
        """Detect Time-Weighted Average Price pattern."""
        if len(self.volume_history) < 10:
            return 0.0
        
        recent_vols = list(self.volume_history)[-10:]
        
        # TWAP shows consistent volume over time
        cv = np.std(recent_vols) / (np.mean(recent_vols) + 1e-8)
        
        if cv < 0.3:  # Low coefficient of variation
            return 0.7
        return 0.0
    
    def _match_vwap_pattern(self, price: float, volume: float) -> float:
        """Detect Volume-Weighted Average Price pattern."""
        if len(self.volume_history) < 20:
            return 0.0
        
        # VWAP tracks price with volume
        prices = list(self.price_history)[-20:]
        volumes = list(self.volume_history)[-20:]
        
        vwap = np.sum(np.array(prices) * np.array(volumes)) / np.sum(volumes)
        
        # If current price is close to VWAP
        if abs(price - vwap) / price < 0.001:
            return 0.65
        return 0.0
    
    def _match_iceberg_pattern(self, price: float, volume: float) -> float:
        """Detect iceberg order pattern."""
        if len(self.volume_history) < 5:
            return 0.0
        
        recent_vols = list(self.volume_history)[-5:]
        
        # Iceberg shows similar-sized fills
        vol_std = np.std(recent_vols)
        vol_mean = np.mean(recent_vols)
        
        if vol_std / (vol_mean + 1e-8) < 0.2:  # Very consistent sizes
            # Check for price stability (absorbed volume)
            prices = list(self.price_history)[-5:]
            price_range = max(prices) - min(prices)
            
            if price_range / price < 0.0005:  # Price not moving despite volume
                return 0.8
        return 0.0
    
    def _match_block_pattern(self, price: float, volume: float) -> float:
        """Detect block trade pattern."""
        if len(self.volume_history) < 2:
            return 0.0
        
        mean_vol = np.mean(list(self.volume_history)[-50:]) if len(self.volume_history) >= 50 else volume
        
        # Block trades are 3x+ normal volume
        if volume > mean_vol * 3:
            return 0.75
        return 0.0
    
    def _clean_old_pools(self, current_time: datetime, max_age_minutes: int = 30):
        """Remove old pool detections."""
        from datetime import timedelta
        cutoff = current_time - timedelta(minutes=max_age_minutes)
        self.detected_pools = [
            p for p in self.detected_pools
            if p.detection_time > cutoff
        ]
    
    def _estimate_hidden_volume(self, current_price: float) -> Tuple[float, float]:
        """Estimate hidden bid and ask volume."""
        buy_pools = [p for p in self.detected_pools if p.direction == 'BUY']
        sell_pools = [p for p in self.detected_pools if p.direction == 'SELL']
        
        hidden_bid = sum(p.estimated_size * p.confidence for p in buy_pools)
        hidden_ask = sum(p.estimated_size * p.confidence for p in sell_pools)
        
        return hidden_bid, hidden_ask
    
    def _calculate_institutional_presence(self) -> float:
        """Calculate institutional presence based on detected pools."""
        if not self.detected_pools:
            return 0.0
        
        # Weight by confidence and recency
        total_weight = sum(p.confidence for p in self.detected_pools)
        
        # Normalize to 0-1
        presence = min(1.0, total_weight / 10)
        
        return presence
    
    def _predict_support_levels(self, current_price: float) -> List[float]:
        """Predict support levels from dark pool activity."""
        buy_pools = [
            p for p in self.detected_pools
            if p.direction == 'BUY' and p.price_level < current_price
        ]
        
        if not buy_pools:
            return []
        
        # Group by price level
        levels = {}
        for pool in buy_pools:
            key = round(pool.price_level, 4)
            if key not in levels:
                levels[key] = 0
            levels[key] += pool.estimated_size * pool.confidence
        
        # Return top 3 by strength
        sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
        return [level for level, _ in sorted_levels[:3]]
    
    def _predict_resistance_levels(self, current_price: float) -> List[float]:
        """Predict resistance levels from dark pool activity."""
        sell_pools = [
            p for p in self.detected_pools
            if p.direction == 'SELL' and p.price_level > current_price
        ]
        
        if not sell_pools:
            return []
        
        levels = {}
        for pool in sell_pools:
            key = round(pool.price_level, 4)
            if key not in levels:
                levels[key] = 0
            levels[key] += pool.estimated_size * pool.confidence
        
        sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
        return [level for level, _ in sorted_levels[:3]]
    
    def _analyze_breakout_potential(self, net_pressure: float,
                                   supports: List[float],
                                   resistances: List[float],
                                   current_price: float) -> Tuple[float, str]:
        """Analyze breakout probability and direction."""
        # Strong net buying pressure suggests upward breakout
        if net_pressure > 100:
            direction = 'BULLISH'
            prob = min(0.8, 0.5 + net_pressure / 500)
        elif net_pressure < -100:
            direction = 'BEARISH'
            prob = min(0.8, 0.5 + abs(net_pressure) / 500)
        else:
            direction = 'NEUTRAL'
            prob = 0.3
        
        # Adjust by proximity to levels
        if resistances and current_price > resistances[0] * 0.998:
            prob *= 1.2
        if supports and current_price < supports[0] * 1.002:
            prob *= 1.2
        
        return float(np.clip(prob, 0, 0.9)), direction
    
    def get_hidden_order_estimate(self, price_level: float,
                                 direction: str) -> float:
        """Estimate hidden orders at a price level."""
        pools = [
            p for p in self.detected_pools
            if abs(p.price_level - price_level) < price_level * 0.001
            and p.direction == direction
        ]
        
        return sum(p.estimated_size for p in pools)
