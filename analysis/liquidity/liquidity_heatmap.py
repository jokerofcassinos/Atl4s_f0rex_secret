"""
Liquidity Heatmap - 3D Visualization of Liquidity Distribution.

Implements holographic memory encoding for multi-dimensional
liquidity analysis across price levels, time, and volume.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("LiquidityHeatmap")


@dataclass
class LiquidityZone:
    """A zone of concentrated liquidity."""
    price_level: float
    strength: float  # 0-1 intensity
    zone_type: str  # 'SUPPORT', 'RESISTANCE', 'NEUTRAL'
    age_minutes: int
    touch_count: int  # How many times price interacted
    estimated_volume: float
    decay_rate: float


@dataclass
class HeatmapReading:
    """Current liquidity heatmap state."""
    zones: List[LiquidityZone]
    overall_liquidity: float  # 0-100
    bid_liquidity: float
    ask_liquidity: float
    imbalance_ratio: float  # >1 = more bids, <1 = more asks
    
    # Holographic encoding
    encoded_state: np.ndarray
    pattern_similarity: float
    
    # Recommendations
    optimal_entry_zones: List[float]
    avoid_zones: List[float]


class LiquidityHeatmap:
    """
    The Liquidity Eye.
    
    Creates 3D visualization of liquidity through:
    - Holographic memory encoding of price levels
    - Temporal decay modeling
    - Volume-weighted zone identification
    - Pattern similarity matching
    """
    
    def __init__(self, price_granularity: float = 0.0001):
        self.price_granularity = price_granularity  # 1 pip for Forex
        self.zones: Dict[float, LiquidityZone] = {}
        self.volume_memory: Dict[float, List[float]] = defaultdict(list)
        
        # Holographic encoding parameters
        self.encoding_dim = 64
        self.holographic_plate: np.ndarray = np.zeros((100, self.encoding_dim))
        self.pattern_memory: List[np.ndarray] = []
        
        # Decay parameters
        self.base_decay_rate = 0.98
        self.touch_reinforcement = 1.1
        
        logger.info("LiquidityHeatmap initialized with holographic encoding")
    
    def update(self, price: float, volume: float, bid: float, ask: float,
              tick_time: Optional[datetime] = None) -> HeatmapReading:
        """
        Update heatmap with new tick data.
        
        Args:
            price: Current price
            volume: Tick volume
            bid: Current bid
            ask: Current ask
            tick_time: Timestamp
            
        Returns:
            HeatmapReading with current liquidity analysis.
        """
        if tick_time is None:
            tick_time = datetime.now(timezone.utc)
        
        # Quantize price to granularity
        quantized_price = self._quantize_price(price)
        
        # Update volume memory
        self.volume_memory[quantized_price].append(volume)
        if len(self.volume_memory[quantized_price]) > 100:
            self.volume_memory[quantized_price] = self.volume_memory[quantized_price][-50:]
        
        # Update or create zone
        if quantized_price in self.zones:
            zone = self.zones[quantized_price]
            zone.touch_count += 1
            zone.strength = min(1.0, zone.strength * self.touch_reinforcement)
            zone.estimated_volume = np.mean(self.volume_memory[quantized_price])
        else:
            zone = LiquidityZone(
                price_level=quantized_price,
                strength=0.3,
                zone_type='NEUTRAL',
                age_minutes=0,
                touch_count=1,
                estimated_volume=volume,
                decay_rate=self.base_decay_rate
            )
            self.zones[quantized_price] = zone
        
        # Apply decay to all zones
        self._apply_decay()
        
        # Classify zones
        self._classify_zones(price)
        
        # Calculate metrics
        spread = ask - bid
        bid_liq = self._estimate_bid_liquidity(price)
        ask_liq = self._estimate_ask_liquidity(price)
        imbalance = bid_liq / max(ask_liq, 0.01)
        
        # Holographic encoding
        encoded = self._holographic_encode(price)
        similarity = self._pattern_match(encoded)
        
        # Get active zones
        active_zones = sorted(
            [z for z in self.zones.values() if z.strength > 0.2],
            key=lambda z: z.strength,
            reverse=True
        )[:20]
        
        # Find optimal entry and avoid zones
        optimal = self._find_optimal_entries(price, active_zones)
        avoid = self._find_avoid_zones(price, active_zones)
        
        return HeatmapReading(
            zones=active_zones,
            overall_liquidity=min(100, (bid_liq + ask_liq) * 50),
            bid_liquidity=bid_liq,
            ask_liquidity=ask_liq,
            imbalance_ratio=imbalance,
            encoded_state=encoded,
            pattern_similarity=similarity,
            optimal_entry_zones=optimal,
            avoid_zones=avoid
        )
    
    def _quantize_price(self, price: float) -> float:
        """Quantize price to granularity."""
        return round(price / self.price_granularity) * self.price_granularity
    
    def _apply_decay(self):
        """Apply temporal decay to all zones."""
        to_remove = []
        
        for price, zone in self.zones.items():
            zone.strength *= zone.decay_rate
            zone.age_minutes += 1
            
            if zone.strength < 0.05:
                to_remove.append(price)
        
        for price in to_remove:
            del self.zones[price]
    
    def _classify_zones(self, current_price: float):
        """Classify zones as support, resistance, or neutral."""
        for zone in self.zones.values():
            if zone.price_level < current_price:
                if zone.touch_count >= 2 and zone.strength > 0.4:
                    zone.zone_type = 'SUPPORT'
                else:
                    zone.zone_type = 'NEUTRAL'
            elif zone.price_level > current_price:
                if zone.touch_count >= 2 and zone.strength > 0.4:
                    zone.zone_type = 'RESISTANCE'
                else:
                    zone.zone_type = 'NEUTRAL'
    
    def _estimate_bid_liquidity(self, current_price: float) -> float:
        """Estimate bid-side liquidity."""
        bid_zones = [
            z for z in self.zones.values()
            if z.price_level < current_price and z.zone_type == 'SUPPORT'
        ]
        
        if not bid_zones:
            return 0.5
        
        return np.mean([z.strength for z in bid_zones])
    
    def _estimate_ask_liquidity(self, current_price: float) -> float:
        """Estimate ask-side liquidity."""
        ask_zones = [
            z for z in self.zones.values()
            if z.price_level > current_price and z.zone_type == 'RESISTANCE'
        ]
        
        if not ask_zones:
            return 0.5
        
        return np.mean([z.strength for z in ask_zones])
    
    def _holographic_encode(self, current_price: float) -> np.ndarray:
        """
        Holographic encoding of current liquidity state.
        
        Creates a distributed representation that captures the
        essence of the current liquidity distribution.
        """
        # Create feature vector
        features = np.zeros(self.encoding_dim)
        
        # Encode zone distribution
        for i, zone in enumerate(sorted(self.zones.values(), key=lambda z: z.price_level)):
            if i >= self.encoding_dim // 2:
                break
            
            # Distance-weighted encoding
            distance = abs(zone.price_level - current_price)
            weight = zone.strength * np.exp(-distance * 100)
            
            idx = i * 2
            features[idx] = weight
            features[idx + 1] = 1 if zone.zone_type == 'SUPPORT' else -1
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        
        # Update holographic plate
        self._update_holographic_plate(features)
        
        return features
    
    def _update_holographic_plate(self, features: np.ndarray):
        """Update holographic memory plate with new encoding."""
        # Shift plate down and add new encoding at top
        self.holographic_plate = np.roll(self.holographic_plate, 1, axis=0)
        self.holographic_plate[0] = features
    
    def _pattern_match(self, encoded: np.ndarray) -> float:
        """Match current pattern against historical patterns."""
        if len(self.pattern_memory) < 10:
            self.pattern_memory.append(encoded.copy())
            return 0.5
        
        # Cosine similarity with recent patterns
        similarities = []
        for pattern in self.pattern_memory[-20:]:
            sim = np.dot(encoded, pattern) / (
                np.linalg.norm(encoded) * np.linalg.norm(pattern) + 1e-8
            )
            similarities.append(sim)
        
        # Store current pattern
        self.pattern_memory.append(encoded.copy())
        if len(self.pattern_memory) > 100:
            self.pattern_memory = self.pattern_memory[-50:]
        
        return float(np.max(similarities))
    
    def _find_optimal_entries(self, current_price: float,
                             zones: List[LiquidityZone]) -> List[float]:
        """Find optimal entry price levels."""
        optimal = []
        
        # Near strong support for longs
        supports = [z for z in zones if z.zone_type == 'SUPPORT']
        for s in supports[:3]:
            if s.strength > 0.5 and abs(s.price_level - current_price) < current_price * 0.002:
                optimal.append(s.price_level)
        
        # Near strong resistance for shorts
        resistances = [z for z in zones if z.zone_type == 'RESISTANCE']
        for r in resistances[:3]:
            if r.strength > 0.5 and abs(r.price_level - current_price) < current_price * 0.002:
                optimal.append(r.price_level)
        
        return optimal
    
    def _find_avoid_zones(self, current_price: float,
                         zones: List[LiquidityZone]) -> List[float]:
        """Find price levels to avoid (low liquidity)."""
        avoid = []
        
        # Price levels with low liquidity between current and next zone
        sorted_zones = sorted(zones, key=lambda z: z.price_level)
        
        for i in range(len(sorted_zones) - 1):
            gap = sorted_zones[i + 1].price_level - sorted_zones[i].price_level
            if gap > current_price * 0.003:  # Large gap
                mid_point = (sorted_zones[i].price_level + sorted_zones[i + 1].price_level) / 2
                avoid.append(mid_point)
        
        return avoid
    
    def get_liquidity_score(self, price: float) -> float:
        """Get liquidity score at a specific price level."""
        quantized = self._quantize_price(price)
        
        if quantized in self.zones:
            return self.zones[quantized].strength
        
        # Check nearby zones
        nearby = [
            z.strength for p, z in self.zones.items()
            if abs(p - quantized) <= self.price_granularity * 5
        ]
        
        return np.mean(nearby) if nearby else 0.0
    
    def clear_old_zones(self, max_age_minutes: int = 120):
        """Clear zones older than max_age."""
        to_remove = [
            p for p, z in self.zones.items()
            if z.age_minutes > max_age_minutes
        ]
        for p in to_remove:
            del self.zones[p]
