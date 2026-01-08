"""
Iceberg Detector - Hidden Iceberg Order Detection.

Detects iceberg orders through volume anomaly detection
and repetitive fill pattern recognition.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("IcebergDetector")


@dataclass
class IcebergSignature:
    """Detected iceberg order signature."""
    price_level: float
    estimated_total_size: float
    visible_size: float
    hidden_ratio: float  # Hidden/Total
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    refill_count: int  # Times the visible portion was refilled
    first_detected: datetime
    last_updated: datetime


@dataclass
class IcebergAnalysis:
    """Iceberg detection analysis result."""
    detected_icebergs: List[IcebergSignature]
    total_hidden_bid_volume: float
    total_hidden_ask_volume: float
    iceberg_presence: float  # 0-1 probability of icebergs
    
    # Impact predictions
    price_support_strength: float
    price_resistance_strength: float
    expected_absorption_zones: List[float]


class IcebergDetector:
    """
    The Iceberg Hunter.
    
    Detects hidden iceberg orders through:
    - Repetitive fill pattern recognition
    - Volume anomaly at price levels
    - Price absorption analysis
    - Statistical pattern matching
    """
    
    def __init__(self):
        self.fill_history: deque = deque(maxlen=500)
        self.detected_icebergs: List[IcebergSignature] = []
        self.price_level_fills: Dict[float, List[Dict]] = {}
        
        # Detection parameters
        self.min_refills = 3  # Minimum refills to confirm iceberg
        self.size_variance_threshold = 0.3  # Max variance in fill sizes
        self.price_granularity = 0.0001  # 1 pip
        
        logger.info("IcebergDetector initialized")
    
    def detect(self, price: float, volume: float, bid: float, ask: float,
              tick_time: Optional[datetime] = None) -> IcebergAnalysis:
        """
        Detect iceberg orders from current tick.
        
        Returns:
            IcebergAnalysis with detected icebergs and predictions.
        """
        if tick_time is None:
            tick_time = datetime.now(timezone.utc)
        
        # Record fill
        fill = {
            'price': price,
            'volume': volume,
            'bid': bid,
            'ask': ask,
            'time': tick_time
        }
        self.fill_history.append(fill)
        
        # Update price level fills
        quantized = self._quantize_price(price)
        if quantized not in self.price_level_fills:
            self.price_level_fills[quantized] = []
        self.price_level_fills[quantized].append(fill)
        
        # Keep only recent fills per level
        if len(self.price_level_fills[quantized]) > 50:
            self.price_level_fills[quantized] = self.price_level_fills[quantized][-25:]
        
        # Analyze for iceberg patterns
        new_icebergs = self._analyze_iceberg_patterns(tick_time)
        
        # Update existing icebergs
        self._update_existing_icebergs(price, volume, tick_time)
        
        # Add new detections
        for iceberg in new_icebergs:
            self._add_or_update_iceberg(iceberg)
        
        # Clean old icebergs
        self._clean_old_icebergs(tick_time)
        
        # Calculate volumes
        bid_vol = sum(i.estimated_total_size for i in self.detected_icebergs if i.direction == 'BUY')
        ask_vol = sum(i.estimated_total_size for i in self.detected_icebergs if i.direction == 'SELL')
        
        # Calculate presence
        presence = self._calculate_iceberg_presence()
        
        # Calculate support/resistance strength
        support = self._calculate_iceberg_support(price)
        resistance = self._calculate_iceberg_resistance(price)
        
        # Expected absorption zones
        absorption = self._get_absorption_zones(price)
        
        return IcebergAnalysis(
            detected_icebergs=self.detected_icebergs.copy(),
            total_hidden_bid_volume=bid_vol,
            total_hidden_ask_volume=ask_vol,
            iceberg_presence=presence,
            price_support_strength=support,
            price_resistance_strength=resistance,
            expected_absorption_zones=absorption
        )
    
    def _quantize_price(self, price: float) -> float:
        """Quantize price to detection granularity."""
        return round(price / self.price_granularity) * self.price_granularity
    
    def _analyze_iceberg_patterns(self, current_time: datetime) -> List[IcebergSignature]:
        """Analyze fill history for iceberg patterns."""
        detected = []
        
        for price_level, fills in self.price_level_fills.items():
            if len(fills) < self.min_refills:
                continue
            
            # Recent fills only
            recent = fills[-20:]
            volumes = [f['volume'] for f in recent]
            
            # Check for consistent fill sizes (iceberg signature)
            if len(volumes) >= self.min_refills:
                mean_vol = np.mean(volumes)
                std_vol = np.std(volumes)
                cv = std_vol / (mean_vol + 1e-8)  # Coefficient of variation
                
                if cv < self.size_variance_threshold:
                    # Consistent sizes - potential iceberg
                    
                    # Determine direction from price relative to bid/ask
                    last_fill = recent[-1]
                    if last_fill['price'] <= last_fill['bid']:
                        direction = 'SELL'  # Selling into bid
                    else:
                        direction = 'BUY'  # Buying into ask
                    
                    # Estimate total size
                    estimated_total = mean_vol * len(recent) * 3  # Assume 3x more hidden
                    
                    detected.append(IcebergSignature(
                        price_level=price_level,
                        estimated_total_size=estimated_total,
                        visible_size=mean_vol,
                        hidden_ratio=0.7,  # Assume 70% hidden
                        direction=direction,
                        confidence=1 - cv,  # Higher consistency = higher confidence
                        refill_count=len(recent),
                        first_detected=recent[0]['time'],
                        last_updated=current_time
                    ))
        
        return detected
    
    def _add_or_update_iceberg(self, new_iceberg: IcebergSignature):
        """Add new iceberg or update existing."""
        for i, existing in enumerate(self.detected_icebergs):
            if abs(existing.price_level - new_iceberg.price_level) < self.price_granularity * 2:
                # Update existing
                self.detected_icebergs[i] = new_iceberg
                return
        
        # Add new
        self.detected_icebergs.append(new_iceberg)
    
    def _update_existing_icebergs(self, price: float, volume: float, 
                                  current_time: datetime):
        """Update existing icebergs with new fill data."""
        quantized = self._quantize_price(price)
        
        for iceberg in self.detected_icebergs:
            if abs(iceberg.price_level - quantized) < self.price_granularity * 2:
                iceberg.refill_count += 1
                iceberg.estimated_total_size += volume
                iceberg.last_updated = current_time
    
    def _clean_old_icebergs(self, current_time: datetime, max_age_minutes: int = 30):
        """Remove old or exhausted icebergs."""
        from datetime import timedelta
        cutoff = current_time - timedelta(minutes=max_age_minutes)
        
        self.detected_icebergs = [
            i for i in self.detected_icebergs
            if i.last_updated > cutoff
        ]
    
    def _calculate_iceberg_presence(self) -> float:
        """Calculate overall iceberg presence probability."""
        if not self.detected_icebergs:
            return 0.0
        
        avg_confidence = np.mean([i.confidence for i in self.detected_icebergs])
        count_factor = min(1.0, len(self.detected_icebergs) / 5)
        
        return float(np.clip(avg_confidence * count_factor, 0, 1))
    
    def _calculate_iceberg_support(self, current_price: float) -> float:
        """Calculate support strength from buying icebergs."""
        buying_icebergs = [
            i for i in self.detected_icebergs
            if i.direction == 'BUY' and i.price_level < current_price
        ]
        
        if not buying_icebergs:
            return 0.0
        
        # Weight by size and confidence
        total = sum(i.estimated_total_size * i.confidence for i in buying_icebergs)
        return float(np.clip(total / 1000, 0, 1))
    
    def _calculate_iceberg_resistance(self, current_price: float) -> float:
        """Calculate resistance strength from selling icebergs."""
        selling_icebergs = [
            i for i in self.detected_icebergs
            if i.direction == 'SELL' and i.price_level > current_price
        ]
        
        if not selling_icebergs:
            return 0.0
        
        total = sum(i.estimated_total_size * i.confidence for i in selling_icebergs)
        return float(np.clip(total / 1000, 0, 1))
    
    def _get_absorption_zones(self, current_price: float) -> List[float]:
        """Get price zones where volume absorption is expected."""
        zones = []
        
        for iceberg in self.detected_icebergs:
            if iceberg.confidence > 0.6:
                zones.append(iceberg.price_level)
        
        # Sort by distance from current price
        zones.sort(key=lambda p: abs(p - current_price))
        return zones[:5]
    
    def get_iceberg_at_price(self, price: float) -> Optional[IcebergSignature]:
        """Get iceberg at a specific price level if exists."""
        quantized = self._quantize_price(price)
        
        for iceberg in self.detected_icebergs:
            if abs(iceberg.price_level - quantized) < self.price_granularity * 2:
                return iceberg
        
        return None
    
    def get_nearest_iceberg(self, price: float, direction: str) -> Optional[Tuple[float, float]]:
        """Get nearest iceberg in given direction."""
        if direction == 'BUY':
            icebergs = [i for i in self.detected_icebergs if i.price_level > price]
        else:
            icebergs = [i for i in self.detected_icebergs if i.price_level < price]
        
        if not icebergs:
            return None
        
        nearest = min(icebergs, key=lambda i: abs(i.price_level - price))
        return nearest.price_level, nearest.estimated_total_size
