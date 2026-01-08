"""
Temporal Abstraction - Multi-scale Temporal Reasoning.

Implements hierarchical temporal abstraction for reasoning
across multiple time scales simultaneously.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("TemporalAbstraction")


@dataclass
class TemporalLevel:
    """A temporal abstraction level."""
    name: str
    scale_minutes: int
    current_state: str
    confidence: float
    trend_direction: str
    volatility: float


@dataclass
class TemporalView:
    """Complete temporal view across all scales."""
    levels: List[TemporalLevel]
    alignment: float  # How aligned are different scales
    dominant_scale: str
    time_horizon_recommendation: str
    multi_scale_signal: str


class TemporalAbstraction:
    """
    The Temporal Synthesizer.
    
    Implements multi-scale temporal reasoning through:
    - Hierarchical time scale decomposition
    - Cross-scale alignment detection
    - Temporal pattern aggregation
    - Time horizon optimization
    """
    
    def __init__(self):
        self.scales = {
            'TICK': 1,
            'MICRO': 5,
            'SHORT': 15,
            'MEDIUM': 60,
            'LONG': 240,
            'DAILY': 1440,
        }
        
        self.level_states: Dict[str, deque] = {
            name: deque(maxlen=100) for name in self.scales
        }
        
        logger.info("TemporalAbstraction initialized")
    
    def analyze(self, data_by_timeframe: Dict[str, Dict]) -> TemporalView:
        """
        Analyze market across multiple timeframes.
        
        Args:
            data_by_timeframe: Dict mapping timeframe name to OHLCV data
            
        Returns:
            TemporalView with multi-scale analysis.
        """
        levels = []
        
        for name, scale in self.scales.items():
            if name in data_by_timeframe:
                data = data_by_timeframe[name]
            else:
                data = self._interpolate_data(name, data_by_timeframe)
            
            level = self._analyze_level(name, scale, data)
            levels.append(level)
            
            # Store state
            self.level_states[name].append(level)
        
        # Calculate alignment
        alignment = self._calculate_alignment(levels)
        
        # Find dominant scale
        dominant = self._find_dominant_scale(levels)
        
        # Time horizon recommendation
        horizon = self._recommend_horizon(levels, alignment)
        
        # Multi-scale signal
        signal = self._synthesize_signal(levels)
        
        return TemporalView(
            levels=levels,
            alignment=alignment,
            dominant_scale=dominant,
            time_horizon_recommendation=horizon,
            multi_scale_signal=signal
        )
    
    def _analyze_level(self, name: str, scale: int, 
                       data: Optional[Dict]) -> TemporalLevel:
        """Analyze a single temporal level."""
        if data is None:
            return TemporalLevel(
                name=name,
                scale_minutes=scale,
                current_state='UNKNOWN',
                confidence=0.0,
                trend_direction='NEUTRAL',
                volatility=0.0
            )
        
        # Extract trend
        close = data.get('close', [])
        if isinstance(close, (list, np.ndarray)) and len(close) >= 2:
            trend = close[-1] - close[0]
            if trend > 0:
                direction = 'UP'
            elif trend < 0:
                direction = 'DOWN'
            else:
                direction = 'FLAT'
            
            # Volatility
            volatility = np.std(close) if len(close) > 1 else 0.0
        else:
            direction = 'UNKNOWN'
            volatility = 0.0
        
        # Determine state
        if direction == 'UP' and volatility < np.mean(close) * 0.01:
            state = 'TRENDING_UP'
        elif direction == 'DOWN' and volatility < np.mean(close) * 0.01:
            state = 'TRENDING_DOWN'
        elif volatility > np.mean(close) * 0.02:
            state = 'VOLATILE'
        else:
            state = 'RANGING'
        
        return TemporalLevel(
            name=name,
            scale_minutes=scale,
            current_state=state,
            confidence=0.7,
            trend_direction=direction,
            volatility=float(volatility) if not np.isnan(volatility) else 0.0
        )
    
    def _interpolate_data(self, target: str, 
                          available: Dict[str, Dict]) -> Optional[Dict]:
        """Interpolate data for missing timeframe."""
        # Find closest available timeframe
        target_scale = self.scales.get(target, 60)
        
        best_match = None
        min_diff = float('inf')
        
        for name, data in available.items():
            if name in self.scales:
                diff = abs(self.scales[name] - target_scale)
                if diff < min_diff:
                    min_diff = diff
                    best_match = data
        
        return best_match
    
    def _calculate_alignment(self, levels: List[TemporalLevel]) -> float:
        """Calculate how aligned the temporal levels are."""
        if len(levels) < 2:
            return 1.0
        
        directions = [l.trend_direction for l in levels if l.trend_direction != 'UNKNOWN']
        
        if not directions:
            return 0.5
        
        # Count most common direction
        from collections import Counter
        counts = Counter(directions)
        most_common = counts.most_common(1)[0][1]
        
        return most_common / len(directions)
    
    def _find_dominant_scale(self, levels: List[TemporalLevel]) -> str:
        """Find the dominant temporal scale."""
        # Dominant = highest confidence with clear trend
        trending = [l for l in levels if 'TRENDING' in l.current_state]
        
        if trending:
            return max(trending, key=lambda l: l.confidence).name
        
        return 'MEDIUM'  # Default
    
    def _recommend_horizon(self, levels: List[TemporalLevel], 
                           alignment: float) -> str:
        """Recommend trading time horizon."""
        if alignment > 0.8:
            # Strong alignment - can trade longer timeframes
            return 'SWING'
        elif alignment > 0.5:
            return 'INTRADAY'
        else:
            # Low alignment - short trades only
            return 'SCALP'
    
    def _synthesize_signal(self, levels: List[TemporalLevel]) -> str:
        """Synthesize signal from all levels."""
        up_count = sum(1 for l in levels if l.trend_direction == 'UP')
        down_count = sum(1 for l in levels if l.trend_direction == 'DOWN')
        
        if up_count > len(levels) * 0.6:
            return 'STRONG_BUY'
        elif down_count > len(levels) * 0.6:
            return 'STRONG_SELL'
        elif up_count > down_count:
            return 'LEAN_BUY'
        elif down_count > up_count:
            return 'LEAN_SELL'
        else:
            return 'NEUTRAL'
