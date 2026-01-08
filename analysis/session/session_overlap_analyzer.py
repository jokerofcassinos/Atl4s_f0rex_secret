"""
Session Overlap Analyzer - Multi-dimensional Analysis of Session Overlaps.

Implements causal web navigation and temporal abstraction for
predicting liquidity injection patterns during session overlaps.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("SessionOverlap")


@dataclass
class OverlapProfile:
    """Profile of a session overlap period."""
    name: str
    primary_session: str
    secondary_session: str
    start_hour_utc: int
    end_hour_utc: int
    
    # Quality metrics
    liquidity_injection_rate: float  # 0-1
    volatility_multiplier: float
    spread_compression: float  # Higher = tighter spreads
    momentum_bias: str  # 'TREND' or 'REVERSAL'
    
    # Causal relationships
    causes_breakout_prob: float
    causes_reversal_prob: float
    institutional_activity: float


@dataclass
class OverlapAnalysis:
    """Result of overlap analysis."""
    current_overlap: Optional[OverlapProfile]
    overlap_intensity: float  # 0-1
    liquidity_forecast: Dict[str, float]  # Next 60 minutes
    optimal_entry_window: Tuple[int, int]  # Minutes from now
    
    # Causal inference results
    causal_chains: List[Dict]
    temporal_abstraction_level: int
    predicted_volatility_spike: Optional[int]  # Minutes from now


class SessionOverlapAnalyzer:
    """
    The Confluence Oracle.
    
    Analyzes session overlaps through:
    - Causal web navigation across temporal dimensions
    - Temporal abstraction for multi-scale analysis
    - Liquidity injection prediction
    - Cross-domain causal inference
    """
    
    def __init__(self):
        self.overlaps = self._initialize_overlaps()
        self.causal_graph = self._build_causal_graph()
        self.temporal_memory: List[Dict] = []
        
        # Temporal abstraction levels
        self.abstraction_levels = {
            1: 'TICK',      # Sub-minute
            2: 'MINUTE',    # 1-5 min
            3: 'SESSION',   # 1-4 hours
            4: 'DAILY',     # 24 hours
            5: 'WEEKLY',    # 7 days
        }
        
        logger.info("SessionOverlapAnalyzer initialized with causal web")
    
    def _initialize_overlaps(self) -> List[OverlapProfile]:
        """Initialize known session overlaps with causal profiles."""
        return [
            OverlapProfile(
                name="LONDON_NY",
                primary_session="LONDON",
                secondary_session="NEW_YORK",
                start_hour_utc=12,
                end_hour_utc=16,
                liquidity_injection_rate=0.95,
                volatility_multiplier=1.5,
                spread_compression=0.9,
                momentum_bias='TREND',
                causes_breakout_prob=0.65,
                causes_reversal_prob=0.35,
                institutional_activity=0.95
            ),
            OverlapProfile(
                name="ASIA_LONDON",
                primary_session="ASIA",
                secondary_session="LONDON",
                start_hour_utc=7,
                end_hour_utc=9,
                liquidity_injection_rate=0.70,
                volatility_multiplier=1.2,
                spread_compression=0.75,
                momentum_bias='REVERSAL',
                causes_breakout_prob=0.40,
                causes_reversal_prob=0.60,
                institutional_activity=0.70
            ),
            OverlapProfile(
                name="NY_ASIA",
                primary_session="NEW_YORK",
                secondary_session="ASIA",
                start_hour_utc=21,
                end_hour_utc=0,
                liquidity_injection_rate=0.30,
                volatility_multiplier=0.6,
                spread_compression=0.4,
                momentum_bias='REVERSAL',
                causes_breakout_prob=0.20,
                causes_reversal_prob=0.80,
                institutional_activity=0.25
            ),
        ]
    
    def _build_causal_graph(self) -> Dict[str, List[Dict]]:
        """Build causal web for session transitions and effects."""
        return {
            'session_start': [
                {'effect': 'liquidity_spike', 'strength': 0.8, 'delay_minutes': 5},
                {'effect': 'volatility_increase', 'strength': 0.7, 'delay_minutes': 10},
                {'effect': 'spread_compression', 'strength': 0.6, 'delay_minutes': 15},
            ],
            'overlap_begin': [
                {'effect': 'liquidity_surge', 'strength': 0.95, 'delay_minutes': 0},
                {'effect': 'momentum_acceleration', 'strength': 0.8, 'delay_minutes': 5},
                {'effect': 'institutional_entry', 'strength': 0.85, 'delay_minutes': 10},
            ],
            'overlap_end': [
                {'effect': 'liquidity_drain', 'strength': 0.7, 'delay_minutes': 0},
                {'effect': 'volatility_spike', 'strength': 0.6, 'delay_minutes': 5},
                {'effect': 'trend_reversal', 'strength': 0.4, 'delay_minutes': 15},
            ],
            'fix_time': [
                {'effect': 'price_manipulation', 'strength': 0.5, 'delay_minutes': -10},
                {'effect': 'volatility_spike', 'strength': 0.8, 'delay_minutes': 0},
                {'effect': 'mean_reversion', 'strength': 0.7, 'delay_minutes': 10},
            ],
        }
    
    def analyze(self, current_time: Optional[datetime] = None) -> OverlapAnalysis:
        """
        Analyze current overlap status with causal inference.
        
        Returns:
            OverlapAnalysis with predictions and causal chains.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        hour = current_time.hour
        
        # Find active overlap
        current_overlap = None
        for overlap in self.overlaps:
            if overlap.start_hour_utc <= hour < overlap.end_hour_utc:
                current_overlap = overlap
                break
            # Handle overnight overlaps
            if overlap.start_hour_utc > overlap.end_hour_utc:
                if hour >= overlap.start_hour_utc or hour < overlap.end_hour_utc:
                    current_overlap = overlap
                    break
        
        # Calculate overlap intensity (stronger in middle)
        intensity = self._calculate_overlap_intensity(current_overlap, hour)
        
        # Forecast liquidity for next 60 minutes
        liquidity_forecast = self._forecast_liquidity(current_time, current_overlap)
        
        # Find optimal entry window
        optimal_window = self._find_optimal_entry(liquidity_forecast)
        
        # Navigate causal web
        causal_chains = self._navigate_causal_web(current_time, current_overlap)
        
        # Determine temporal abstraction level
        abstraction_level = self._determine_abstraction_level(intensity)
        
        # Predict volatility spike
        vol_spike = self._predict_volatility_spike(current_time, current_overlap, causal_chains)
        
        return OverlapAnalysis(
            current_overlap=current_overlap,
            overlap_intensity=intensity,
            liquidity_forecast=liquidity_forecast,
            optimal_entry_window=optimal_window,
            causal_chains=causal_chains,
            temporal_abstraction_level=abstraction_level,
            predicted_volatility_spike=vol_spike
        )
    
    def _calculate_overlap_intensity(self, overlap: Optional[OverlapProfile], 
                                     hour: int) -> float:
        """Calculate intensity based on position within overlap."""
        if overlap is None:
            return 0.0
        
        start = overlap.start_hour_utc
        end = overlap.end_hour_utc
        
        if end < start:  # Overnight
            duration = (24 - start) + end
            progress = ((hour - start) % 24) / duration
        else:
            duration = end - start
            progress = (hour - start) / duration
        
        # Bell curve - strongest in middle
        intensity = np.exp(-((progress - 0.5) ** 2) / 0.15)
        
        # Scale by overlap quality
        intensity *= overlap.liquidity_injection_rate
        
        return float(np.clip(intensity, 0, 1))
    
    def _forecast_liquidity(self, current_time: datetime, 
                           overlap: Optional[OverlapProfile]) -> Dict[str, float]:
        """Forecast liquidity for next 60 minutes in 5-minute buckets."""
        forecast = {}
        base_liquidity = overlap.liquidity_injection_rate if overlap else 0.3
        
        for minute_offset in range(0, 60, 5):
            future_time = current_time + timedelta(minutes=minute_offset)
            future_hour = future_time.hour
            
            # Check if still in overlap
            if overlap:
                if overlap.start_hour_utc <= future_hour < overlap.end_hour_utc:
                    liquidity = base_liquidity
                else:
                    liquidity = base_liquidity * 0.5
            else:
                # Check if entering an overlap
                for ov in self.overlaps:
                    if ov.start_hour_utc <= future_hour < ov.end_hour_utc:
                        liquidity = ov.liquidity_injection_rate
                        break
                else:
                    liquidity = 0.3
            
            forecast[f"+{minute_offset}m"] = liquidity
        
        return forecast
    
    def _find_optimal_entry(self, forecast: Dict[str, float]) -> Tuple[int, int]:
        """Find optimal entry window based on liquidity forecast."""
        max_liquidity = 0
        optimal_start = 0
        
        for key, liquidity in forecast.items():
            minute = int(key.replace('+', '').replace('m', ''))
            if liquidity > max_liquidity:
                max_liquidity = liquidity
                optimal_start = minute
        
        # Entry window is 10 minutes around peak
        return (max(0, optimal_start - 5), optimal_start + 5)
    
    def _navigate_causal_web(self, current_time: datetime,
                            overlap: Optional[OverlapProfile]) -> List[Dict]:
        """Navigate causal graph to find relevant causal chains."""
        chains = []
        
        hour = current_time.hour
        minute = current_time.minute
        
        # Check for overlap boundaries
        if overlap:
            minutes_into_overlap = (hour - overlap.start_hour_utc) * 60 + minute
            minutes_until_end = (overlap.end_hour_utc - hour) * 60 - minute
            
            if minutes_into_overlap < 30:
                # Near overlap start
                for effect in self.causal_graph['overlap_begin']:
                    chains.append({
                        'cause': 'overlap_begin',
                        **effect,
                        'time_from_now': effect['delay_minutes'] - minutes_into_overlap
                    })
            
            if minutes_until_end < 30:
                # Near overlap end
                for effect in self.causal_graph['overlap_end']:
                    chains.append({
                        'cause': 'overlap_end',
                        **effect,
                        'time_from_now': minutes_until_end + effect['delay_minutes']
                    })
        
        # Check for fix time (4 PM London = 16:00 UTC in winter)
        if hour == 15 and minute >= 45:
            for effect in self.causal_graph['fix_time']:
                chains.append({
                    'cause': 'fix_time',
                    **effect,
                    'time_from_now': (60 - minute) + effect['delay_minutes']
                })
        
        return chains
    
    def _determine_abstraction_level(self, intensity: float) -> int:
        """Determine appropriate temporal abstraction level."""
        if intensity > 0.8:
            return 1  # TICK level - high precision needed
        elif intensity > 0.5:
            return 2  # MINUTE level
        elif intensity > 0.2:
            return 3  # SESSION level
        else:
            return 4  # DAILY level - coarse analysis sufficient
    
    def _predict_volatility_spike(self, current_time: datetime,
                                  overlap: Optional[OverlapProfile],
                                  causal_chains: List[Dict]) -> Optional[int]:
        """Predict next volatility spike based on causal chains."""
        for chain in causal_chains:
            if chain['effect'] == 'volatility_spike' and chain['strength'] > 0.6:
                time_from_now = chain.get('time_from_now', 0)
                if 0 <= time_from_now <= 30:
                    return time_from_now
        
        return None
    
    def get_trading_bias(self) -> Dict[str, any]:
        """Get current trading bias based on overlap analysis."""
        analysis = self.analyze()
        
        if analysis.current_overlap:
            bias = analysis.current_overlap.momentum_bias
            breakout_prob = analysis.current_overlap.causes_breakout_prob
            reversal_prob = analysis.current_overlap.causes_reversal_prob
            
            if breakout_prob > reversal_prob:
                strategy = "TREND_FOLLOW"
            else:
                strategy = "MEAN_REVERSION"
        else:
            bias = "NEUTRAL"
            strategy = "WAIT"
            breakout_prob = 0.5
            reversal_prob = 0.5
        
        return {
            'bias': bias,
            'strategy': strategy,
            'breakout_probability': breakout_prob,
            'reversal_probability': reversal_prob,
            'overlap_intensity': analysis.overlap_intensity,
            'optimal_entry': analysis.optimal_entry_window,
            'vol_spike_warning': analysis.predicted_volatility_spike,
        }
