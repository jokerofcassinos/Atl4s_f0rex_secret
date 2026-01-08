"""
Institutional Clock - Models Institutional Trading Patterns.

Tracks fix times, option expiries, fund flows, and institutional
trading patterns through recursive pattern synthesis.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("InstitutionalClock")


class InstitutionalEvent(Enum):
    LONDON_FIX = "LONDON_FIX"  # 4 PM London
    TOKYO_FIX = "TOKYO_FIX"   # 9:55 AM Tokyo
    ECB_FIX = "ECB_FIX"       # 2:15 PM Frankfurt
    OPTION_EXPIRY = "OPTION_EXPIRY"
    FUND_REBALANCE = "FUND_REBALANCE"
    MARGIN_CALL_WINDOW = "MARGIN_CALL_WINDOW"
    ALGO_RESET = "ALGO_RESET"  # Midnight algo resets


@dataclass
class InstitutionalTimeWindow:
    """A window of institutional activity."""
    event_type: InstitutionalEvent
    hour_utc: int
    minute: int
    duration_minutes: int
    impact_level: float  # 0-1
    direction_bias: Optional[str]  # 'BUY', 'SELL', or None
    affected_pairs: List[str]
    volatility_multiplier: float


@dataclass
class ClockReading:
    """Current institutional clock state."""
    current_events: List[InstitutionalTimeWindow]
    upcoming_events: List[tuple]  # (event, minutes_until)
    institutional_pressure: float  # -1 to 1 (sell to buy)
    flow_direction: str
    recommended_position_size: float  # Multiplier
    meta_pattern_score: float


class InstitutionalClock:
    """
    The Institutional Pulse.
    
    Models institutional trading patterns through:
    - Fix time tracking and prediction
    - Option expiry impact analysis
    - Fund flow pattern recognition
    - Recursive pattern synthesis for institutional behavior
    """
    
    def __init__(self):
        self.events = self._initialize_events()
        self.pattern_memory: Dict[str, List[float]] = {}
        self.flow_accumulator = 0.0
        
        # Historical pattern weights (evolved over time)
        self.pattern_weights = {
            'fix_time_direction': 0.7,
            'option_gamma': 0.6,
            'fund_flow': 0.5,
            'algo_momentum': 0.4,
        }
        
        logger.info("InstitutionalClock initialized")
    
    def _initialize_events(self) -> List[InstitutionalTimeWindow]:
        """Initialize known institutional events."""
        return [
            # London Fix - 4 PM London (16:00 UTC in winter, 15:00 in summer)
            InstitutionalTimeWindow(
                event_type=InstitutionalEvent.LONDON_FIX,
                hour_utc=16, minute=0,
                duration_minutes=10,
                impact_level=0.9,
                direction_bias=None,  # Direction determined by flow
                affected_pairs=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
                volatility_multiplier=1.5
            ),
            # Tokyo Fix - 9:55 AM Tokyo (00:55 UTC)
            InstitutionalTimeWindow(
                event_type=InstitutionalEvent.TOKYO_FIX,
                hour_utc=0, minute=55,
                duration_minutes=10,
                impact_level=0.6,
                direction_bias=None,
                affected_pairs=['USDJPY', 'EURJPY', 'GBPJPY'],
                volatility_multiplier=1.3
            ),
            # ECB Fix - 2:15 PM Frankfurt (13:15 UTC)
            InstitutionalTimeWindow(
                event_type=InstitutionalEvent.ECB_FIX,
                hour_utc=13, minute=15,
                duration_minutes=5,
                impact_level=0.7,
                direction_bias=None,
                affected_pairs=['EURUSD', 'EURGBP', 'EURJPY'],
                volatility_multiplier=1.4
            ),
            # NY Option Expiry - 10 AM NY (15:00 UTC)
            InstitutionalTimeWindow(
                event_type=InstitutionalEvent.OPTION_EXPIRY,
                hour_utc=14, minute=45,
                duration_minutes=30,
                impact_level=0.8,
                direction_bias=None,
                affected_pairs=['EURUSD', 'USDJPY', 'GBPUSD'],
                volatility_multiplier=1.6
            ),
            # Monthly Fund Rebalance (End of month)
            InstitutionalTimeWindow(
                event_type=InstitutionalEvent.FUND_REBALANCE,
                hour_utc=15, minute=0,
                duration_minutes=120,
                impact_level=0.5,
                direction_bias=None,
                affected_pairs=['EURUSD', 'USDJPY'],
                volatility_multiplier=1.2
            ),
            # Algo Reset Window - Midnight UTC
            InstitutionalTimeWindow(
                event_type=InstitutionalEvent.ALGO_RESET,
                hour_utc=0, minute=0,
                duration_minutes=15,
                impact_level=0.4,
                direction_bias=None,
                affected_pairs=['EURUSD', 'USDJPY'],
                volatility_multiplier=0.8  # Lower vol during reset
            ),
        ]
    
    def read_clock(self, current_time: Optional[datetime] = None) -> ClockReading:
        """
        Get current institutional clock reading.
        
        Returns:
            ClockReading with current events and institutional pressure.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        hour = current_time.hour
        minute = current_time.minute
        current_minutes = hour * 60 + minute
        
        # Find active events
        current_events = []
        upcoming_events = []
        
        for event in self.events:
            event_start = event.hour_utc * 60 + event.minute
            event_end = event_start + event.duration_minutes
            
            if event_start <= current_minutes < event_end:
                current_events.append(event)
            elif 0 < (event_start - current_minutes) <= 60:
                upcoming_events.append((event, event_start - current_minutes))
        
        # Sort upcoming by time
        upcoming_events.sort(key=lambda x: x[1])
        
        # Calculate institutional pressure
        pressure = self._calculate_institutional_pressure(current_events, current_time)
        
        # Determine flow direction
        if pressure > 0.3:
            flow_direction = "INSTITUTIONAL_BUY"
        elif pressure < -0.3:
            flow_direction = "INSTITUTIONAL_SELL"
        else:
            flow_direction = "NEUTRAL"
        
        # Position size recommendation
        if current_events:
            max_impact = max(e.impact_level for e in current_events)
            recommended_size = 1.0 + (max_impact * 0.5)  # Up to 1.5x
        else:
            recommended_size = 1.0
        
        # Meta pattern synthesis
        meta_score = self._synthesize_meta_patterns(current_time, current_events)
        
        return ClockReading(
            current_events=current_events,
            upcoming_events=upcoming_events[:3],
            institutional_pressure=pressure,
            flow_direction=flow_direction,
            recommended_position_size=recommended_size,
            meta_pattern_score=meta_score
        )
    
    def _calculate_institutional_pressure(self, events: List[InstitutionalTimeWindow],
                                         current_time: datetime) -> float:
        """Calculate net institutional buy/sell pressure."""
        if not events:
            return 0.0
        
        # Aggregate pressure from all active events
        total_pressure = 0.0
        total_weight = 0.0
        
        for event in events:
            weight = event.impact_level
            
            # Estimate direction based on historical patterns
            direction = self._estimate_flow_direction(event, current_time)
            
            total_pressure += direction * weight
            total_weight += weight
        
        if total_weight > 0:
            return np.clip(total_pressure / total_weight, -1, 1)
        return 0.0
    
    def _estimate_flow_direction(self, event: InstitutionalTimeWindow,
                                 current_time: datetime) -> float:
        """Estimate flow direction for an event based on patterns."""
        # Use day of month for fund rebalancing
        day = current_time.day
        
        if event.event_type == InstitutionalEvent.FUND_REBALANCE:
            # End of month tends to be USD selling
            if day >= 25:
                return -0.3
        
        if event.event_type == InstitutionalEvent.LONDON_FIX:
            # Random walk - use accumulated flow
            return self.flow_accumulator
        
        # Default to slight buy bias (historical observation)
        return 0.1
    
    def _synthesize_meta_patterns(self, current_time: datetime,
                                  events: List[InstitutionalTimeWindow]) -> float:
        """Synthesize meta-patterns from historical data."""
        # Create time signature
        hour = current_time.hour
        day = current_time.weekday()
        
        key = f"{hour}_{day}"
        
        # Get historical pattern for this time
        if key in self.pattern_memory:
            historical = np.mean(self.pattern_memory[key][-20:])
        else:
            historical = 0.5
        
        # Current pattern strength
        if events:
            current = np.mean([e.impact_level for e in events])
        else:
            current = 0.0
        
        # Blend historical and current
        meta_score = historical * 0.6 + current * 0.4
        
        return float(np.clip(meta_score, 0, 1))
    
    def update_flow(self, direction: str, magnitude: float):
        """Update institutional flow accumulator."""
        if direction == "BUY":
            self.flow_accumulator += magnitude * 0.1
        elif direction == "SELL":
            self.flow_accumulator -= magnitude * 0.1
        
        # Decay over time
        self.flow_accumulator *= 0.95
        self.flow_accumulator = np.clip(self.flow_accumulator, -1, 1)
    
    def record_pattern(self, current_time: datetime, outcome: float):
        """Record pattern outcome for future synthesis."""
        key = f"{current_time.hour}_{current_time.weekday()}"
        
        if key not in self.pattern_memory:
            self.pattern_memory[key] = []
        
        self.pattern_memory[key].append(outcome)
        
        # Keep only recent history
        if len(self.pattern_memory[key]) > 100:
            self.pattern_memory[key] = self.pattern_memory[key][-50:]
    
    def get_event_risk_for_pair(self, pair: str) -> Dict[str, any]:
        """Get institutional event risk for a specific pair."""
        reading = self.read_clock()
        
        relevant_events = [
            e for e in reading.current_events 
            if pair in e.affected_pairs or pair.replace('m', '') in e.affected_pairs
        ]
        
        upcoming_relevant = [
            (e, m) for e, m in reading.upcoming_events
            if pair in e.affected_pairs or pair.replace('m', '') in e.affected_pairs
        ]
        
        if relevant_events:
            max_vol = max(e.volatility_multiplier for e in relevant_events)
            risk_level = "HIGH" if max_vol > 1.4 else "MEDIUM"
        elif upcoming_relevant and upcoming_relevant[0][1] <= 15:
            risk_level = "ELEVATED"
            max_vol = upcoming_relevant[0][0].volatility_multiplier
        else:
            risk_level = "NORMAL"
            max_vol = 1.0
        
        return {
            'risk_level': risk_level,
            'volatility_multiplier': max_vol,
            'active_events': [e.event_type.value for e in relevant_events],
            'minutes_until_event': upcoming_relevant[0][1] if upcoming_relevant else None,
            'institutional_pressure': reading.institutional_pressure,
        }
