"""
Macro Event Horizon - Economic Calendar Integration with Session Timing.

Provides risk-adjusted execution recommendations based on upcoming
economic events and their causal impact predictions.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("MacroEventHorizon")


class EventImpact(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"  # NFP, FOMC, etc.


class EventType(Enum):
    INTEREST_RATE = "INTEREST_RATE"
    EMPLOYMENT = "EMPLOYMENT"
    INFLATION = "INFLATION"
    GDP = "GDP"
    TRADE_BALANCE = "TRADE_BALANCE"
    RETAIL_SALES = "RETAIL_SALES"
    PMI = "PMI"
    CENTRAL_BANK_SPEECH = "CENTRAL_BANK_SPEECH"
    GEOPOLITICAL = "GEOPOLITICAL"


@dataclass
class EconomicEvent:
    """Represents an economic calendar event."""
    name: str
    event_type: EventType
    currency: str
    impact: EventImpact
    scheduled_time: datetime
    
    # Causal predictions
    expected_volatility: float  # Pips
    pre_event_drift: Optional[str]  # 'UP', 'DOWN', or None
    post_event_reversion_prob: float
    
    # Risk parameters
    no_trade_window_before: int  # Minutes
    no_trade_window_after: int  # Minutes


@dataclass 
class EventHorizonReading:
    """Current macro event horizon state."""
    imminent_events: List[EconomicEvent]  # Within 4 hours
    active_no_trade_zones: List[str]  # Currencies to avoid
    risk_level: str  # 'CLEAR', 'CAUTION', 'DANGER', 'BLACKOUT'
    volatility_forecast: Dict[str, float]
    causal_chains: List[Dict]
    recommended_pairs: List[str]
    avoid_pairs: List[str]


class MacroEventHorizon:
    """
    The Macro Consciousness.
    
    Integrates economic calendar with session timing through:
    - Event impact prediction with causal inference
    - No-trade zone management
    - Cross-currency risk assessment
    - Temporal abstraction for event clustering
    """
    
    # Static event definitions (would be fetched from calendar API in production)
    RECURRING_EVENTS = {
        'NFP': {
            'type': EventType.EMPLOYMENT,
            'currency': 'USD',
            'impact': EventImpact.EXTREME,
            'expected_volatility': 100,
            'no_trade_before': 60,
            'no_trade_after': 30,
        },
        'FOMC': {
            'type': EventType.INTEREST_RATE,
            'currency': 'USD',
            'impact': EventImpact.EXTREME,
            'expected_volatility': 150,
            'no_trade_before': 120,
            'no_trade_after': 60,
        },
        'ECB_RATE': {
            'type': EventType.INTEREST_RATE,
            'currency': 'EUR',
            'impact': EventImpact.EXTREME,
            'expected_volatility': 80,
            'no_trade_before': 60,
            'no_trade_after': 45,
        },
        'BOE_RATE': {
            'type': EventType.INTEREST_RATE,
            'currency': 'GBP',
            'impact': EventImpact.EXTREME,
            'expected_volatility': 100,
            'no_trade_before': 60,
            'no_trade_after': 45,
        },
        'CPI_US': {
            'type': EventType.INFLATION,
            'currency': 'USD',
            'impact': EventImpact.HIGH,
            'expected_volatility': 60,
            'no_trade_before': 30,
            'no_trade_after': 20,
        },
        'GDP_US': {
            'type': EventType.GDP,
            'currency': 'USD',
            'impact': EventImpact.HIGH,
            'expected_volatility': 50,
            'no_trade_before': 30,
            'no_trade_after': 15,
        },
    }
    
    # Currency pairs affected by each currency
    CURRENCY_PAIRS = {
        'USD': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'],
        'EUR': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD'],
        'GBP': ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPCHF', 'GBPAUD'],
        'JPY': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY'],
        'CHF': ['USDCHF', 'EURCHF', 'GBPCHF'],
        'CAD': ['USDCAD', 'EURCAD', 'CADJPY'],
        'AUD': ['AUDUSD', 'EURAUD', 'GBPAUD', 'AUDJPY'],
        'NZD': ['NZDUSD', 'EURNZD', 'NZDJPY'],
    }
    
    def __init__(self):
        self.scheduled_events: List[EconomicEvent] = []
        self.event_outcomes: Dict[str, List[float]] = {}  # For learning
        
        # Causal model parameters
        self.causal_weights = {
            'pre_event_momentum': 0.3,
            'event_surprise': 0.5,
            'post_event_reversion': 0.2,
        }
        
        logger.info("MacroEventHorizon initialized")
    
    def scan_horizon(self, current_time: Optional[datetime] = None,
                    lookahead_hours: int = 4) -> EventHorizonReading:
        """
        Scan the macro event horizon.
        
        Returns:
            EventHorizonReading with event analysis and recommendations.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        lookahead = timedelta(hours=lookahead_hours)
        
        # Find imminent events
        imminent = [
            e for e in self.scheduled_events
            if current_time <= e.scheduled_time <= current_time + lookahead
        ]
        
        # Determine active no-trade zones
        no_trade_currencies = set()
        for event in self.scheduled_events:
            minutes_until = (event.scheduled_time - current_time).total_seconds() / 60
            minutes_since = (current_time - event.scheduled_time).total_seconds() / 60
            
            if 0 <= minutes_until <= event.no_trade_window_before:
                no_trade_currencies.add(event.currency)
            elif 0 <= minutes_since <= event.no_trade_window_after:
                no_trade_currencies.add(event.currency)
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(imminent, no_trade_currencies)
        
        # Volatility forecast per currency
        vol_forecast = self._forecast_volatility(imminent, current_time)
        
        # Build causal chains
        causal_chains = self._build_causal_chains(imminent, current_time)
        
        # Get recommended and avoid pairs
        avoid_pairs = self._get_affected_pairs(no_trade_currencies)
        recommended = self._get_safe_pairs(no_trade_currencies)
        
        return EventHorizonReading(
            imminent_events=imminent,
            active_no_trade_zones=list(no_trade_currencies),
            risk_level=risk_level,
            volatility_forecast=vol_forecast,
            causal_chains=causal_chains,
            recommended_pairs=recommended,
            avoid_pairs=avoid_pairs
        )
    
    def _calculate_risk_level(self, imminent: List[EconomicEvent],
                             no_trade: set) -> str:
        """Calculate overall risk level."""
        if not imminent and not no_trade:
            return 'CLEAR'
        
        # Check for extreme events
        extreme_count = sum(1 for e in imminent if e.impact == EventImpact.EXTREME)
        high_count = sum(1 for e in imminent if e.impact == EventImpact.HIGH)
        
        if extreme_count > 0 and len(no_trade) > 0:
            return 'BLACKOUT'
        elif extreme_count > 0 or len(no_trade) >= 3:
            return 'DANGER'
        elif high_count > 0 or len(no_trade) >= 1:
            return 'CAUTION'
        else:
            return 'CLEAR'
    
    def _forecast_volatility(self, imminent: List[EconomicEvent],
                            current_time: datetime) -> Dict[str, float]:
        """Forecast volatility by currency."""
        forecast = {curr: 1.0 for curr in self.CURRENCY_PAIRS.keys()}
        
        for event in imminent:
            minutes_until = max(0, (event.scheduled_time - current_time).total_seconds() / 60)
            
            # Volatility builds as event approaches
            if minutes_until <= 60:
                proximity_factor = 1 + (1 - minutes_until / 60) * 0.5
            else:
                proximity_factor = 1.0
            
            impact_multiplier = {
                EventImpact.LOW: 1.2,
                EventImpact.MEDIUM: 1.5,
                EventImpact.HIGH: 2.0,
                EventImpact.EXTREME: 3.0,
            }.get(event.impact, 1.0)
            
            forecast[event.currency] = max(
                forecast[event.currency],
                impact_multiplier * proximity_factor
            )
        
        return forecast
    
    def _build_causal_chains(self, imminent: List[EconomicEvent],
                            current_time: datetime) -> List[Dict]:
        """Build causal prediction chains for events."""
        chains = []
        
        for event in imminent:
            minutes_until = (event.scheduled_time - current_time).total_seconds() / 60
            
            chain = {
                'event': event.name,
                'currency': event.currency,
                'minutes_until': int(minutes_until),
                'causal_predictions': []
            }
            
            # Pre-event momentum
            if minutes_until <= 60:
                chain['causal_predictions'].append({
                    'effect': 'pre_event_positioning',
                    'direction': event.pre_event_drift,
                    'probability': 0.6,
                    'magnitude': event.expected_volatility * 0.3
                })
            
            # Event impact
            chain['causal_predictions'].append({
                'effect': 'event_volatility_spike',
                'direction': 'UNKNOWN',
                'probability': 0.95,
                'magnitude': event.expected_volatility
            })
            
            # Post-event reversion
            if event.post_event_reversion_prob > 0.5:
                chain['causal_predictions'].append({
                    'effect': 'mean_reversion',
                    'direction': 'OPPOSITE_TO_SPIKE',
                    'probability': event.post_event_reversion_prob,
                    'magnitude': event.expected_volatility * 0.5
                })
            
            chains.append(chain)
        
        return chains
    
    def _get_affected_pairs(self, currencies: set) -> List[str]:
        """Get all pairs affected by given currencies."""
        affected = set()
        for curr in currencies:
            if curr in self.CURRENCY_PAIRS:
                affected.update(self.CURRENCY_PAIRS[curr])
        return list(affected)
    
    def _get_safe_pairs(self, avoid_currencies: set) -> List[str]:
        """Get pairs safe to trade (not affected by events)."""
        all_pairs = set()
        for pairs in self.CURRENCY_PAIRS.values():
            all_pairs.update(pairs)
        
        affected = set(self._get_affected_pairs(avoid_currencies))
        safe = all_pairs - affected
        
        return list(safe) if safe else ['AUDUSD']  # Fallback
    
    def add_event(self, name: str, currency: str, impact: EventImpact,
                 scheduled_time: datetime, event_type: EventType = EventType.GDP):
        """Add an economic event to the calendar."""
        # Get default parameters
        defaults = self.RECURRING_EVENTS.get(name, {
            'expected_volatility': 30,
            'no_trade_before': 15,
            'no_trade_after': 10,
        })
        
        event = EconomicEvent(
            name=name,
            event_type=event_type,
            currency=currency,
            impact=impact,
            scheduled_time=scheduled_time,
            expected_volatility=defaults.get('expected_volatility', 30),
            pre_event_drift=None,
            post_event_reversion_prob=0.4,
            no_trade_window_before=defaults.get('no_trade_before', 15),
            no_trade_window_after=defaults.get('no_trade_after', 10),
        )
        
        self.scheduled_events.append(event)
        logger.info(f"Added event: {name} for {currency} at {scheduled_time}")
    
    def clear_past_events(self, current_time: Optional[datetime] = None):
        """Remove events that have passed."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        self.scheduled_events = [
            e for e in self.scheduled_events
            if e.scheduled_time > current_time - timedelta(hours=1)
        ]
    
    def should_trade_pair(self, pair: str) -> tuple:
        """Quick check if a pair is safe to trade now."""
        reading = self.scan_horizon()
        
        # Normalize pair name
        pair_clean = pair.replace('m', '').upper()
        
        if pair_clean in reading.avoid_pairs or pair in reading.avoid_pairs:
            if reading.imminent_events:
                reason = f"Event risk: {reading.imminent_events[0].name}"
            else:
                reason = f"No-trade zone: {reading.active_no_trade_zones}"
            return False, reason
        
        if reading.risk_level == 'BLACKOUT':
            return False, "Market blackout - extreme event imminent"
        
        return True, f"Clear to trade ({reading.risk_level})"
