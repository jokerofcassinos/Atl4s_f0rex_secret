"""
Counterparty Modeler - Counterparty Behavior Modeling.

Models likely counterparty behavior and response patterns
for enhanced execution strategy.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("CounterpartyModeler")


@dataclass
class CounterpartyProfile:
    """A modeled counterparty profile."""
    profile_type: str  # 'RETAIL', 'INSTITUTIONAL', 'HFT', 'MARKET_MAKER'
    aggressiveness: float  # 0-1
    typical_size: float
    response_time_ms: int
    fade_probability: float  # Probability they fade moves


@dataclass
class CounterpartyAnalysis:
    """Counterparty analysis result."""
    likely_counterparty: str
    expected_response: str
    our_advantage: float  # -1 to 1, positive = we have advantage
    recommended_strategy: str
    aggression_level: str


class CounterpartyModeler:
    """
    The Adversary Reader.
    
    Models counterparty behavior through:
    - Counterparty classification
    - Response pattern prediction
    - Adversarial advantage calculation
    - Strategy recommendation
    """
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
        self.interaction_history: deque = deque(maxlen=200)
        
        logger.info("CounterpartyModeler initialized")
    
    def _initialize_profiles(self) -> Dict[str, CounterpartyProfile]:
        """Initialize counterparty profiles."""
        return {
            'RETAIL': CounterpartyProfile(
                profile_type='RETAIL',
                aggressiveness=0.3,
                typical_size=0.1,
                response_time_ms=2000,
                fade_probability=0.2
            ),
            'INSTITUTIONAL': CounterpartyProfile(
                profile_type='INSTITUTIONAL',
                aggressiveness=0.6,
                typical_size=10.0,
                response_time_ms=500,
                fade_probability=0.5
            ),
            'HFT': CounterpartyProfile(
                profile_type='HFT',
                aggressiveness=0.9,
                typical_size=1.0,
                response_time_ms=1,
                fade_probability=0.8
            ),
            'MARKET_MAKER': CounterpartyProfile(
                profile_type='MARKET_MAKER',
                aggressiveness=0.5,
                typical_size=5.0,
                response_time_ms=10,
                fade_probability=0.6
            )
        }
    
    def analyze(self, our_order_size: float, market_condition: str,
               time_of_day_score: float) -> CounterpartyAnalysis:
        """
        Analyze likely counterparty.
        
        Args:
            our_order_size: Our order size in lots
            market_condition: 'TRENDING', 'RANGING', 'VOLATILE'
            time_of_day_score: 0-1 liquidity score
            
        Returns:
            CounterpartyAnalysis.
        """
        # Determine likely counterparty based on conditions
        if time_of_day_score < 0.3:
            likely = 'HFT'  # Low liquidity = HFT dominant
        elif our_order_size > 5:
            likely = 'INSTITUTIONAL'  # Large orders attract institutions
        elif time_of_day_score > 0.7:
            likely = 'MARKET_MAKER'  # High liquidity = MMs
        else:
            likely = 'RETAIL'
        
        profile = self.profiles[likely]
        
        # Predict response
        if market_condition == 'TRENDING':
            if profile.fade_probability > 0.5:
                response = 'FADE_ATTEMPT'
            else:
                response = 'FOLLOW_MOMENTUM'
        elif market_condition == 'VOLATILE':
            response = 'WIDEN_SPREAD'
        else:
            response = 'PROVIDE_LIQUIDITY'
        
        # Calculate our advantage
        if likely == 'RETAIL':
            advantage = 0.3  # We're likely smarter
        elif likely == 'HFT':
            advantage = -0.5  # They're faster
        elif likely == 'MARKET_MAKER':
            advantage = -0.2  # They have spread advantage
        else:
            advantage = 0.0  # Even match
        
        # Recommended strategy
        if advantage > 0.2:
            strategy = 'AGGRESSIVE'
            aggression = 'HIGH'
        elif advantage < -0.3:
            strategy = 'PASSIVE_LIMIT'
            aggression = 'LOW'
        else:
            strategy = 'BALANCED'
            aggression = 'MEDIUM'
        
        return CounterpartyAnalysis(
            likely_counterparty=likely,
            expected_response=response,
            our_advantage=advantage,
            recommended_strategy=strategy,
            aggression_level=aggression
        )
    
    def record_interaction(self, counterparty_type: str, 
                          our_result: str, slippage: float):
        """Record interaction for learning."""
        self.interaction_history.append({
            'counterparty': counterparty_type,
            'result': our_result,
            'slippage': slippage
        })
    
    def get_adversary_profile(self, profile_type: str) -> Optional[CounterpartyProfile]:
        """Get specific counterparty profile."""
        return self.profiles.get(profile_type)
