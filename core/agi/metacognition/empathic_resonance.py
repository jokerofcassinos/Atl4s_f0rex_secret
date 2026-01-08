"""
Empathic Resonance Engine - Human Trader Psychology Modeling.

Models market sentiment and trader emotion patterns through
resonance pattern matching and collective psychology simulation.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("EmpathicResonance")


@dataclass
class EmotionalState:
    """Modeled emotional state of market participants."""
    fear_level: float  # 0-1
    greed_level: float  # 0-1
    uncertainty: float  # 0-1
    euphoria: float  # 0-1
    capitulation: float  # 0-1
    
    @property
    def dominant_emotion(self) -> str:
        emotions = {
            'FEAR': self.fear_level,
            'GREED': self.greed_level,
            'UNCERTAINTY': self.uncertainty,
            'EUPHORIA': self.euphoria,
            'CAPITULATION': self.capitulation
        }
        return max(emotions, key=emotions.get)


@dataclass
class ResonanceReading:
    """Empathic resonance analysis result."""
    collective_state: EmotionalState
    sentiment_trend: str  # 'IMPROVING', 'DETERIORATING', 'STABLE'
    crowd_behavior: str  # 'HERDING', 'DIVERGING', 'NEUTRAL'
    contrarian_signal: Optional[str]
    resonance_strength: float
    
    # Predictions
    expected_reaction: str
    reaction_probability: float


class EmpathicResonance:
    """
    The Collective Mind Reader.
    
    Models market psychology through:
    - Fear/Greed oscillation tracking
    - Crowd behavior pattern recognition
    - Contrarian signal generation
    - Emotional resonance matching
    """
    
    def __init__(self):
        self.emotion_history: deque = deque(maxlen=200)
        self.current_state = EmotionalState(0.5, 0.5, 0.5, 0.0, 0.0)
        
        # Resonance patterns
        self.resonance_templates = {
            'PANIC_BOTTOM': np.array([0.9, 0.1, 0.8, 0.0, 0.7]),
            'EUPHORIC_TOP': np.array([0.1, 0.9, 0.2, 0.8, 0.0]),
            'HEALTHY_TREND': np.array([0.3, 0.6, 0.4, 0.3, 0.0]),
            'CAPITULATION': np.array([0.95, 0.05, 0.9, 0.0, 0.9]),
        }
        
        logger.info("EmpathicResonance initialized")
    
    def analyze(self, market_data: Dict) -> ResonanceReading:
        """Analyze market data for emotional content."""
        # Update emotional state from market data
        self._update_emotional_state(market_data)
        
        # Track sentiment trend
        trend = self._calculate_sentiment_trend()
        
        # Detect crowd behavior
        crowd = self._detect_crowd_behavior()
        
        # Check for contrarian signals
        contrarian = self._generate_contrarian_signal()
        
        # Match resonance patterns
        resonance = self._match_resonance_patterns()
        
        # Predict reaction
        reaction, probability = self._predict_market_reaction()
        
        return ResonanceReading(
            collective_state=self.current_state,
            sentiment_trend=trend,
            crowd_behavior=crowd,
            contrarian_signal=contrarian,
            resonance_strength=resonance,
            expected_reaction=reaction,
            reaction_probability=probability
        )
    
    def _update_emotional_state(self, data: Dict):
        """Update emotional state based on market data."""
        # Fear from volatility
        volatility = data.get('volatility', 0.5)
        self.current_state.fear_level = np.clip(volatility * 1.5, 0, 1)
        
        # Greed from trend strength
        trend = data.get('trend_strength', 0)
        self.current_state.greed_level = np.clip(0.5 + trend * 0.5, 0, 1)
        
        # Uncertainty from conflicting signals
        conflict = data.get('signal_conflict', 0.5)
        self.current_state.uncertainty = conflict
        
        # Euphoria from extended gains
        gain_streak = data.get('consecutive_gains', 0)
        self.current_state.euphoria = np.clip(gain_streak / 10, 0, 1)
        
        # Capitulation from extreme losses
        loss_streak = data.get('consecutive_losses', 0)
        self.current_state.capitulation = np.clip(loss_streak / 5, 0, 1)
        
        # Store in history
        self.emotion_history.append({
            'time': datetime.now(timezone.utc),
            'state': EmotionalState(**self.current_state.__dict__)
        })
    
    def _calculate_sentiment_trend(self) -> str:
        """Calculate overall sentiment trend."""
        if len(self.emotion_history) < 10:
            return 'STABLE'
        
        recent = [e['state'].fear_level for e in list(self.emotion_history)[-10:]]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0.02:
            return 'DETERIORATING'  # Fear increasing
        elif trend < -0.02:
            return 'IMPROVING'  # Fear decreasing
        return 'STABLE'
    
    def _detect_crowd_behavior(self) -> str:
        """Detect crowd behavior patterns."""
        state = self.current_state
        
        # Herding when emotions are extreme
        if state.fear_level > 0.8 or state.greed_level > 0.8:
            return 'HERDING'
        
        # Diverging when uncertainty is high
        if state.uncertainty > 0.7:
            return 'DIVERGING'
        
        return 'NEUTRAL'
    
    def _generate_contrarian_signal(self) -> Optional[str]:
        """Generate contrarian signals at emotional extremes."""
        state = self.current_state
        
        if state.fear_level > 0.85 and state.capitulation > 0.6:
            return 'BUY_CONTRARIAN'  # Extreme fear = buy opportunity
        
        if state.greed_level > 0.85 and state.euphoria > 0.6:
            return 'SELL_CONTRARIAN'  # Extreme greed = sell opportunity
        
        return None
    
    def _match_resonance_patterns(self) -> float:
        """Match current state to known patterns."""
        current = np.array([
            self.current_state.fear_level,
            self.current_state.greed_level,
            self.current_state.uncertainty,
            self.current_state.euphoria,
            self.current_state.capitulation
        ])
        
        best_match = 0.0
        for pattern in self.resonance_templates.values():
            similarity = np.dot(current, pattern) / (
                np.linalg.norm(current) * np.linalg.norm(pattern) + 1e-8
            )
            best_match = max(best_match, similarity)
        
        return float(best_match)
    
    def _predict_market_reaction(self) -> tuple:
        """Predict likely market reaction."""
        state = self.current_state
        
        if state.capitulation > 0.7:
            return 'BOUNCE', 0.7
        elif state.euphoria > 0.7:
            return 'PULLBACK', 0.65
        elif state.fear_level > 0.6:
            return 'CONTINUED_SELLING', 0.55
        elif state.greed_level > 0.6:
            return 'CONTINUED_BUYING', 0.55
        
        return 'CONSOLIDATION', 0.5
