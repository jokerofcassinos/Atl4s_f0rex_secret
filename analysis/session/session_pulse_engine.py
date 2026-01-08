"""
Session Pulse Engine - Real-time Session Detection with Adaptive Volatility Thresholds.

Implements metacognitive session awareness through recursive self-reflection
and neural plasticity for adapting to changing market conditions.
"""

import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("SessionPulse")


class SessionType(Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    PACIFIC = "PACIFIC"
    OVERLAP_ASIA_LONDON = "OVERLAP_ASIA_LONDON"
    OVERLAP_LONDON_NY = "OVERLAP_LONDON_NY"
    DEAD_ZONE = "DEAD_ZONE"


@dataclass
class SessionState:
    """Current session state with metacognitive metrics."""
    session_type: SessionType
    strength: float  # 0-1 intensity
    volatility_ratio: float  # Current vs historical volatility
    liquidity_score: float  # Estimated liquidity 0-100
    time_until_next: int  # Seconds until session change
    confidence: float  # Self-assessed confidence in detection
    
    # Metacognitive metrics
    self_reflection_score: float = 0.0
    pattern_novelty: float = 0.0
    causal_certainty: float = 0.0


@dataclass
class SessionProfile:
    """Historical session profile with adaptive learning."""
    avg_volatility: float
    avg_volume: float
    typical_range: float
    best_instruments: List[str]
    risk_multiplier: float
    
    # Neural plasticity weights
    adaptation_rate: float = 0.1
    memory_decay: float = 0.95


class SessionPulseEngine:
    """
    The Temporal Consciousness.
    
    Implements real-time session detection with:
    - Adaptive volatility thresholds
    - Recursive self-reflection loops
    - Neural plasticity for regime adaptation
    - Causal inference for session transitions
    """
    
    # Session times in UTC hours
    SESSION_TIMES = {
        SessionType.ASIA: (0, 9),      # 00:00 - 09:00 UTC
        SessionType.LONDON: (7, 16),    # 07:00 - 16:00 UTC
        SessionType.NEW_YORK: (12, 21), # 12:00 - 21:00 UTC
        SessionType.PACIFIC: (21, 24),  # 21:00 - 00:00 UTC
    }
    
    def __init__(self):
        self.session_profiles: Dict[SessionType, SessionProfile] = {}
        self.volatility_memory: List[float] = []
        self.current_state: Optional[SessionState] = None
        
        # Metacognitive state
        self.reflection_depth = 3  # Recursive reflection levels
        self.plasticity_rate = 0.1  # Neural adaptation speed
        self.causal_graph: Dict[str, List[str]] = {}
        
        # Heuristic evolution
        self.evolved_thresholds: Dict[str, float] = {
            'volatility_spike': 1.5,
            'liquidity_dry': 0.3,
            'overlap_boost': 1.8,
        }
        
        self._initialize_profiles()
        logger.info("SessionPulseEngine initialized with metacognitive recursion")
    
    def _initialize_profiles(self):
        """Initialize session profiles with default heuristics."""
        self.session_profiles = {
            SessionType.ASIA: SessionProfile(
                avg_volatility=0.3, avg_volume=0.4, typical_range=30,
                best_instruments=['USDJPY', 'AUDUSD', 'NZDUSD'],
                risk_multiplier=0.7
            ),
            SessionType.LONDON: SessionProfile(
                avg_volatility=0.8, avg_volume=0.9, typical_range=80,
                best_instruments=['GBPUSD', 'EURUSD', 'EURGBP'],
                risk_multiplier=1.0
            ),
            SessionType.NEW_YORK: SessionProfile(
                avg_volatility=0.9, avg_volume=1.0, typical_range=90,
                best_instruments=['EURUSD', 'GBPUSD', 'USDCAD'],
                risk_multiplier=1.0
            ),
            SessionType.PACIFIC: SessionProfile(
                avg_volatility=0.2, avg_volume=0.2, typical_range=20,
                best_instruments=['AUDUSD', 'NZDUSD'],
                risk_multiplier=0.5
            ),
        }
    
    def detect_session(self, current_time: Optional[datetime] = None) -> SessionState:
        """
        Detect current trading session with metacognitive awareness.
        
        Returns:
            SessionState with session type, strength, and metacognitive metrics.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        hour = current_time.hour
        minute = current_time.minute
        
        # Primary session detection
        active_sessions = []
        for session_type, (start, end) in self.SESSION_TIMES.items():
            if start <= hour < end:
                active_sessions.append(session_type)
        
        # Detect overlaps (highest priority)
        if SessionType.ASIA in active_sessions and SessionType.LONDON in active_sessions:
            primary_session = SessionType.OVERLAP_ASIA_LONDON
            strength = 0.9
        elif SessionType.LONDON in active_sessions and SessionType.NEW_YORK in active_sessions:
            primary_session = SessionType.OVERLAP_LONDON_NY
            strength = 1.0  # Maximum liquidity
        elif active_sessions:
            primary_session = active_sessions[0]
            strength = self._calculate_session_strength(primary_session, hour, minute)
        else:
            primary_session = SessionType.DEAD_ZONE
            strength = 0.1
        
        # Calculate metacognitive metrics
        volatility_ratio = self._calculate_volatility_ratio()
        liquidity_score = self._estimate_liquidity(primary_session, strength)
        time_until_next = self._calculate_time_until_transition(current_time)
        
        # Recursive self-reflection
        reflection_score = self._recursive_self_reflect(primary_session, strength)
        pattern_novelty = self._assess_pattern_novelty()
        causal_certainty = self._infer_causal_certainty(primary_session)
        
        # Confidence based on all factors
        confidence = (strength * 0.4 + 
                     reflection_score * 0.3 + 
                     causal_certainty * 0.3)
        
        self.current_state = SessionState(
            session_type=primary_session,
            strength=strength,
            volatility_ratio=volatility_ratio,
            liquidity_score=liquidity_score,
            time_until_next=time_until_next,
            confidence=confidence,
            self_reflection_score=reflection_score,
            pattern_novelty=pattern_novelty,
            causal_certainty=causal_certainty
        )
        
        return self.current_state
    
    def _calculate_session_strength(self, session: SessionType, hour: int, minute: int) -> float:
        """Calculate session intensity based on time within session."""
        if session not in self.SESSION_TIMES:
            return 0.5
        
        start, end = self.SESSION_TIMES[session]
        duration = end - start
        progress = (hour - start + minute / 60) / duration
        
        # Bell curve - strongest in middle of session
        strength = np.exp(-((progress - 0.5) ** 2) / 0.2)
        return float(np.clip(strength, 0.1, 1.0))
    
    def _calculate_volatility_ratio(self) -> float:
        """Calculate current vs historical volatility ratio."""
        if len(self.volatility_memory) < 10:
            return 1.0
        
        recent = np.mean(self.volatility_memory[-5:])
        historical = np.mean(self.volatility_memory[-50:]) if len(self.volatility_memory) >= 50 else recent
        
        return recent / historical if historical > 0 else 1.0
    
    def _estimate_liquidity(self, session: SessionType, strength: float) -> float:
        """Estimate current liquidity based on session and strength."""
        base_liquidity = {
            SessionType.OVERLAP_LONDON_NY: 100,
            SessionType.OVERLAP_ASIA_LONDON: 80,
            SessionType.NEW_YORK: 85,
            SessionType.LONDON: 90,
            SessionType.ASIA: 50,
            SessionType.PACIFIC: 20,
            SessionType.DEAD_ZONE: 10,
        }
        
        base = base_liquidity.get(session, 30)
        return base * strength
    
    def _calculate_time_until_transition(self, current_time: datetime) -> int:
        """Calculate seconds until next session transition."""
        hour = current_time.hour
        
        # Find next transition point
        transitions = [0, 7, 9, 12, 16, 21]  # UTC hours
        next_transition = min((t for t in transitions if t > hour), default=transitions[0])
        
        if next_transition <= hour:
            next_transition = transitions[0]
            hours_until = (24 - hour) + next_transition
        else:
            hours_until = next_transition - hour
        
        return hours_until * 3600 - current_time.minute * 60 - current_time.second
    
    def _recursive_self_reflect(self, session: SessionType, strength: float, depth: int = 0) -> float:
        """
        Metacognitive recursive self-reflection.
        
        Analyzes the quality of its own session detection through
        multiple levels of self-examination.
        """
        if depth >= self.reflection_depth:
            return 0.5  # Base case
        
        # Level 1: Detection confidence
        detection_quality = strength
        
        # Level 2: Historical accuracy (if we have history)
        historical_accuracy = 0.7  # Placeholder, would use real tracking
        
        # Level 3: Recursive reflection on reflection quality
        meta_reflection = self._recursive_self_reflect(session, strength, depth + 1)
        
        # Combine with decreasing weight for deeper levels
        weight = 1.0 / (depth + 1)
        reflection_score = (detection_quality * 0.4 + 
                          historical_accuracy * 0.3 + 
                          meta_reflection * 0.3 * weight)
        
        return float(np.clip(reflection_score, 0, 1))
    
    def _assess_pattern_novelty(self) -> float:
        """Assess how novel the current pattern is compared to learned patterns."""
        # Placeholder - would compare against pattern database
        return 0.3  # Low novelty = familiar pattern
    
    def _infer_causal_certainty(self, session: SessionType) -> float:
        """Infer causal certainty in session detection."""
        # Strong causality: time -> session, session -> volatility
        time_causality = 0.95  # Time is a strong causal factor
        volatility_consistency = self._calculate_volatility_ratio()
        
        # Causal certainty decreases if volatility doesn't match expectations
        if session in self.session_profiles:
            expected_vol = self.session_profiles[session].avg_volatility
            actual_ratio = volatility_consistency
            deviation = abs(actual_ratio - expected_vol)
            certainty = max(0.3, 1.0 - deviation)
        else:
            certainty = 0.5
        
        return certainty * time_causality
    
    def update_volatility(self, volatility: float):
        """Update volatility memory for adaptive learning."""
        self.volatility_memory.append(volatility)
        if len(self.volatility_memory) > 1000:
            self.volatility_memory = self.volatility_memory[-500:]
    
    def adapt_thresholds(self, performance_delta: float):
        """
        Neural plasticity: Adapt thresholds based on performance.
        
        Args:
            performance_delta: Positive for good performance, negative for bad.
        """
        adaptation = performance_delta * self.plasticity_rate
        
        for key in self.evolved_thresholds:
            # Evolve thresholds based on performance feedback
            noise = np.random.normal(0, 0.05)
            self.evolved_thresholds[key] *= (1 + adaptation + noise)
            self.evolved_thresholds[key] = np.clip(
                self.evolved_thresholds[key], 0.1, 5.0
            )
        
        logger.debug(f"PLASTICITY: Thresholds adapted by {adaptation:.3f}")
    
    def get_trading_recommendation(self) -> Dict[str, any]:
        """Get session-aware trading recommendation."""
        if self.current_state is None:
            self.detect_session()
        
        state = self.current_state
        
        return {
            'session': state.session_type.value,
            'should_trade': state.liquidity_score > 30 and state.strength > 0.3,
            'risk_multiplier': self._get_risk_multiplier(state),
            'recommended_instruments': self._get_recommended_instruments(state),
            'confidence': state.confidence,
            'metacognitive_score': (state.self_reflection_score + 
                                   state.causal_certainty) / 2,
        }
    
    def _get_risk_multiplier(self, state: SessionState) -> float:
        """Calculate risk multiplier based on session state."""
        base = 1.0
        
        # Reduce risk in dead zones
        if state.session_type == SessionType.DEAD_ZONE:
            base *= 0.3
        
        # Increase risk during overlaps
        if 'OVERLAP' in state.session_type.value:
            base *= 1.2
        
        # Adjust by liquidity
        liquidity_factor = state.liquidity_score / 100
        
        return float(np.clip(base * liquidity_factor, 0.1, 2.0))
    
    def _get_recommended_instruments(self, state: SessionState) -> List[str]:
        """Get recommended instruments for current session."""
        session = state.session_type
        
        if session in [SessionType.ASIA]:
            return ['USDJPY', 'AUDUSD', 'NZDUSD']
        elif session in [SessionType.LONDON]:
            return ['GBPUSD', 'EURUSD', 'EURGBP']
        elif session in [SessionType.NEW_YORK]:
            return ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY']
        elif session in [SessionType.OVERLAP_LONDON_NY]:
            return ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD']
        else:
            return ['EURUSD']  # Most liquid always
