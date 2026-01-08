"""
Execution Timing Oracle - Optimal Execution Timing.

Combines session, liquidity, and spread analysis for
optimal trade execution timing.
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("ExecutionTimingOracle")


@dataclass
class TimingRecommendation:
    """Execution timing recommendation."""
    should_execute_now: bool
    optimal_delay_seconds: int
    execution_quality_score: float  # 0-1
    reasons: List[str]
    risk_factors: List[str]
    confidence: float


class ExecutionTimingOracle:
    """
    The Timing Master.
    
    Determines optimal execution timing through:
    - Session quality integration
    - Spread condition monitoring
    - Liquidity window detection
    - Multi-factor timing optimization
    """
    
    def __init__(self):
        self.timing_history: List[Dict] = []
        
        # Weights for different factors
        self.weights = {
            'session': 0.25,
            'spread': 0.30,
            'liquidity': 0.25,
            'volatility': 0.20,
        }
        
        logger.info("ExecutionTimingOracle initialized")
    
    def analyze(self, 
               session_score: float,
               spread_score: float,
               liquidity_score: float,
               volatility: float,
               urgency: float = 0.5) -> TimingRecommendation:
        """
        Analyze optimal execution timing.
        
        Args:
            session_score: 0-1 session quality
            spread_score: 0-1 spread favorability (higher = tighter)
            liquidity_score: 0-1 liquidity availability
            volatility: Current volatility
            urgency: 0-1 how urgent is the trade
            
        Returns:
            TimingRecommendation.
        """
        reasons = []
        risk_factors = []
        
        # Calculate composite score
        vol_score = 1 - min(1.0, volatility * 50)  # Lower vol = better
        
        composite = (
            session_score * self.weights['session'] +
            spread_score * self.weights['spread'] +
            liquidity_score * self.weights['liquidity'] +
            vol_score * self.weights['volatility']
        )
        
        # Analyze factors
        if session_score > 0.7:
            reasons.append(f"Session quality excellent ({session_score:.1%})")
        elif session_score < 0.4:
            risk_factors.append(f"Poor session quality ({session_score:.1%})")
        
        if spread_score > 0.7:
            reasons.append(f"Spreads tight ({spread_score:.1%})")
        elif spread_score < 0.4:
            risk_factors.append(f"Spreads wide ({spread_score:.1%})")
        
        if liquidity_score > 0.7:
            reasons.append(f"Good liquidity ({liquidity_score:.1%})")
        elif liquidity_score < 0.4:
            risk_factors.append(f"Low liquidity ({liquidity_score:.1%})")
        
        if volatility > 0.02:
            risk_factors.append(f"High volatility detected")
        
        # Decision threshold (adjusted by urgency)
        threshold = 0.6 - (urgency * 0.2)  # Higher urgency = lower threshold
        
        should_execute = composite >= threshold
        
        # Calculate delay if not executing
        if not should_execute:
            if session_score < 0.4:
                delay = 1800  # Wait for better session
            elif spread_score < 0.4:
                delay = 300  # Wait for spread improvement
            else:
                delay = 120  # Short wait
        else:
            delay = 0
        
        # Confidence
        confidence = 0.5 + abs(composite - threshold) * 0.5
        
        return TimingRecommendation(
            should_execute_now=should_execute,
            optimal_delay_seconds=delay,
            execution_quality_score=composite,
            reasons=reasons,
            risk_factors=risk_factors,
            confidence=min(0.95, confidence)
        )
    
    def get_quick_check(self, session_active: bool, spread_ok: bool,
                       liquidity_ok: bool) -> Tuple[bool, str]:
        """Quick execution check."""
        if not session_active:
            return False, "Wait for active session"
        
        if not spread_ok:
            return False, "Spread too wide"
        
        if not liquidity_ok:
            return False, "Low liquidity"
        
        return True, "Execution conditions OK"
    
    def should_use_limit(self, urgency: float, spread_score: float) -> bool:
        """Determine if limit order should be used."""
        if urgency > 0.8:
            return False  # Market order for urgency
        
        if spread_score < 0.5:
            return True  # Use limit when spreads wide
        
        return urgency < 0.3  # Use limit when not urgent
    
    def record_timing_outcome(self, recommendation: TimingRecommendation,
                             actual_slippage: float, success: bool):
        """Record timing outcome for learning."""
        self.timing_history.append({
            'score': recommendation.execution_quality_score,
            'executed': recommendation.should_execute_now,
            'slippage': actual_slippage,
            'success': success
        })
        
        # Keep history bounded
        if len(self.timing_history) > 500:
            self.timing_history = self.timing_history[-250:]
    
    def get_timing_stats(self) -> Dict:
        """Get timing performance statistics."""
        if not self.timing_history:
            return {}
        
        executed = [h for h in self.timing_history if h['executed']]
        waited = [h for h in self.timing_history if not h['executed']]
        
        return {
            'total_decisions': len(self.timing_history),
            'immediate_executions': len(executed),
            'delayed_executions': len(waited),
            'avg_slippage_immediate': np.mean([h['slippage'] for h in executed]) if executed else 0,
            'success_rate': np.mean([h['success'] for h in self.timing_history])
        }
