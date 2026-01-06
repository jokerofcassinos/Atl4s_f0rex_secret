"""
AGI Ultra: Advanced Risk Management System

Features:
- Recursive reasoning about risk
- Memory-based risk patterns
- Multi-scale risk (trade, session, portfolio)
- Adaptive risk limits
- Black Swan protection with AGI integration
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("AdvancedRisk")


class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    overall_level: RiskLevel
    trade_risk: float  # 0-1
    session_risk: float  # 0-1
    portfolio_risk: float  # 0-1
    
    # Limits
    recommended_size: float
    max_allowed_size: float
    
    # Reasoning
    risk_factors: List[str]
    mitigations: List[str]
    
    # Meta
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskEvent:
    """Historical risk event for learning."""
    event_type: str
    risk_level: RiskLevel
    outcome: str  # hit_sl, hit_tp, timeout, etc
    pnl: float
    timestamp: float
    context: Dict[str, Any]


class AdvancedRiskSystem:
    """
    AGI Ultra: Advanced Risk Management.
    
    Features:
    - Multi-scale risk analysis
    - Recursive reasoning integration
    - Adaptive limits based on performance
    - Pattern-based risk detection
    """
    
    def __init__(
        self,
        base_risk_per_trade: float = 0.01,
        max_daily_drawdown: float = 0.05,
        max_portfolio_risk: float = 0.10,
        adaptation_rate: float = 0.05
    ):
        self.base_risk_per_trade = base_risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_portfolio_risk = max_portfolio_risk
        self.adaptation_rate = adaptation_rate
        
        # Current state
        self.current_risk_level = RiskLevel.MODERATE
        self.daily_pnl = 0.0
        self.daily_drawdown = 0.0
        self.open_positions = 0
        self.portfolio_heat = 0.0
        
        # Historical data
        self.risk_events: deque = deque(maxlen=500)
        self.recent_outcomes: deque = deque(maxlen=50)
        
        # Adaptive limits
        self.adaptive_limits = {
            'trade_risk': base_risk_per_trade,
            'session_risk': max_daily_drawdown / 2,
            'portfolio_risk': max_portfolio_risk
        }
        
        # Risk patterns from memory
        self.risk_patterns: Dict[str, float] = {}
        
        # Statistics
        self.assessments_made = 0
        self.trades_blocked = 0
        
        logger.info("AdvancedRiskSystem initialized")
    
    # -------------------------------------------------------------------------
    # MULTI-SCALE RISK ANALYSIS
    # -------------------------------------------------------------------------
    def assess_risk(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        account_balance: float,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Comprehensive multi-scale risk assessment.
        
        Analyzes:
        - Trade-level risk
        - Session-level risk
        - Portfolio-level risk
        """
        self.assessments_made += 1
        context = context or {}
        risk_factors = []
        mitigations = []
        
        # 1. TRADE RISK
        trade_risk = self._calculate_trade_risk(
            size, entry_price, sl_price, account_balance
        )
        
        if trade_risk > self.adaptive_limits['trade_risk'] * 1.5:
            risk_factors.append(f"Trade risk too high: {trade_risk:.1%}")
        
        # 2. SESSION RISK
        session_risk = self._calculate_session_risk()
        
        if self.daily_drawdown > self.max_daily_drawdown * 0.5:
            risk_factors.append(f"Session drawdown elevated: {self.daily_drawdown:.1%}")
        
        # 3. PORTFOLIO RISK
        portfolio_risk = self._calculate_portfolio_risk(
            size, account_balance
        )
        
        if self.portfolio_heat > self.max_portfolio_risk * 0.7:
            risk_factors.append(f"Portfolio heat high: {self.portfolio_heat:.1%}")
        
        # 4. PATTERN-BASED RISK
        pattern_risk = self._check_risk_patterns(symbol, side, context)
        
        # 5. CONTEXTUAL RISK
        volatility = context.get('volatility', 1.0)
        regime = context.get('regime', 'unknown')
        
        if volatility > 2.0:
            risk_factors.append(f"High volatility: {volatility:.1f}x")
            trade_risk *= 1.3
        
        if regime == 'CHOPPY':
            risk_factors.append("Choppy market regime")
            session_risk *= 1.2
        
        # 6. CALCULATE OVERALL RISK LEVEL
        combined_risk = (trade_risk + session_risk + portfolio_risk) / 3
        combined_risk += pattern_risk * 0.2
        
        overall_level = self._risk_to_level(combined_risk)
        
        # 7. CALCULATE LIMITS
        recommended_size, max_size = self._calculate_size_limits(
            combined_risk, account_balance
        )
        
        if size > max_size:
            risk_factors.append(f"Size exceeds max: {size:.2f} > {max_size:.2f}")
            mitigations.append(f"Reduce size to {max_size:.2f}")
        
        # 8. ADD MITIGATIONS
        if overall_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            mitigations.extend(self._suggest_mitigations(risk_factors))
        
        return RiskAssessment(
            overall_level=overall_level,
            trade_risk=trade_risk,
            session_risk=session_risk,
            portfolio_risk=portfolio_risk,
            recommended_size=recommended_size,
            max_allowed_size=max_size,
            risk_factors=risk_factors,
            mitigations=mitigations,
            confidence=1.0 - combined_risk * 0.5
        )
    
    def _calculate_trade_risk(
        self,
        size: float,
        entry: float,
        sl: float,
        balance: float
    ) -> float:
        """Calculate single trade risk."""
        if balance == 0:
            return 1.0
        
        # Distance to SL in price units
        sl_distance = abs(entry - sl)
        
        # Risk amount (simplified - assumes forex)
        risk_amount = sl_distance * size * 10000
        
        return min(1.0, risk_amount / balance)
    
    def _calculate_session_risk(self) -> float:
        """Calculate session/daily risk."""
        # Based on current drawdown and recent performance
        drawdown_risk = self.daily_drawdown / self.max_daily_drawdown
        
        # Recent win rate
        if self.recent_outcomes:
            wins = sum(1 for o in self.recent_outcomes if o > 0)
            recent_winrate = wins / len(self.recent_outcomes)
        else:
            recent_winrate = 0.5
        
        # Lower win rate = higher risk
        performance_risk = 1 - recent_winrate
        
        return (drawdown_risk + performance_risk) / 2
    
    def _calculate_portfolio_risk(self, new_size: float, balance: float) -> float:
        """Calculate portfolio-level risk."""
        # Current heat + new position
        new_heat = self.portfolio_heat + (new_size * 0.01 / balance if balance > 0 else 0)
        
        return min(1.0, new_heat / self.max_portfolio_risk)
    
    def _check_risk_patterns(
        self,
        symbol: str,
        side: str,
        context: Dict[str, Any]
    ) -> float:
        """Check for known risk patterns."""
        pattern_risk = 0.0
        
        # Check symbol-specific patterns
        key = f"{symbol}:{side}"
        if key in self.risk_patterns:
            pattern_risk += self.risk_patterns[key]
        
        # Time-based patterns
        hour = int((time.time() % 86400) / 3600)
        time_key = f"hour:{hour}"
        if time_key in self.risk_patterns:
            pattern_risk += self.risk_patterns[time_key]
        
        return min(1.0, pattern_risk)
    
    def _risk_to_level(self, risk: float) -> RiskLevel:
        """Convert numeric risk to level."""
        if risk < 0.1:
            return RiskLevel.MINIMAL
        elif risk < 0.25:
            return RiskLevel.LOW
        elif risk < 0.5:
            return RiskLevel.MODERATE
        elif risk < 0.75:
            return RiskLevel.HIGH
        elif risk < 0.9:
            return RiskLevel.EXTREME
        else:
            return RiskLevel.CRITICAL
    
    def _calculate_size_limits(
        self,
        risk: float,
        balance: float
    ) -> Tuple[float, float]:
        """Calculate recommended and max position sizes."""
        # Base size from balance
        base_size = balance * self.adaptive_limits['trade_risk']
        
        # Adjust by current risk level
        risk_multiplier = 1.0 - risk * 0.5
        
        recommended = base_size * risk_multiplier
        max_size = base_size * (1.5 - risk)
        
        return max(0.01, recommended), max(0.01, max_size)
    
    def _suggest_mitigations(self, risk_factors: List[str]) -> List[str]:
        """Generate risk mitigation suggestions."""
        mitigations = []
        
        for factor in risk_factors:
            if "trade risk" in factor.lower():
                mitigations.append("Use tighter stop loss")
                mitigations.append("Reduce position size by 30-50%")
            elif "drawdown" in factor.lower():
                mitigations.append("Consider pausing trading for 1 hour")
                mitigations.append("Reduce next trade size by 50%")
            elif "volatility" in factor.lower():
                mitigations.append("Wait for volatility to decrease")
                mitigations.append("Use wider stops if entering")
            elif "portfolio" in factor.lower():
                mitigations.append("Close weakest open position")
                mitigations.append("Don't add new positions")
        
        return list(set(mitigations))[:5]
    
    # -------------------------------------------------------------------------
    # UPDATES AND LEARNING
    # -------------------------------------------------------------------------
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking."""
        self.daily_pnl += pnl
        
        if self.daily_pnl < self.daily_drawdown:
            self.daily_drawdown = self.daily_pnl
    
    def record_trade_outcome(
        self,
        symbol: str,
        side: str,
        outcome: str,
        pnl: float,
        context: Dict[str, Any]
    ):
        """Record trade outcome for learning."""
        self.recent_outcomes.append(pnl)
        
        event = RiskEvent(
            event_type=f"{symbol}:{side}",
            risk_level=self.current_risk_level,
            outcome=outcome,
            pnl=pnl,
            timestamp=time.time(),
            context=context
        )
        self.risk_events.append(event)
        
        # Update patterns
        self._update_risk_patterns(symbol, side, outcome, pnl)
        
        # Adapt limits
        self._adapt_limits(pnl)
    
    def _update_risk_patterns(
        self,
        symbol: str,
        side: str,
        outcome: str,
        pnl: float
    ):
        """Update risk patterns from outcomes."""
        key = f"{symbol}:{side}"
        
        if key not in self.risk_patterns:
            self.risk_patterns[key] = 0.5
        
        # If loss, increase pattern risk
        if pnl < 0:
            self.risk_patterns[key] = min(1.0, self.risk_patterns[key] * 1.1)
        else:
            self.risk_patterns[key] = max(0.1, self.risk_patterns[key] * 0.95)
    
    def _adapt_limits(self, pnl: float):
        """Adapt risk limits based on performance."""
        if pnl > 0:
            # Good trade - slightly increase limits
            for key in self.adaptive_limits:
                max_val = self.base_risk_per_trade * 2 if key == 'trade_risk' else self.max_portfolio_risk * 1.5
                self.adaptive_limits[key] = min(
                    max_val,
                    self.adaptive_limits[key] * (1 + self.adaptation_rate * 0.5)
                )
        else:
            # Bad trade - reduce limits
            for key in self.adaptive_limits:
                min_val = self.base_risk_per_trade * 0.5 if key == 'trade_risk' else self.max_portfolio_risk * 0.3
                self.adaptive_limits[key] = max(
                    min_val,
                    self.adaptive_limits[key] * (1 - self.adaptation_rate)
                )
    
    def add_position(self, risk_amount: float, balance: float):
        """Track new position opening."""
        self.open_positions += 1
        self.portfolio_heat += risk_amount / balance if balance > 0 else 0
    
    def close_position(self, risk_amount: float, balance: float, pnl: float):
        """Track position closing."""
        self.open_positions = max(0, self.open_positions - 1)
        self.portfolio_heat = max(0, self.portfolio_heat - risk_amount / balance if balance > 0 else 0)
        self.update_daily_pnl(pnl)
    
    def reset_daily(self):
        """Reset daily tracking."""
        self.daily_pnl = 0.0
        self.daily_drawdown = 0.0
    
    def should_block_trade(self, assessment: RiskAssessment) -> Tuple[bool, str]:
        """Determine if trading should be blocked."""
        if assessment.overall_level == RiskLevel.CRITICAL:
            self.trades_blocked += 1
            return True, "CRITICAL risk level"
        
        if self.daily_drawdown < -self.max_daily_drawdown:
            self.trades_blocked += 1
            return True, "Daily drawdown limit reached"
        
        if assessment.overall_level == RiskLevel.EXTREME and self.open_positions >= 3:
            self.trades_blocked += 1
            return True, "EXTREME risk with max positions"
        
        return False, ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get risk system statistics."""
        return {
            'assessments_made': self.assessments_made,
            'trades_blocked': self.trades_blocked,
            'current_risk_level': self.current_risk_level.value,
            'daily_pnl': self.daily_pnl,
            'daily_drawdown': self.daily_drawdown,
            'open_positions': self.open_positions,
            'portfolio_heat': self.portfolio_heat,
            'adaptive_limits': self.adaptive_limits,
            'pattern_count': len(self.risk_patterns)
        }
