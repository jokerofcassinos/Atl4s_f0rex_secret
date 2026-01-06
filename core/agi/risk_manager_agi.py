"""
AGI Ultra-Complete: RiskManager AGI Components

Sistema de Risco Ultra-Inteligente:
- DeepRiskReasoningEngine: Raciocínio profundo sobre risco
- PredictiveRiskModelingEngine: Modelagem preditiva
- DynamicRiskLimitManager: Limites dinâmicos
- AdvancedCorrelationAnalyzer: Correlação avançada
- RiskSituationMemory: Memória de situações
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("RiskManagerAGI")


class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    level: RiskLevel
    score: float
    factors: List[str]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskPrediction:
    """Risk prediction."""
    horizon_minutes: int
    predicted_risk: float
    confidence: float
    warning_signs: List[str]


class DeepRiskReasoningEngine:
    """Deep reasoning about risk."""
    
    def __init__(self):
        self.reasoning_history: deque = deque(maxlen=100)
        self.risk_factors: Dict[str, float] = {}
        
        self._init_risk_factors()
        logger.info("DeepRiskReasoningEngine initialized")
    
    def _init_risk_factors(self):
        """Initialize risk factors."""
        self.risk_factors = {
            'volatility': 0.3,
            'correlation': 0.2,
            'drawdown': 0.25,
            'concentration': 0.15,
            'liquidity': 0.1
        }
    
    def reason(self, market_data: Dict, positions: List[Dict]) -> RiskAssessment:
        """Perform deep risk reasoning."""
        risks = []
        factors = []
        score = 0.0
        
        volatility = market_data.get('volatility', 0.01)
        if volatility > 0.03:
            risks.append("High volatility detected")
            score += self.risk_factors['volatility'] * (volatility / 0.03)
            factors.append(f"Volatility: {volatility:.1%}")
        
        total_exposure = sum(abs(p.get('size', 0)) for p in positions)
        if total_exposure > 0:
            concentration = max(abs(p.get('size', 0)) for p in positions) / total_exposure
            if concentration > 0.5:
                risks.append("High position concentration")
                score += self.risk_factors['concentration'] * concentration
                factors.append(f"Concentration: {concentration:.0%}")
        
        current_dd = market_data.get('drawdown', 0)
        if current_dd > 0.1:
            risks.append("Significant drawdown")
            score += self.risk_factors['drawdown'] * (current_dd / 0.1)
            factors.append(f"Drawdown: {current_dd:.1%}")
        
        level = self._score_to_level(score)
        
        recommendations = self._generate_recommendations(level, factors)
        
        assessment = RiskAssessment(
            level=level,
            score=min(1.0, score),
            factors=factors,
            recommendations=recommendations
        )
        
        self.reasoning_history.append(assessment)
        return assessment
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert score to risk level."""
        if score < 0.1:
            return RiskLevel.MINIMAL
        elif score < 0.25:
            return RiskLevel.LOW
        elif score < 0.5:
            return RiskLevel.MODERATE
        elif score < 0.75:
            return RiskLevel.HIGH
        elif score < 0.9:
            return RiskLevel.EXTREME
        return RiskLevel.CRITICAL
    
    def _generate_recommendations(self, level: RiskLevel, factors: List[str]) -> List[str]:
        """Generate recommendations."""
        recs = []
        
        if level in [RiskLevel.HIGH, RiskLevel.EXTREME, RiskLevel.CRITICAL]:
            recs.append("Reduce position sizes")
            recs.append("Increase stop-loss distances")
        
        if level == RiskLevel.CRITICAL:
            recs.append("Consider closing all positions")
            recs.append("Halt new entries")
        
        for factor in factors:
            if 'Volatility' in factor:
                recs.append("Adjust for high volatility")
            elif 'Concentration' in factor:
                recs.append("Diversify positions")
        
        return recs


class PredictiveRiskModelingEngine:
    """Predicts future risk."""
    
    def __init__(self):
        self.predictions: deque = deque(maxlen=100)
        self.accuracy_history: deque = deque(maxlen=50)
        
        logger.info("PredictiveRiskModelingEngine initialized")
    
    def predict(self, historical_risk: List[float], horizon_minutes: int = 60) -> RiskPrediction:
        """Predict future risk."""
        if not historical_risk:
            return RiskPrediction(
                horizon_minutes=horizon_minutes,
                predicted_risk=0.5,
                confidence=0.3,
                warning_signs=[]
            )
        
        trend = 0.0
        if len(historical_risk) > 5:
            recent = historical_risk[-5:]
            older = historical_risk[-10:-5] if len(historical_risk) > 10 else historical_risk[:5]
            trend = np.mean(recent) - np.mean(older)
        
        current = historical_risk[-1]
        predicted = current + trend * (horizon_minutes / 60)
        predicted = max(0, min(1, predicted))
        
        volatility = np.std(historical_risk) if len(historical_risk) > 2 else 0.1
        confidence = max(0.3, 1 - volatility * 2)
        
        warnings = []
        if trend > 0.1:
            warnings.append("Risk trending upward")
        if predicted > 0.7:
            warnings.append("High risk expected")
        if volatility > 0.2:
            warnings.append("Risk highly volatile")
        
        prediction = RiskPrediction(
            horizon_minutes=horizon_minutes,
            predicted_risk=predicted,
            confidence=confidence,
            warning_signs=warnings
        )
        
        self.predictions.append(prediction)
        return prediction
    
    def evaluate_accuracy(self, predicted: float, actual: float):
        """Evaluate prediction accuracy."""
        error = abs(predicted - actual)
        accuracy = 1 - error
        self.accuracy_history.append(accuracy)


class DynamicRiskLimitManager:
    """Manages dynamic risk limits."""
    
    def __init__(self):
        self.base_limits = {
            'max_position_size': 0.1,
            'max_total_exposure': 0.5,
            'max_per_trade_risk': 0.02,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15
        }
        self.current_limits: Dict[str, float] = {}
        self.limit_history: deque = deque(maxlen=100)
        
        self._reset_limits()
        logger.info("DynamicRiskLimitManager initialized")
    
    def _reset_limits(self):
        """Reset to base limits."""
        self.current_limits = self.base_limits.copy()
    
    def adjust_limits(self, risk_level: RiskLevel, performance: float = 0.0):
        """Adjust limits based on conditions."""
        multiplier = 1.0
        
        if risk_level == RiskLevel.MINIMAL:
            multiplier = 1.3
        elif risk_level == RiskLevel.LOW:
            multiplier = 1.1
        elif risk_level == RiskLevel.MODERATE:
            multiplier = 1.0
        elif risk_level == RiskLevel.HIGH:
            multiplier = 0.7
        elif risk_level == RiskLevel.EXTREME:
            multiplier = 0.4
        elif risk_level == RiskLevel.CRITICAL:
            multiplier = 0.1
        
        if performance > 0.1:
            multiplier *= 1.1
        elif performance < -0.05:
            multiplier *= 0.8
        
        for limit_name in self.current_limits:
            self.current_limits[limit_name] = self.base_limits[limit_name] * multiplier
        
        self.limit_history.append({
            'limits': self.current_limits.copy(),
            'risk_level': risk_level.value,
            'multiplier': multiplier,
            'timestamp': time.time()
        })
    
    def check_limit(self, limit_name: str, value: float) -> Tuple[bool, str]:
        """Check if value exceeds limit."""
        limit = self.current_limits.get(limit_name, 1.0)
        
        if value > limit:
            return False, f"{limit_name} exceeded: {value:.2%} > {limit:.2%}"
        return True, "Within limits"


class AdvancedCorrelationAnalyzer:
    """Analyzes correlations between positions."""
    
    def __init__(self):
        self.correlation_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.analysis_history: List[Dict] = []
        
        logger.info("AdvancedCorrelationAnalyzer initialized")
    
    def analyze(self, positions: List[Dict], price_history: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """Analyze correlations."""
        if len(positions) < 2:
            return {'portfolio_correlation': 0.0, 'highly_correlated_pairs': []}
        
        symbols = [p['symbol'] for p in positions if 'symbol' in p]
        
        if price_history:
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i < j and sym1 in price_history and sym2 in price_history:
                        corr = np.corrcoef(price_history[sym1], price_history[sym2])[0, 1]
                        self.correlation_matrix[sym1][sym2] = corr
                        self.correlation_matrix[sym2][sym1] = corr
        
        highly_correlated = []
        for sym1 in self.correlation_matrix:
            for sym2, corr in self.correlation_matrix[sym1].items():
                if abs(corr) > 0.7 and sym1 < sym2:
                    highly_correlated.append({
                        'pair': (sym1, sym2),
                        'correlation': corr
                    })
        
        all_corrs = [
            abs(corr) for d in self.correlation_matrix.values() 
            for corr in d.values()
        ]
        avg_correlation = np.mean(all_corrs) if all_corrs else 0.0
        
        return {
            'portfolio_correlation': avg_correlation,
            'highly_correlated_pairs': highly_correlated
        }


class RiskSituationMemory:
    """Remembers past risk situations."""
    
    def __init__(self):
        self.situations: List[Dict] = []
        self.patterns: Dict[str, Dict] = {}
        
        logger.info("RiskSituationMemory initialized")
    
    def remember(self, situation: Dict, outcome: str, severity: float):
        """Remember a risk situation."""
        entry = {
            'situation': situation,
            'outcome': outcome,
            'severity': severity,
            'timestamp': time.time()
        }
        self.situations.append(entry)
        
        pattern_key = self._extract_pattern(situation)
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {'occurrences': 0, 'outcomes': defaultdict(int)}
        
        self.patterns[pattern_key]['occurrences'] += 1
        self.patterns[pattern_key]['outcomes'][outcome] += 1
    
    def _extract_pattern(self, situation: Dict) -> str:
        """Extract pattern from situation."""
        risk_level = situation.get('risk_level', 'unknown')
        volatility = 'high' if situation.get('volatility', 0) > 0.02 else 'low'
        trend = situation.get('trend', 'unknown')
        
        return f"{risk_level}_{volatility}_{trend}"
    
    def recall_similar(self, current_situation: Dict) -> List[Dict]:
        """Recall similar past situations."""
        pattern = self._extract_pattern(current_situation)
        
        similar = [s for s in self.situations if self._extract_pattern(s['situation']) == pattern]
        return similar[-10:]
    
    def get_expected_outcome(self, situation: Dict) -> Optional[str]:
        """Get expected outcome based on history."""
        pattern = self._extract_pattern(situation)
        
        if pattern in self.patterns:
            outcomes = self.patterns[pattern]['outcomes']
            if outcomes:
                return max(outcomes, key=outcomes.get)
        return None


class RiskManagerAGI:
    """Main RiskManager AGI System."""
    
    def __init__(self):
        self.reasoning = DeepRiskReasoningEngine()
        self.predictor = PredictiveRiskModelingEngine()
        self.limits = DynamicRiskLimitManager()
        self.correlation = AdvancedCorrelationAnalyzer()
        self.memory = RiskSituationMemory()
        
        self.current_assessment: Optional[RiskAssessment] = None
        self.risk_history: deque = deque(maxlen=100)
        
        logger.info("RiskManagerAGI initialized")
    
    def assess_risk(self, market_data: Dict, positions: List[Dict]) -> RiskAssessment:
        """Comprehensive risk assessment."""
        assessment = self.reasoning.reason(market_data, positions)
        
        self.current_assessment = assessment
        self.risk_history.append(assessment.score)
        
        self.limits.adjust_limits(assessment.level)
        
        corr_analysis = self.correlation.analyze(positions)
        if corr_analysis['portfolio_correlation'] > 0.6:
            assessment.factors.append(f"High portfolio correlation: {corr_analysis['portfolio_correlation']:.0%}")
        
        similar = self.memory.recall_similar({
            'risk_level': assessment.level.value,
            'volatility': market_data.get('volatility', 0)
        })
        
        if similar:
            bad_outcomes = sum(1 for s in similar if s['outcome'] == 'loss')
            if bad_outcomes > len(similar) * 0.6:
                assessment.recommendations.append("Historical pattern suggests caution")
        
        return assessment
    
    def predict_risk(self, horizon_minutes: int = 60) -> RiskPrediction:
        """Predict future risk."""
        history = list(self.risk_history)
        return self.predictor.predict(history, horizon_minutes)
    
    def check_trade(self, trade: Dict) -> Tuple[bool, List[str]]:
        """Check if trade is within risk limits."""
        issues = []
        
        size = trade.get('size', 0)
        ok, msg = self.limits.check_limit('max_position_size', size)
        if not ok:
            issues.append(msg)
        
        risk = trade.get('risk', 0)
        ok, msg = self.limits.check_limit('max_per_trade_risk', risk)
        if not ok:
            issues.append(msg)
        
        return len(issues) == 0, issues
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGI status."""
        return {
            'current_risk': self.current_assessment.level.value if self.current_assessment else 'unknown',
            'risk_score': self.current_assessment.score if self.current_assessment else 0,
            'limits': self.limits.current_limits,
            'situations_remembered': len(self.memory.situations)
        }
