"""
AGI Ultra: Unified Reasoning Layer

Resolves conflicts and synthesizes insights across different AGI modules.
Provides a single coherent decision from multiple reasoning paths.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("UnifiedReasoning")


class ConflictType(Enum):
    """Types of reasoning conflicts."""
    DIRECTION = "direction"      # BUY vs SELL
    TIMING = "timing"            # Now vs Wait
    MAGNITUDE = "magnitude"      # Position size disagreement
    CONFIDENCE = "confidence"    # Confidence level mismatch
    NONE = "none"


@dataclass
class ModuleInsight:
    """An insight from a single AGI module."""
    module_name: str
    decision: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedDecision:
    """The final unified decision from all modules."""
    decision: str
    confidence: float
    reasoning: str
    
    # Module agreement metrics
    agreement_score: float
    conflict_type: ConflictType
    dissenting_modules: List[str]
    
    # Evidence synthesis
    supporting_evidence: List[str]
    key_factors: List[str]
    risk_factors: List[str]
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0


class UnifiedReasoningLayer:
    """
    AGI Ultra: Unified Reasoning Layer.
    
    Synthesizes insights from multiple AGI modules into a single coherent decision:
    - Conflict detection and resolution
    - Weighted evidence aggregation
    - Uncertainty quantification
    - Decision explanation generation
    """
    
    def __init__(
        self,
        default_module_weights: Optional[Dict[str, float]] = None,
        conflict_resolution_strategy: str = "weighted_vote"
    ):
        # Module importance weights
        self.module_weights = default_module_weights or {
            'trend_architect': 1.2,
            'sniper': 1.0,
            'quant': 1.1,
            'pattern_recognizer': 0.9,
            'risk_manager': 1.3,
            'news_analyzer': 0.7,
            'meta_reasoning': 1.4,
            'holographic_memory': 1.0,
            'why_engine': 1.2
        }
        
        self.conflict_strategy = conflict_resolution_strategy
        
        # Historical accuracy tracking per module
        self.module_accuracy: Dict[str, List[bool]] = {}
        self.accuracy_window = 100
        
        # Decision history
        self.decision_history: List[UnifiedDecision] = []
        self.max_history = 1000
        
        logger.info(f"UnifiedReasoningLayer initialized: strategy={conflict_resolution_strategy}")
    
    def synthesize(
        self,
        insights: List[ModuleInsight],
        context: Optional[Dict[str, Any]] = None
    ) -> UnifiedDecision:
        """
        Synthesize multiple module insights into a unified decision.
        
        Args:
            insights: List of insights from different modules
            context: Additional context for resolution
            
        Returns:
            UnifiedDecision with synthesized reasoning
        """
        start_time = time.time()
        
        if not insights:
            return self._empty_decision()
        
        # Detect conflicts
        conflict_type, conflicting_groups = self._detect_conflicts(insights)
        
        # Resolve conflicts and get final decision
        if conflict_type == ConflictType.NONE:
            decision, confidence = self._aggregate_unanimous(insights)
        else:
            decision, confidence = self._resolve_conflict(
                insights, conflict_type, conflicting_groups, context
            )
        
        # Compute agreement
        agreement_score = self._compute_agreement(insights, decision)
        dissenting = [i.module_name for i in insights if i.decision != decision]
        
        # Synthesize evidence
        supporting, contradicting = self._aggregate_evidence(insights, decision)
        
        # Generate explanation
        reasoning = self._generate_reasoning(insights, decision, conflict_type, agreement_score)
        
        # Extract key factors
        key_factors = self._extract_key_factors(insights, decision)
        risk_factors = self._extract_risk_factors(insights, decision)
        
        unified = UnifiedDecision(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            agreement_score=agreement_score,
            conflict_type=conflict_type,
            dissenting_modules=dissenting,
            supporting_evidence=supporting,
            key_factors=key_factors,
            risk_factors=risk_factors,
            timestamp=time.time(),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Store in history
        self.decision_history.append(unified)
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
        
        logger.debug(f"Unified decision: {decision} ({confidence:.1%}), agreement={agreement_score:.1%}")
        
        return unified
    
    def _detect_conflicts(
        self,
        insights: List[ModuleInsight]
    ) -> Tuple[ConflictType, Dict[str, List[ModuleInsight]]]:
        """Detect type and groups of conflicting insights."""
        # Group by decision
        groups: Dict[str, List[ModuleInsight]] = {}
        for insight in insights:
            if insight.decision not in groups:
                groups[insight.decision] = []
            groups[insight.decision].append(insight)
        
        # Check for direction conflict (BUY vs SELL)
        has_buy = 'BUY' in groups
        has_sell = 'SELL' in groups
        
        if has_buy and has_sell:
            return ConflictType.DIRECTION, groups
        
        # Check for timing conflict
        has_action = has_buy or has_sell or 'CLOSE' in groups
        has_wait = 'WAIT' in groups or 'HOLD' in groups
        
        if has_action and has_wait:
            return ConflictType.TIMING, groups
        
        # Check for confidence mismatch
        if len(insights) > 1:
            confidences = [i.confidence for i in insights]
            if max(confidences) - min(confidences) > 0.4:
                return ConflictType.CONFIDENCE, groups
        
        return ConflictType.NONE, groups
    
    def _aggregate_unanimous(
        self,
        insights: List[ModuleInsight]
    ) -> Tuple[str, float]:
        """Aggregate when all modules agree."""
        decision = insights[0].decision
        
        # Weighted average confidence
        total_weight = 0
        weighted_conf = 0
        
        for insight in insights:
            weight = self.module_weights.get(insight.module_name, 1.0)
            
            # Boost weight by module accuracy
            if insight.module_name in self.module_accuracy:
                acc = self._get_module_accuracy(insight.module_name)
                weight *= (0.5 + 0.5 * acc)
            
            weighted_conf += insight.confidence * weight
            total_weight += weight
        
        confidence = weighted_conf / total_weight if total_weight > 0 else 0.5
        
        return decision, confidence
    
    def _resolve_conflict(
        self,
        insights: List[ModuleInsight],
        conflict_type: ConflictType,
        groups: Dict[str, List[ModuleInsight]],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Resolve conflicts between modules."""
        if self.conflict_strategy == "weighted_vote":
            return self._weighted_vote_resolution(insights, groups)
        elif self.conflict_strategy == "max_confidence":
            return self._max_confidence_resolution(insights)
        elif self.conflict_strategy == "risk_averse":
            return self._risk_averse_resolution(insights, groups)
        else:
            return self._weighted_vote_resolution(insights, groups)
    
    def _weighted_vote_resolution(
        self,
        insights: List[ModuleInsight],
        groups: Dict[str, List[ModuleInsight]]
    ) -> Tuple[str, float]:
        """Resolve by weighted voting."""
        decision_scores: Dict[str, float] = {}
        
        for decision, group in groups.items():
            score = 0
            for insight in group:
                weight = self.module_weights.get(insight.module_name, 1.0)
                acc = self._get_module_accuracy(insight.module_name)
                weight *= (0.5 + 0.5 * acc)
                score += insight.confidence * weight
            decision_scores[decision] = score
        
        # Pick highest score
        best_decision = max(decision_scores, key=decision_scores.get)
        total_score = sum(decision_scores.values())
        
        confidence = decision_scores[best_decision] / total_score if total_score > 0 else 0.5
        
        # Reduce confidence due to conflict
        confidence *= 0.85
        
        return best_decision, confidence
    
    def _max_confidence_resolution(
        self,
        insights: List[ModuleInsight]
    ) -> Tuple[str, float]:
        """Pick decision from most confident module."""
        best = max(insights, key=lambda i: i.confidence)
        return best.decision, best.confidence * 0.9  # Slight reduction for conflict
    
    def _risk_averse_resolution(
        self,
        insights: List[ModuleInsight],
        groups: Dict[str, List[ModuleInsight]]
    ) -> Tuple[str, float]:
        """Prefer WAIT/HOLD when there's conflict."""
        if 'WAIT' in groups:
            wait_conf = np.mean([i.confidence for i in groups['WAIT']])
            return 'WAIT', wait_conf
        elif 'HOLD' in groups:
            hold_conf = np.mean([i.confidence for i in groups['HOLD']])
            return 'HOLD', hold_conf
        else:
            # Fall back to weighted vote
            return self._weighted_vote_resolution(insights, groups)
    
    def _compute_agreement(self, insights: List[ModuleInsight], decision: str) -> float:
        """Compute agreement score."""
        if not insights:
            return 0.0
        
        agreeing = sum(1 for i in insights if i.decision == decision)
        return agreeing / len(insights)
    
    def _aggregate_evidence(
        self,
        insights: List[ModuleInsight],
        decision: str
    ) -> Tuple[List[str], List[str]]:
        """Aggregate supporting and contradicting evidence."""
        supporting = []
        contradicting = []
        
        for insight in insights:
            if insight.decision == decision:
                supporting.extend(insight.supporting_evidence[:3])
            else:
                contradicting.extend(insight.supporting_evidence[:2])
        
        # Deduplicate
        supporting = list(dict.fromkeys(supporting))[:10]
        contradicting = list(dict.fromkeys(contradicting))[:5]
        
        return supporting, contradicting
    
    def _generate_reasoning(
        self,
        insights: List[ModuleInsight],
        decision: str,
        conflict_type: ConflictType,
        agreement: float
    ) -> str:
        """Generate explanation for the unified decision."""
        agreeing = [i for i in insights if i.decision == decision]
        
        if conflict_type == ConflictType.NONE:
            prefix = f"All {len(insights)} modules agree on {decision}"
        else:
            prefix = f"{len(agreeing)}/{len(insights)} modules voted for {decision} (conflict: {conflict_type.value})"
        
        # Add key reasoning from top modules
        key_reasons = []
        for insight in sorted(agreeing, key=lambda i: i.confidence, reverse=True)[:3]:
            if insight.reasoning:
                key_reasons.append(f"{insight.module_name}: {insight.reasoning[:100]}")
        
        if key_reasons:
            return f"{prefix}. {'; '.join(key_reasons)}"
        return prefix
    
    def _extract_key_factors(
        self,
        insights: List[ModuleInsight],
        decision: str
    ) -> List[str]:
        """Extract key decision factors."""
        factors = []
        
        for insight in insights:
            if insight.decision == decision:
                for evidence in insight.supporting_evidence[:2]:
                    factors.append(evidence)
        
        return list(dict.fromkeys(factors))[:5]
    
    def _extract_risk_factors(
        self,
        insights: List[ModuleInsight],
        decision: str
    ) -> List[str]:
        """Extract risk factors (contradicting evidence)."""
        risks = []
        
        for insight in insights:
            if insight.decision != decision:
                for evidence in insight.supporting_evidence[:2]:
                    risks.append(f"[{insight.module_name}] {evidence}")
        
        return list(dict.fromkeys(risks))[:5]
    
    def _empty_decision(self) -> UnifiedDecision:
        """Return empty decision when no insights provided."""
        return UnifiedDecision(
            decision='WAIT',
            confidence=0.0,
            reasoning="No module insights provided",
            agreement_score=0.0,
            conflict_type=ConflictType.NONE,
            dissenting_modules=[],
            supporting_evidence=[],
            key_factors=[],
            risk_factors=[]
        )
    
    def _get_module_accuracy(self, module_name: str) -> float:
        """Get historical accuracy for a module."""
        if module_name not in self.module_accuracy:
            return 0.5
        
        history = self.module_accuracy[module_name]
        if not history:
            return 0.5
        
        return sum(history) / len(history)
    
    def update_module_accuracy(self, module_name: str, was_correct: bool):
        """Update module accuracy tracking."""
        if module_name not in self.module_accuracy:
            self.module_accuracy[module_name] = []
        
        self.module_accuracy[module_name].append(was_correct)
        
        # Maintain window
        if len(self.module_accuracy[module_name]) > self.accuracy_window:
            self.module_accuracy[module_name] = self.module_accuracy[module_name][-self.accuracy_window:]
    
    def get_module_rankings(self) -> List[Tuple[str, float]]:
        """Get modules ranked by historical accuracy."""
        rankings = []
        
        for name, history in self.module_accuracy.items():
            if history:
                acc = sum(history) / len(history)
                rankings.append((name, acc))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
