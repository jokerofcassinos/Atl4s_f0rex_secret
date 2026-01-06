"""
AGI Ultra: Decision Memory Expanded

Complete decision tracking and analysis:
- Full decision history with context
- Deep retrospective analysis
- Pattern identification in decisions
- Pre-execution result prediction
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import pickle
import hashlib

logger = logging.getLogger("DecisionMemoryExpanded")


class DecisionOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class DecisionRecord:
    """Complete record of a trading decision."""
    decision_id: str
    timestamp: float
    
    # Decision details
    decision: str  # BUY, SELL, WAIT
    symbol: str
    timeframe: str
    confidence: float
    
    # Context at decision time
    market_context: Dict[str, Any]
    module_votes: Dict[str, str]
    reasoning_chain: List[str]
    
    # Execution details
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    size: Optional[float] = None
    
    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_seconds: float = 0.0
    
    # Meta
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def get_signature(self) -> str:
        """Generate unique signature for pattern matching."""
        key = f"{self.decision}:{self.symbol}:{self.market_context.get('regime', 'unknown')}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class DecisionPattern:
    """Identified pattern in decisions."""
    pattern_id: str
    signature: str
    decision_ids: List[str]
    avg_pnl: float
    win_rate: float
    frequency: int
    last_seen: float
    context_fingerprint: Dict[str, Any]


class DecisionMemoryExpanded:
    """
    AGI Ultra: Expanded Decision Memory System.
    
    Features:
    - Complete decision tracking
    - Retrospective analysis
    - Pattern identification
    - Pre-execution prediction
    """
    
    def __init__(
        self,
        max_history: int = 10000,
        pattern_min_frequency: int = 5
    ):
        self.max_history = max_history
        self.pattern_min_frequency = pattern_min_frequency
        
        # Decision storage
        self.decisions: Dict[str, DecisionRecord] = {}
        self.decision_order: deque = deque(maxlen=max_history)
        
        # Pattern storage
        self.patterns: Dict[str, DecisionPattern] = {}
        
        # Indexes
        self.by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.by_outcome: Dict[DecisionOutcome, List[str]] = defaultdict(list)
        self.by_signature: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.total_decisions = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        
        logger.info("DecisionMemoryExpanded initialized")
    
    # -------------------------------------------------------------------------
    # RECORDING
    # -------------------------------------------------------------------------
    def record_decision(
        self,
        decision: str,
        symbol: str,
        timeframe: str,
        confidence: float,
        market_context: Dict[str, Any],
        module_votes: Dict[str, str],
        reasoning_chain: List[str],
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        size: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Record a new decision.
        
        Returns:
            Decision ID
        """
        decision_id = f"dec:{int(time.time() * 1000)}:{symbol}"
        
        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=time.time(),
            decision=decision,
            symbol=symbol,
            timeframe=timeframe,
            confidence=confidence,
            market_context=market_context,
            module_votes=module_votes,
            reasoning_chain=reasoning_chain,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            tags=tags or []
        )
        
        # Store
        self.decisions[decision_id] = record
        self.decision_order.append(decision_id)
        
        # Index
        self.by_symbol[symbol].append(decision_id)
        self.by_signature[record.get_signature()].append(decision_id)
        
        self.total_decisions += 1
        
        # Pattern detection
        self._detect_patterns(record)
        
        logger.debug(f"Decision recorded: {decision_id}")
        return decision_id
    
    def update_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        duration_seconds: float
    ):
        """Update decision with final outcome."""
        if decision_id not in self.decisions:
            logger.warning(f"Decision not found: {decision_id}")
            return
        
        record = self.decisions[decision_id]
        record.outcome = outcome
        record.exit_price = exit_price
        record.pnl = pnl
        record.pnl_pct = pnl_pct
        record.duration_seconds = duration_seconds
        
        # Update indexes
        self.by_outcome[outcome].append(decision_id)
        
        # Update statistics
        self.total_pnl += pnl
        if outcome == DecisionOutcome.WIN:
            self.win_count += 1
        elif outcome == DecisionOutcome.LOSS:
            self.loss_count += 1
        
        # Update patterns
        self._update_pattern_stats(record)
    
    # -------------------------------------------------------------------------
    # RETROSPECTIVE ANALYSIS
    # -------------------------------------------------------------------------
    def analyze_decision(self, decision_id: str) -> Dict[str, Any]:
        """
        Deep retrospective analysis of a decision.
        
        Analyzes what went right/wrong and how to improve.
        """
        if decision_id not in self.decisions:
            return {'error': 'Decision not found'}
        
        record = self.decisions[decision_id]
        
        # Find similar decisions
        similar = self.find_similar_decisions(record)
        
        # Calculate pattern match
        pattern = self.patterns.get(record.get_signature())
        
        # Analyze vote agreement
        votes = list(record.module_votes.values())
        vote_agreement = votes.count(record.decision) / len(votes) if votes else 0
        
        # Retrospective insights
        insights = []
        
        if record.outcome == DecisionOutcome.LOSS:
            # Analyze why it failed
            if record.confidence < 0.6:
                insights.append("Low confidence entry - consider waiting for stronger signals")
            
            if vote_agreement < 0.5:
                insights.append("Module disagreement was high - consensus not reached")
            
            if pattern and pattern.win_rate < 0.4:
                insights.append(f"Pattern has low win rate ({pattern.win_rate:.1%}) - avoid similar setups")
        
        elif record.outcome == DecisionOutcome.WIN:
            # Analyze why it succeeded
            if record.confidence > 0.8:
                insights.append("High confidence entries tend to work well")
            
            if vote_agreement > 0.7:
                insights.append("Strong module consensus led to success")
        
        return {
            'decision_id': decision_id,
            'decision': record.decision,
            'outcome': record.outcome.value,
            'pnl': record.pnl,
            'pnl_pct': record.pnl_pct,
            'duration': record.duration_seconds,
            'confidence': record.confidence,
            'vote_agreement': vote_agreement,
            'pattern_match': pattern.pattern_id if pattern else None,
            'pattern_win_rate': pattern.win_rate if pattern else None,
            'similar_decisions': len(similar),
            'similar_win_rate': sum(1 for s in similar if s.outcome == DecisionOutcome.WIN) / len(similar) if similar else 0,
            'insights': insights,
            'reasoning_chain': record.reasoning_chain
        }
    
    def find_similar_decisions(
        self,
        record: DecisionRecord,
        max_results: int = 20
    ) -> List[DecisionRecord]:
        """Find similar historical decisions."""
        signature = record.get_signature()
        similar_ids = self.by_signature.get(signature, [])
        
        similar = []
        for did in similar_ids[-max_results:]:
            if did != record.decision_id and did in self.decisions:
                similar.append(self.decisions[did])
        
        return similar
    
    # -------------------------------------------------------------------------
    # PATTERN DETECTION
    # -------------------------------------------------------------------------
    def _detect_patterns(self, record: DecisionRecord):
        """Detect patterns from new decision."""
        signature = record.get_signature()
        
        if signature not in self.patterns:
            # Check if we have enough history
            sig_decisions = self.by_signature.get(signature, [])
            
            if len(sig_decisions) >= self.pattern_min_frequency:
                # Create new pattern
                self._create_pattern(signature, sig_decisions)
    
    def _create_pattern(self, signature: str, decision_ids: List[str]):
        """Create a new pattern from decision cluster."""
        decisions = [self.decisions[did] for did in decision_ids if did in self.decisions]
        
        # Calculate statistics
        completed = [d for d in decisions if d.outcome != DecisionOutcome.PENDING]
        
        if not completed:
            return
        
        wins = sum(1 for d in completed if d.outcome == DecisionOutcome.WIN)
        total_pnl = sum(d.pnl for d in completed)
        
        # Extract context fingerprint
        sample = decisions[0]
        fingerprint = {
            'decision': sample.decision,
            'regime': sample.market_context.get('regime', 'unknown'),
            'symbol': sample.symbol
        }
        
        pattern = DecisionPattern(
            pattern_id=f"pat:{signature}",
            signature=signature,
            decision_ids=decision_ids,
            avg_pnl=total_pnl / len(completed),
            win_rate=wins / len(completed),
            frequency=len(decisions),
            last_seen=time.time(),
            context_fingerprint=fingerprint
        )
        
        self.patterns[signature] = pattern
        logger.info(f"Pattern detected: {pattern.pattern_id} (win_rate={pattern.win_rate:.1%})")
    
    def _update_pattern_stats(self, record: DecisionRecord):
        """Update pattern statistics after outcome."""
        signature = record.get_signature()
        
        if signature in self.patterns:
            pattern = self.patterns[signature]
            
            # Recalculate stats
            decisions = [self.decisions[did] for did in pattern.decision_ids if did in self.decisions]
            completed = [d for d in decisions if d.outcome != DecisionOutcome.PENDING]
            
            if completed:
                wins = sum(1 for d in completed if d.outcome == DecisionOutcome.WIN)
                pattern.win_rate = wins / len(completed)
                pattern.avg_pnl = sum(d.pnl for d in completed) / len(completed)
                pattern.last_seen = time.time()
    
    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    def predict_outcome(
        self,
        decision: str,
        symbol: str,
        market_context: Dict[str, Any],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Predict outcome before execution based on historical patterns.
        
        Returns:
            Prediction with probability and expected PnL
        """
        # Create temporary signature
        key = f"{decision}:{symbol}:{market_context.get('regime', 'unknown')}"
        signature = hashlib.md5(key.encode()).hexdigest()[:12]
        
        # Look for matching pattern
        pattern = self.patterns.get(signature)
        
        if pattern and pattern.frequency >= self.pattern_min_frequency:
            # We have pattern data
            predicted_win_prob = pattern.win_rate
            expected_pnl = pattern.avg_pnl
            
            # Adjust by confidence
            confidence_factor = (confidence - 0.5) * 0.4  # ±20% adjustment
            predicted_win_prob = min(1.0, max(0.0, predicted_win_prob + confidence_factor))
            
            prediction = {
                'has_pattern': True,
                'pattern_id': pattern.pattern_id,
                'predicted_win_prob': predicted_win_prob,
                'expected_pnl': expected_pnl,
                'pattern_frequency': pattern.frequency,
                'recommendation': 'PROCEED' if predicted_win_prob > 0.5 else 'CAUTION'
            }
        else:
            # No pattern - use global stats
            global_win_rate = self.win_count / max(1, self.win_count + self.loss_count)
            
            prediction = {
                'has_pattern': False,
                'predicted_win_prob': global_win_rate,
                'expected_pnl': self.total_pnl / max(1, self.total_decisions),
                'recommendation': 'UNKNOWN'
            }
        
        logger.debug(f"Prediction: {prediction['recommendation']} (win_prob={prediction['predicted_win_prob']:.1%})")
        return prediction
    
    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    def get_recent_decisions(self, limit: int = 50) -> List[DecisionRecord]:
        """Get recent decisions."""
        recent_ids = list(self.decision_order)[-limit:]
        return [self.decisions[did] for did in recent_ids if did in self.decisions]
    
    def get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance statistics for a symbol."""
        decision_ids = self.by_symbol.get(symbol, [])
        decisions = [self.decisions[did] for did in decision_ids if did in self.decisions]
        
        if not decisions:
            return {'error': 'No decisions for symbol'}
        
        completed = [d for d in decisions if d.outcome != DecisionOutcome.PENDING]
        wins = sum(1 for d in completed if d.outcome == DecisionOutcome.WIN)
        total_pnl = sum(d.pnl for d in completed)
        
        return {
            'symbol': symbol,
            'total_decisions': len(decisions),
            'completed': len(completed),
            'win_count': wins,
            'win_rate': wins / len(completed) if completed else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(completed) if completed else 0
        }
    
    def get_pattern_report(self) -> List[Dict[str, Any]]:
        """Get report of all identified patterns."""
        report = []
        
        for pattern in sorted(self.patterns.values(), key=lambda p: -p.frequency):
            report.append({
                'pattern_id': pattern.pattern_id,
                'frequency': pattern.frequency,
                'win_rate': pattern.win_rate,
                'avg_pnl': pattern.avg_pnl,
                'last_seen_hours_ago': (time.time() - pattern.last_seen) / 3600,
                'context': pattern.context_fingerprint
            })
        
        return report
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            'total_decisions': self.total_decisions,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': self.win_count / max(1, self.win_count + self.loss_count),
            'total_pnl': self.total_pnl,
            'avg_pnl': self.total_pnl / max(1, self.total_decisions),
            'patterns_identified': len(self.patterns),
            'symbols_traded': len(self.by_symbol)
        }
    
    def save(self, filepath: str):
        """Save decision memory to file."""
        data = {
            'decisions': self.decisions,
            'patterns': self.patterns,
            'stats': self.get_statistics()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Decision memory saved to {filepath}")
    
    def load(self, filepath: str):
        """Load decision memory from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.decisions = data.get('decisions', {})
            self.patterns = data.get('patterns', {})
            
            # Rebuild indexes
            self.decision_order.clear()
            self.by_symbol.clear()
            self.by_outcome.clear()
            self.by_signature.clear()
            
            for dec_id, record in sorted(self.decisions.items(), key=lambda x: x[1].timestamp):
                self.decision_order.append(dec_id)
                self.by_symbol[record.symbol].append(dec_id)
                self.by_outcome[record.outcome].append(dec_id)
                self.by_signature[record.get_signature()].append(dec_id)
            
            logger.info(f"Decision memory loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load decision memory: {e}")


# Global instance
_global_memory: Optional[DecisionMemoryExpanded] = None

def get_decision_memory() -> DecisionMemoryExpanded:
    """Get global decision memory instance."""
    global _global_memory
    if _global_memory is None:
        _global_memory = DecisionMemoryExpanded()
    return _global_memory
