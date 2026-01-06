"""
AGI Fase 2: Intuition and Dual Process System

Intuição e Processamento Inconsciente:
- Intuição Artificial
- Processamento Dual (Sistema 1 e 2)
- Inconsciente Coletivo
- Gut Feelings
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("Intuition")


class IntuitionStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    COMPELLING = "compelling"


@dataclass
class GutFeeling:
    """An intuitive feeling about a decision."""
    feeling_id: str
    subject: str
    valence: float  # -1 (bad) to 1 (good)
    strength: IntuitionStrength
    basis: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntuitivePattern:
    """A pattern recognized intuitively."""
    pattern_id: str
    features: Dict[str, float]
    association: str
    confidence: float
    times_activated: int = 0


class IntuitionEngine:
    """Rapid intuitive processing."""
    
    def __init__(self, experience_weight: float = 0.7):
        self.experience_weight = experience_weight
        self.patterns: Dict[str, IntuitivePattern] = {}
        self.feelings: List[GutFeeling] = []
        self.experience_base: deque = deque(maxlen=10000)
        self._feeling_counter = 0
        
        logger.info("IntuitionEngine initialized")
    
    def learn_pattern(self, features: Dict[str, float], outcome: str):
        """Learn a pattern from experience."""
        pattern_key = self._hash_features(features)
        
        if pattern_key in self.patterns:
            pat = self.patterns[pattern_key]
            pat.times_activated += 1
            pat.confidence = min(0.95, pat.confidence + 0.05)
        else:
            self.patterns[pattern_key] = IntuitivePattern(
                pattern_id=pattern_key,
                features=features,
                association=outcome,
                confidence=0.5
            )
        
        self.experience_base.append({
            'features': features,
            'outcome': outcome,
            'timestamp': time.time()
        })
    
    def _hash_features(self, features: Dict[str, float]) -> str:
        """Create pattern hash from features."""
        sorted_items = sorted(features.items())
        return str(hash(str(sorted_items)))[:12]
    
    def intuit(self, situation: Dict[str, float]) -> GutFeeling:
        """Generate gut feeling about situation."""
        pattern_key = self._hash_features(situation)
        
        if pattern_key in self.patterns:
            pat = self.patterns[pattern_key]
            pat.times_activated += 1
            
            valence = 1.0 if 'positive' in pat.association.lower() else -0.5
            strength = IntuitionStrength.STRONG if pat.confidence > 0.7 else IntuitionStrength.MODERATE
            basis = f"Recognized pattern from {pat.times_activated} experiences"
        else:
            valence = self._aggregate_similar(situation)
            strength = IntuitionStrength.WEAK
            basis = "Aggregated from similar experiences"
        
        self._feeling_counter += 1
        feeling = GutFeeling(
            feeling_id=f"gut_{self._feeling_counter}",
            subject=str(list(situation.keys())[:3]),
            valence=valence,
            strength=strength,
            basis=basis
        )
        
        self.feelings.append(feeling)
        return feeling
    
    def _aggregate_similar(self, situation: Dict[str, float]) -> float:
        """Aggregate evidence from similar situations."""
        if not self.experience_base:
            return 0.0
        
        total = 0.0
        count = 0
        
        for exp in list(self.experience_base)[-100:]:
            similarity = self._calculate_similarity(situation, exp['features'])
            if similarity > 0.5:
                outcome_val = 1.0 if 'positive' in str(exp['outcome']).lower() else -0.5
                total += outcome_val * similarity
                count += 1
        
        return total / count if count > 0 else 0.0
    
    def _calculate_similarity(self, f1: Dict, f2: Dict) -> float:
        """Calculate feature similarity."""
        common = set(f1.keys()) & set(f2.keys())
        if not common:
            return 0.0
        
        diffs = [abs(f1[k] - f2[k]) for k in common]
        avg_diff = sum(diffs) / len(diffs)
        
        return max(0, 1 - avg_diff)


class System1:
    """Fast, intuitive, automatic processing."""
    
    def __init__(self, intuition: IntuitionEngine):
        self.intuition = intuition
        self.heuristics: Dict[str, Callable] = {}
        self.response_time_ms = 10
        
        logger.info("System1 initialized")
    
    def process(self, input_data: Dict) -> Tuple[str, float]:
        """Fast heuristic processing."""
        start = time.time()
        
        feeling = self.intuition.intuit(input_data)
        
        if feeling.valence > 0.5:
            decision = "APPROVE"
        elif feeling.valence < -0.3:
            decision = "REJECT"
        else:
            decision = "UNCERTAIN"
        
        confidence = abs(feeling.valence)
        
        return decision, confidence
    
    def add_heuristic(self, name: str, heuristic: Callable):
        """Add a fast heuristic."""
        self.heuristics[name] = heuristic


class System2:
    """Slow, deliberative, conscious processing."""
    
    def __init__(self):
        self.reasoning_steps: List[Dict] = []
        self.response_time_ms = 500
        
        logger.info("System2 initialized")
    
    def process(self, input_data: Dict, system1_result: Tuple[str, float]) -> Tuple[str, float]:
        """Deliberative analysis."""
        s1_decision, s1_confidence = system1_result
        
        self.reasoning_steps = []
        
        self.reasoning_steps.append({
            'step': 'review_intuition',
            'system1_decision': s1_decision,
            'system1_confidence': s1_confidence
        })
        
        if s1_confidence > 0.8:
            self.reasoning_steps.append({'step': 'accept_intuition'})
            return s1_decision, s1_confidence
        
        self.reasoning_steps.append({'step': 'deeper_analysis'})
        
        pros = sum(1 for v in input_data.values() if isinstance(v, (int, float)) and v > 0.5)
        cons = sum(1 for v in input_data.values() if isinstance(v, (int, float)) and v < 0.3)
        
        if pros > cons:
            decision = "APPROVE"
            confidence = 0.6 + (pros - cons) * 0.05
        elif cons > pros:
            decision = "REJECT"
            confidence = 0.6 + (cons - pros) * 0.05
        else:
            decision = "WAIT"
            confidence = 0.5
        
        self.reasoning_steps.append({
            'step': 'conclusion',
            'decision': decision,
            'confidence': confidence
        })
        
        return decision, min(0.95, confidence)


class CollectiveUnconscious:
    """Shared unconscious patterns and archetypes."""
    
    def __init__(self):
        self.archetypes: Dict[str, Dict] = {}
        self.shared_memory: deque = deque(maxlen=5000)
        self.synchronizations: List[Dict] = []
        
        self._init_archetypes()
        logger.info("CollectiveUnconscious initialized")
    
    def _init_archetypes(self):
        """Initialize market archetypes."""
        self.archetypes = {
            'bull': {'direction': 'up', 'emotion': 'greed', 'behavior': 'buying'},
            'bear': {'direction': 'down', 'emotion': 'fear', 'behavior': 'selling'},
            'consolidation': {'direction': 'sideways', 'emotion': 'uncertainty', 'behavior': 'waiting'},
            'breakout': {'direction': 'explosive', 'emotion': 'excitement', 'behavior': 'chasing'},
            'reversal': {'direction': 'changing', 'emotion': 'surprise', 'behavior': 'repositioning'},
        }
    
    def match_archetype(self, features: Dict[str, float]) -> Optional[str]:
        """Match current situation to archetype."""
        direction = features.get('trend', 0)
        volatility = features.get('volatility', 0.5)
        
        if direction > 0.5:
            return 'bull'
        elif direction < -0.5:
            return 'bear'
        elif volatility > 0.8:
            return 'breakout'
        elif volatility < 0.2:
            return 'consolidation'
        
        return None
    
    def contribute(self, module_id: str, insight: Dict):
        """Contribute to collective unconscious."""
        self.shared_memory.append({
            'source': module_id,
            'insight': insight,
            'timestamp': time.time()
        })
    
    def synchronize(self) -> Dict[str, float]:
        """Synchronize across modules."""
        if not self.shared_memory:
            return {}
        
        recent = list(self.shared_memory)[-50:]
        
        consensus = defaultdict(list)
        for item in recent:
            for key, value in item['insight'].items():
                if isinstance(value, (int, float)):
                    consensus[key].append(value)
        
        synchronized = {}
        for key, values in consensus.items():
            synchronized[key] = sum(values) / len(values)
        
        self.synchronizations.append({
            'timestamp': time.time(),
            'result': synchronized
        })
        
        return synchronized


class DualProcessSystem:
    """Main Dual Process System."""
    
    def __init__(self):
        self.intuition = IntuitionEngine()
        self.system1 = System1(self.intuition)
        self.system2 = System2()
        self.collective = CollectiveUnconscious()
        
        self.balance = 0.5  # 0 = all S1, 1 = all S2
        
        logger.info("DualProcessSystem initialized")
    
    def process(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Process through dual systems."""
        s1_decision, s1_confidence = self.system1.process(input_data)
        
        if s1_confidence > 0.85 and self.balance < 0.7:
            return {
                'decision': s1_decision,
                'confidence': s1_confidence,
                'system_used': 'System1',
                'intuitive': True
            }
        
        s2_decision, s2_confidence = self.system2.process(input_data, (s1_decision, s1_confidence))
        
        if s1_decision == s2_decision:
            final_confidence = (s1_confidence + s2_confidence) / 2 + 0.1
        else:
            final_confidence = max(s1_confidence, s2_confidence) * 0.8
            
        final_decision = s2_decision if s2_confidence > s1_confidence else s1_decision
        
        return {
            'decision': final_decision,
            'confidence': min(0.95, final_confidence),
            'system1': {'decision': s1_decision, 'confidence': s1_confidence},
            'system2': {'decision': s2_decision, 'confidence': s2_confidence},
            'system_used': 'Both',
            'intuitive': False
        }
    
    def learn(self, situation: Dict[str, float], outcome: str):
        """Learn from outcome."""
        self.intuition.learn_pattern(situation, outcome)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'intuitive_patterns': len(self.intuition.patterns),
            'experience_base': len(self.intuition.experience_base),
            'gut_feelings': len(self.intuition.feelings),
            'archetypes': len(self.collective.archetypes),
            'shared_memory': len(self.collective.shared_memory),
            'balance': self.balance
        }
