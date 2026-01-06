"""
AGI Fase 2: Exploration and Discovery System

Exploração Criativa e Descoberta:
- Exploração Ativa
- Sistema de Hipóteses
- Descoberta Científica Autônoma
- Curiosidade Artificial
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("Exploration")


class ExplorationMode(Enum):
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    BALANCED = "balanced"


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"


@dataclass
class ExplorationTarget:
    """A target for exploration."""
    target_id: str
    area: str
    novelty: float
    information_gain: float
    explored: bool = False


@dataclass
class Hypothesis:
    """A scientific hypothesis."""
    hypothesis_id: str
    statement: str
    predictions: List[str]
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    tests: List[Dict] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class Discovery:
    """A discovered pattern or law."""
    discovery_id: str
    discovery_type: str
    description: str
    evidence: List[str]
    confidence: float
    discovered_at: float = field(default_factory=time.time)


class CuriosityEngine:
    """Artificial curiosity for exploration."""
    
    def __init__(self, curiosity_level: float = 0.5):
        self.curiosity_level = curiosity_level
        self.interests: Dict[str, float] = {}
        self.explored: List[str] = []
        
        logger.info("CuriosityEngine initialized")
    
    def get_curiosity(self, area: str) -> float:
        """Get curiosity level for an area."""
        base = self.curiosity_level
        
        if area in self.interests:
            base *= self.interests[area]
        
        if area in self.explored:
            base *= 0.5
        
        return base
    
    def update_interest(self, area: str, outcome: float):
        """Update interest based on outcome."""
        if area not in self.interests:
            self.interests[area] = 1.0
        
        if outcome > 0.5:
            self.interests[area] = min(2.0, self.interests[area] * 1.2)
        else:
            self.interests[area] = max(0.3, self.interests[area] * 0.9)
    
    def mark_explored(self, area: str):
        """Mark area as explored."""
        if area not in self.explored:
            self.explored.append(area)
    
    def get_most_curious(self, areas: List[str]) -> str:
        """Get area with highest curiosity."""
        if not areas:
            return ""
        return max(areas, key=lambda a: self.get_curiosity(a))


class ActiveExplorer:
    """Active exploration of unknown spaces."""
    
    def __init__(self, curiosity: CuriosityEngine):
        self.curiosity = curiosity
        self.exploration_history: List[Dict] = []
        self.mode = ExplorationMode.BALANCED
        self.targets: List[ExplorationTarget] = []
        self._target_counter = 0
        
        logger.info("ActiveExplorer initialized")
    
    def identify_targets(self, known_areas: List[str], potential_areas: List[str]) -> List[ExplorationTarget]:
        """Identify exploration targets."""
        targets = []
        
        for area in potential_areas:
            if area not in known_areas:
                self._target_counter += 1
                novelty = 1.0 if area not in known_areas else 0.3
                info_gain = self.curiosity.get_curiosity(area)
                
                target = ExplorationTarget(
                    target_id=f"target_{self._target_counter}",
                    area=area,
                    novelty=novelty,
                    information_gain=info_gain
                )
                targets.append(target)
        
        self.targets = sorted(targets, key=lambda t: t.novelty * t.information_gain, reverse=True)
        return self.targets
    
    def select_action(self, exploit_value: float, explore_value: float) -> str:
        """Select between exploitation and exploration."""
        if self.mode == ExplorationMode.EXPLOIT:
            return "exploit"
        elif self.mode == ExplorationMode.EXPLORE:
            return "explore"
        else:
            epsilon = self.curiosity.curiosity_level
            if random.random() < epsilon:
                return "explore"
            return "exploit" if exploit_value > explore_value else "explore"
    
    def explore(self, target: ExplorationTarget) -> Dict[str, Any]:
        """Explore a target."""
        result = {
            'target': target.target_id,
            'area': target.area,
            'timestamp': time.time(),
            'findings': []
        }
        
        outcome = random.random()
        
        if outcome > 0.5:
            result['findings'].append(f"Interesting pattern in {target.area}")
        
        target.explored = True
        self.curiosity.mark_explored(target.area)
        self.curiosity.update_interest(target.area, outcome)
        self.exploration_history.append(result)
        
        return result


class HypothesisSystem:
    """Scientific hypothesis management."""
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self._hyp_counter = 0
        
        logger.info("HypothesisSystem initialized")
    
    def formulate(self, observation: str, explanation: str) -> Hypothesis:
        """Formulate a hypothesis."""
        self._hyp_counter += 1
        
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{self._hyp_counter}",
            statement=f"If {observation} then {explanation}",
            predictions=[explanation]
        )
        
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        return hypothesis
    
    def design_test(self, hypothesis_id: str) -> Dict[str, Any]:
        """Design a test for hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return {}
        
        hyp = self.hypotheses[hypothesis_id]
        
        return {
            'hypothesis_id': hypothesis_id,
            'test_type': 'prediction_test',
            'predictions_to_test': hyp.predictions,
            'control': 'baseline',
            'treatment': hyp.statement
        }
    
    def record_result(self, hypothesis_id: str, test_result: Dict):
        """Record test result."""
        if hypothesis_id not in self.hypotheses:
            return
        
        hyp = self.hypotheses[hypothesis_id]
        hyp.tests.append(test_result)
        
        supports = sum(1 for t in hyp.tests if t.get('supports', False))
        total = len(hyp.tests)
        
        hyp.confidence = supports / total if total > 0 else 0.5
        
        if hyp.confidence > 0.7:
            hyp.status = HypothesisStatus.SUPPORTED
        elif hyp.confidence < 0.3:
            hyp.status = HypothesisStatus.REFUTED
        else:
            hyp.status = HypothesisStatus.UNCERTAIN
    
    def get_supported(self) -> List[Hypothesis]:
        """Get supported hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.SUPPORTED]


class ScientificDiscovery:
    """Autonomous scientific discovery."""
    
    def __init__(self):
        self.discoveries: List[Discovery] = []
        self.theories: Dict[str, Dict] = {}
        self.laws: List[Dict] = []
        self._disc_counter = 0
        
        logger.info("ScientificDiscovery initialized")
    
    def discover_pattern(self, data: List[Dict], variable: str) -> Optional[Discovery]:
        """Discover patterns in data."""
        if len(data) < 5:
            return None
        
        values = [d.get(variable, 0) for d in data if variable in d]
        if not values:
            return None
        
        avg = sum(values) / len(values)
        trend = values[-1] - values[0] if len(values) > 1 else 0
        
        if abs(trend) > avg * 0.1:
            self._disc_counter += 1
            discovery = Discovery(
                discovery_id=f"disc_{self._disc_counter}",
                discovery_type="trend",
                description=f"{variable} shows {'upward' if trend > 0 else 'downward'} trend",
                evidence=[f"Change: {trend:.2f}"],
                confidence=0.6
            )
            self.discoveries.append(discovery)
            return discovery
        
        return None
    
    def form_theory(self, discoveries: List[Discovery]) -> Dict:
        """Form theory from discoveries."""
        if not discoveries:
            return {}
        
        theory_id = f"theory_{len(self.theories)}"
        
        theory = {
            'id': theory_id,
            'based_on': [d.discovery_id for d in discoveries],
            'statement': f"Theory combining {len(discoveries)} discoveries",
            'confidence': sum(d.confidence for d in discoveries) / len(discoveries),
            'tested': False
        }
        
        self.theories[theory_id] = theory
        return theory
    
    def discover_law(self, pattern: str, conditions: List[str]) -> Dict:
        """Discover a market law."""
        law = {
            'pattern': pattern,
            'conditions': conditions,
            'reliability': 0.5,
            'observations': 1,
            'discovered_at': time.time()
        }
        
        self.laws.append(law)
        return law


class ExplorationSystem:
    """Main Exploration and Discovery System."""
    
    def __init__(self):
        self.curiosity = CuriosityEngine()
        self.explorer = ActiveExplorer(self.curiosity)
        self.hypothesis = HypothesisSystem()
        self.discovery = ScientificDiscovery()
        
        logger.info("ExplorationSystem initialized")
    
    def scientific_method(self, observation: str, explanation: str, data: List[Dict]) -> Dict:
        """Apply scientific method."""
        hyp = self.hypothesis.formulate(observation, explanation)
        
        test = self.hypothesis.design_test(hyp.hypothesis_id)
        
        discovery = self.discovery.discover_pattern(data, observation)
        
        supports = discovery is not None
        self.hypothesis.record_result(hyp.hypothesis_id, {'supports': supports})
        
        return {
            'hypothesis': hyp.hypothesis_id,
            'test': test,
            'discovery': discovery.discovery_id if discovery else None,
            'result': 'supported' if supports else 'uncertain'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'exploration_mode': self.explorer.mode.value,
            'areas_explored': len(self.curiosity.explored),
            'hypotheses': len(self.hypothesis.hypotheses),
            'supported_hypotheses': len(self.hypothesis.get_supported()),
            'discoveries': len(self.discovery.discoveries),
            'theories': len(self.discovery.theories),
            'laws': len(self.discovery.laws)
        }
