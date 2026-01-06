"""
AGI Fase 2: Continuous Learning System

Sistema de Auto-Aprendizado Contínuo:
- Aprendizado Sem Supervisão
- Aprendizado Ativo
- Aprendizado por Observação
- Aprendizado por Experimentação
"""

import logging
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict

logger = logging.getLogger("ContinuousLearning")


@dataclass
class LearningExample:
    """A single learning example."""
    example_id: str
    features: Dict[str, float]
    label: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    confidence: float = 1.0


@dataclass
class Experiment:
    """A learning experiment."""
    experiment_id: str
    hypothesis: str
    action: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    success: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LearnedConcept:
    """A concept learned from data."""
    name: str
    features: Dict[str, float]
    confidence: float
    examples_seen: int
    last_updated: float


class UnsupervisedLearner:
    """
    Learns patterns without explicit labels.
    
    Uses clustering and pattern discovery.
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.cluster_centers: List[Dict[str, float]] = []
        self.cluster_counts: List[int] = [0] * n_clusters
        
        self.patterns_discovered: List[Dict] = []
        
        logger.info("UnsupervisedLearner initialized")
    
    def learn(self, example: LearningExample):
        """Learn from an unlabeled example."""
        if not self.cluster_centers:
            # Initialize with first examples
            self.cluster_centers.append(example.features.copy())
            self.cluster_counts[0] = 1
            return
        
        # Find nearest cluster
        nearest_idx = self._find_nearest_cluster(example.features)
        
        if nearest_idx is not None:
            # Update cluster center (online k-means)
            count = self.cluster_counts[nearest_idx]
            for key, value in example.features.items():
                if key in self.cluster_centers[nearest_idx]:
                    old = self.cluster_centers[nearest_idx][key]
                    self.cluster_centers[nearest_idx][key] = (old * count + value) / (count + 1)
            self.cluster_counts[nearest_idx] += 1
        elif len(self.cluster_centers) < self.n_clusters:
            # Create new cluster
            self.cluster_centers.append(example.features.copy())
            self.cluster_counts[len(self.cluster_centers) - 1] = 1
    
    def _find_nearest_cluster(self, features: Dict[str, float]) -> Optional[int]:
        """Find nearest cluster center."""
        if not self.cluster_centers:
            return None
        
        min_dist = float('inf')
        nearest = 0
        
        for i, center in enumerate(self.cluster_centers):
            dist = sum(
                (features.get(k, 0) - center.get(k, 0)) ** 2
                for k in set(features.keys()) | set(center.keys())
            ) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest = i
        
        return nearest
    
    def discover_patterns(self) -> List[Dict]:
        """Discover patterns from clusters."""
        patterns = []
        
        for i, center in enumerate(self.cluster_centers):
            if self.cluster_counts[i] >= 5:  # Minimum examples
                pattern = {
                    'cluster_id': i,
                    'center': center,
                    'count': self.cluster_counts[i],
                    'dominant_features': sorted(
                        center.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:3]
                }
                patterns.append(pattern)
        
        self.patterns_discovered = patterns
        return patterns


class ActiveLearner:
    """
    Actively identifies what needs to be learned.
    
    Seeks out informative examples.
    """
    
    def __init__(self):
        self.uncertainty_threshold = 0.3
        self.learning_queue: deque = deque(maxlen=100)
        self.concepts: Dict[str, LearnedConcept] = {}
        
        logger.info("ActiveLearner initialized")
    
    def identify_learning_needs(
        self,
        current_performance: Dict[str, float]
    ) -> List[str]:
        """Identify what needs to be learned."""
        needs = []
        
        for area, score in current_performance.items():
            if score < self.uncertainty_threshold:
                needs.append(area)
        
        # Check for gaps in concepts
        for concept_name, concept in self.concepts.items():
            if concept.confidence < 0.5:
                needs.append(f"reinforce_{concept_name}")
        
        return needs
    
    def request_example(self, area: str) -> Dict[str, Any]:
        """Request an informative example for an area."""
        request = {
            'area': area,
            'type': 'informative',
            'priority': 0.8,
            'timestamp': time.time()
        }
        
        self.learning_queue.append(request)
        return request
    
    def update_concept(
        self,
        name: str,
        features: Dict[str, float],
        confidence_delta: float = 0.1
    ):
        """Update a learned concept."""
        if name not in self.concepts:
            self.concepts[name] = LearnedConcept(
                name=name,
                features=features,
                confidence=0.5,
                examples_seen=0,
                last_updated=time.time()
            )
        
        concept = self.concepts[name]
        
        # Update features (running average)
        for key, value in features.items():
            if key in concept.features:
                concept.features[key] = (concept.features[key] + value) / 2
            else:
                concept.features[key] = value
        
        concept.confidence = min(1.0, concept.confidence + confidence_delta)
        concept.examples_seen += 1
        concept.last_updated = time.time()


class ObservationalLearner:
    """
    Learns by observing other agents/systems.
    
    Imitation learning.
    """
    
    def __init__(self):
        self.observed_behaviors: List[Dict] = []
        self.learned_behaviors: Dict[str, Dict] = {}
        
        logger.info("ObservationalLearner initialized")
    
    def observe(
        self,
        agent_id: str,
        state: Dict[str, Any],
        action: str,
        result: float
    ):
        """Observe an agent's behavior."""
        observation = {
            'agent_id': agent_id,
            'state': state,
            'action': action,
            'result': result,
            'timestamp': time.time()
        }
        
        self.observed_behaviors.append(observation)
    
    def extract_policy(self, agent_id: Optional[str] = None) -> Dict[str, str]:
        """Extract a policy from observations."""
        # Filter observations
        relevant = self.observed_behaviors
        if agent_id:
            relevant = [o for o in relevant if o['agent_id'] == agent_id]
        
        if not relevant:
            return {}
        
        # Group by state features
        state_action_results = defaultdict(list)
        
        for obs in relevant:
            # Simplify state to key
            state_key = str(sorted(obs['state'].items()))
            state_action_results[state_key].append((obs['action'], obs['result']))
        
        # Extract best action for each state
        policy = {}
        for state_key, actions in state_action_results.items():
            # Find action with best average result
            action_results = defaultdict(list)
            for action, result in actions:
                action_results[action].append(result)
            
            best_action = max(
                action_results.items(),
                key=lambda x: sum(x[1]) / len(x[1])
            )[0]
            
            policy[state_key] = best_action
        
        return policy
    
    def imitate(self, agent_id: str) -> Dict[str, Any]:
        """Learn to imitate a specific agent."""
        policy = self.extract_policy(agent_id)
        
        if not policy:
            return {'success': False, 'reason': 'No observations'}
        
        self.learned_behaviors[agent_id] = {
            'policy': policy,
            'learned_at': time.time(),
            'observations_used': len([
                o for o in self.observed_behaviors
                if o['agent_id'] == agent_id
            ])
        }
        
        return {'success': True, 'policy_size': len(policy)}


class ExperimentalLearner:
    """
    Learns through controlled experimentation.
    
    Scientific method for learning.
    """
    
    def __init__(self):
        self.experiments: List[Experiment] = []
        self.hypotheses: Dict[str, Dict] = {}
        self._exp_counter = 0
        
        logger.info("ExperimentalLearner initialized")
    
    def formulate_hypothesis(
        self,
        observation: str,
        proposed_cause: str
    ) -> Dict[str, Any]:
        """Formulate a hypothesis to test."""
        hypothesis = {
            'id': f"hyp_{len(self.hypotheses)}",
            'observation': observation,
            'proposed_cause': proposed_cause,
            'status': 'untested',
            'confidence': 0.5,
            'tests': []
        }
        
        self.hypotheses[hypothesis['id']] = hypothesis
        return hypothesis
    
    def design_experiment(
        self,
        hypothesis_id: str,
        control_condition: Dict,
        test_condition: Dict
    ) -> Experiment:
        """Design an experiment to test a hypothesis."""
        self._exp_counter += 1
        
        hypothesis = self.hypotheses.get(hypothesis_id)
        
        experiment = Experiment(
            experiment_id=f"exp_{self._exp_counter}",
            hypothesis=hypothesis_id,
            action=str(test_condition),
            expected_outcome="Support hypothesis" if hypothesis else "Unknown"
        )
        
        self.experiments.append(experiment)
        return experiment
    
    def record_result(
        self,
        experiment_id: str,
        outcome: str,
        success: bool
    ):
        """Record experiment result."""
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                exp.actual_outcome = outcome
                exp.success = success
                
                # Update hypothesis
                if exp.hypothesis in self.hypotheses:
                    hyp = self.hypotheses[exp.hypothesis]
                    hyp['tests'].append({
                        'experiment': experiment_id,
                        'success': success
                    })
                    
                    # Update confidence
                    if success:
                        hyp['confidence'] = min(0.95, hyp['confidence'] + 0.1)
                    else:
                        hyp['confidence'] = max(0.05, hyp['confidence'] - 0.2)
                    
                    hyp['status'] = 'supported' if hyp['confidence'] > 0.7 else 'testing'
                
                break
    
    def get_validated_hypotheses(self) -> List[Dict]:
        """Get hypotheses with high confidence."""
        return [
            h for h in self.hypotheses.values()
            if h['confidence'] > 0.7
        ]


class ContinuousLearningSystem:
    """
    Main Continuous Learning System.
    
    Integrates:
    - UnsupervisedLearner
    - ActiveLearner
    - ObservationalLearner
    - ExperimentalLearner
    """
    
    def __init__(self):
        self.unsupervised = UnsupervisedLearner()
        self.active = ActiveLearner()
        self.observational = ObservationalLearner()
        self.experimental = ExperimentalLearner()
        
        self.learning_history: deque = deque(maxlen=1000)
        
        logger.info("ContinuousLearningSystem initialized")
    
    def learn(
        self,
        example: LearningExample,
        learning_mode: str = "auto"
    ):
        """Learn from an example using appropriate method."""
        if learning_mode == "auto":
            # Choose mode based on example
            if example.label is None:
                mode = "unsupervised"
            elif example.source == "observation":
                mode = "observational"
            else:
                mode = "active"
        else:
            mode = learning_mode
        
        if mode == "unsupervised":
            self.unsupervised.learn(example)
        elif mode == "observational":
            self.observational.observe(
                agent_id=example.source,
                state=example.features,
                action=example.label or "unknown",
                result=example.confidence
            )
        elif mode == "active":
            self.active.update_concept(
                name=example.label or "unknown",
                features=example.features
            )
        
        self.learning_history.append({
            'example_id': example.example_id,
            'mode': mode,
            'timestamp': time.time()
        })
    
    def get_learning_needs(self) -> List[str]:
        """Identify current learning needs."""
        # Estimate performance
        concept_performance = {
            name: concept.confidence
            for name, concept in self.active.concepts.items()
        }
        
        return self.active.identify_learning_needs(concept_performance)
    
    def get_status(self) -> Dict[str, Any]:
        """Get learning system status."""
        return {
            'clusters': len(self.unsupervised.cluster_centers),
            'patterns_discovered': len(self.unsupervised.patterns_discovered),
            'concepts_learned': len(self.active.concepts),
            'behaviors_observed': len(self.observational.observed_behaviors),
            'behaviors_learned': len(self.observational.learned_behaviors),
            'hypotheses': len(self.experimental.hypotheses),
            'validated_hypotheses': len(self.experimental.get_validated_hypotheses()),
            'total_examples': len(self.learning_history)
        }
