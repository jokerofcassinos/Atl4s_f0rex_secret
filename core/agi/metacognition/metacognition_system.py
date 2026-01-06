"""
AGI Fase 2: Meta-Meta Reasoning and Executive Control

Meta-Cognição Avançada:
- Raciocínio sobre Raciocínio
- Monitoramento Meta-Cognitivo
- Controle Executivo
- Detecção de Vieses
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("MetaCognition")


class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


class CognitiveState(Enum):
    FOCUSED = "focused"
    EXPLORING = "exploring"
    DECIDING = "deciding"
    REFLECTING = "reflecting"
    CONFUSED = "confused"


@dataclass
class ReasoningProcess:
    """A reasoning process to be monitored."""
    process_id: str
    reasoning_type: ReasoningType
    input_data: Dict[str, Any]
    output: Optional[Any] = None
    confidence: float = 0.0
    duration_ms: float = 0.0
    success: Optional[bool] = None


@dataclass
class CognitiveError:
    """A detected cognitive error."""
    error_id: str
    error_type: str
    description: str
    severity: float
    detected_in: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class BiasInstance:
    """An instance of cognitive bias."""
    bias_type: str
    description: str
    evidence: List[str]
    mitigation: str
    detected_at: float = field(default_factory=time.time)


class MetaMetaReasoning:
    """
    Reasoning about reasoning about reasoning.
    
    Understands HOW the system reasons.
    """
    
    def __init__(self):
        self.reasoning_history: deque = deque(maxlen=1000)
        self.reasoning_theories: Dict[str, Dict] = {}
        self.invented_methods: List[Dict] = []
        
        logger.info("MetaMetaReasoning initialized")
    
    def record_reasoning(self, process: ReasoningProcess):
        """Record a reasoning process."""
        self.reasoning_history.append({
            'process': process,
            'timestamp': time.time()
        })
    
    def analyze_reasoning_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in reasoning."""
        if len(self.reasoning_history) < 10:
            return {'insufficient_data': True}
        
        type_stats = defaultdict(lambda: {'count': 0, 'success_rate': 0, 'avg_time': 0})
        
        for record in self.reasoning_history:
            process = record['process']
            stats = type_stats[process.reasoning_type.value]
            stats['count'] += 1
            if process.success is not None:
                n = stats['count']
                stats['success_rate'] = (stats['success_rate'] * (n-1) + int(process.success)) / n
            stats['avg_time'] = (stats['avg_time'] * (stats['count']-1) + process.duration_ms) / stats['count']
        
        return dict(type_stats)
    
    def develop_theory(self, observation: str, hypothesis: str) -> Dict:
        """Develop a theory about reasoning."""
        theory_id = f"theory_{len(self.reasoning_theories)}"
        
        theory = {
            'id': theory_id,
            'observation': observation,
            'hypothesis': hypothesis,
            'evidence': [],
            'confidence': 0.5,
            'created_at': time.time()
        }
        
        self.reasoning_theories[theory_id] = theory
        return theory
    
    def invent_method(self, goal: str, steps: List[str]) -> Dict:
        """Invent a new reasoning method."""
        method = {
            'id': f"method_{len(self.invented_methods)}",
            'goal': goal,
            'steps': steps,
            'tested': False,
            'effectiveness': 0.0
        }
        
        self.invented_methods.append(method)
        logger.info(f"New reasoning method invented for: {goal}")
        return method
    
    def optimize_reasoning(self) -> Dict[str, str]:
        """Optimize reasoning processes based on analysis."""
        analysis = self.analyze_reasoning_patterns()
        recommendations = {}
        
        for reasoning_type, stats in analysis.items():
            if isinstance(stats, dict) and 'success_rate' in stats:
                if stats['success_rate'] < 0.5:
                    recommendations[reasoning_type] = "Consider using alternative method"
                elif stats['avg_time'] > 1000:
                    recommendations[reasoning_type] = "Optimize for speed"
        
        return recommendations


class MetaMonitor:
    """Monitors cognitive processes."""
    
    def __init__(self):
        self.processes: Dict[str, ReasoningProcess] = {}
        self.errors: List[CognitiveError] = []
        self.biases: List[BiasInstance] = []
        self.confidence_calibration: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        
        logger.info("MetaMonitor initialized")
    
    def start_monitoring(self, process_id: str, reasoning_type: ReasoningType, input_data: Dict):
        """Start monitoring a process."""
        self.processes[process_id] = ReasoningProcess(
            process_id=process_id,
            reasoning_type=reasoning_type,
            input_data=input_data
        )
    
    def end_monitoring(self, process_id: str, output: Any, confidence: float, success: bool):
        """End monitoring and record results."""
        if process_id in self.processes:
            process = self.processes[process_id]
            process.output = output
            process.confidence = confidence
            process.success = success
            
            self.confidence_calibration[process.reasoning_type.value].append((confidence, success))
    
    def detect_errors(self) -> List[CognitiveError]:
        """Detect cognitive errors."""
        new_errors = []
        
        for process in self.processes.values():
            if process.confidence > 0.8 and process.success == False:
                error = CognitiveError(
                    error_id=f"err_{len(self.errors)}",
                    error_type="overconfidence",
                    description=f"High confidence ({process.confidence}) but failed",
                    severity=0.7,
                    detected_in=process.process_id
                )
                new_errors.append(error)
                self.errors.append(error)
        
        return new_errors
    
    def detect_biases(self) -> List[BiasInstance]:
        """Detect cognitive biases."""
        new_biases = []
        
        if len(self.confidence_calibration) > 0:
            for rtype, calibrations in self.confidence_calibration.items():
                if len(calibrations) >= 10:
                    recent = calibrations[-10:]
                    high_conf = [c for c in recent if c[0] > 0.7]
                    if high_conf:
                        actual_rate = sum(1 for _, s in high_conf if s) / len(high_conf)
                        if actual_rate < 0.5:
                            bias = BiasInstance(
                                bias_type="overconfidence",
                                description=f"Overconfidence in {rtype}: {actual_rate:.0%} actual vs >70% predicted",
                                evidence=[str(c) for c in high_conf[:3]],
                                mitigation="Reduce confidence thresholds"
                            )
                            new_biases.append(bias)
                            self.biases.append(bias)
        
        return new_biases
    
    def evaluate_confidence(self) -> Dict[str, float]:
        """Evaluate confidence calibration."""
        results = {}
        
        for rtype, calibrations in self.confidence_calibration.items():
            if calibrations:
                bins = defaultdict(list)
                for conf, success in calibrations:
                    bin_key = round(conf, 1)
                    bins[bin_key].append(success)
                
                calibration_error = 0
                for bin_key, successes in bins.items():
                    actual = sum(successes) / len(successes)
                    calibration_error += abs(bin_key - actual)
                
                results[rtype] = 1 - (calibration_error / max(1, len(bins)))
        
        return results


class ExecutiveControl:
    """Controls cognitive resource allocation."""
    
    def __init__(self):
        self.current_state = CognitiveState.FOCUSED
        self.resource_allocation: Dict[str, float] = {}
        self.active_strategies: List[str] = []
        self.conflict_queue: deque = deque(maxlen=50)
        
        logger.info("ExecutiveControl initialized")
    
    def allocate_resources(self, tasks: Dict[str, float]) -> Dict[str, float]:
        """Allocate cognitive resources to tasks."""
        total = sum(tasks.values())
        if total == 0:
            return tasks
        
        self.resource_allocation = {
            task: priority / total
            for task, priority in tasks.items()
        }
        
        return self.resource_allocation
    
    def select_strategy(
        self,
        available_strategies: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Select reasoning strategy based on context."""
        if context.get('uncertainty', 0) > 0.7:
            strategy = 'exploration'
        elif context.get('time_pressure', False):
            strategy = 'heuristic'
        elif context.get('high_stakes', False):
            strategy = 'careful_analysis'
        else:
            strategy = 'balanced'
        
        if strategy in available_strategies:
            self.active_strategies.append(strategy)
            return strategy
        
        return available_strategies[0] if available_strategies else 'default'
    
    def coordinate_processes(self, processes: List[str]) -> Dict[str, int]:
        """Coordinate multiple cognitive processes."""
        priorities = {}
        
        for i, process in enumerate(processes):
            if 'risk' in process.lower():
                priorities[process] = 0
            elif 'decision' in process.lower():
                priorities[process] = 1
            else:
                priorities[process] = 2 + i
        
        return priorities
    
    def resolve_conflict(self, conflict: Dict[str, Any]) -> str:
        """Resolve internal conflict."""
        self.conflict_queue.append(conflict)
        
        options = conflict.get('options', [])
        scores = conflict.get('scores', {})
        
        if not options:
            return 'no_action'
        
        if scores:
            return max(options, key=lambda o: scores.get(o, 0))
        
        return options[0]
    
    def set_cognitive_state(self, state: CognitiveState):
        """Set current cognitive state."""
        self.current_state = state


class MetaCognitionSystem:
    """Main Meta-Cognition System."""
    
    def __init__(self):
        self.meta_meta = MetaMetaReasoning()
        self.monitor = MetaMonitor()
        self.control = ExecutiveControl()
        
        logger.info("MetaCognitionSystem initialized")
    
    def think_about_thinking(self) -> Dict[str, Any]:
        """Meta-cognitive reflection."""
        patterns = self.meta_meta.analyze_reasoning_patterns()
        errors = self.monitor.detect_errors()
        biases = self.monitor.detect_biases()
        calibration = self.monitor.evaluate_confidence()
        optimizations = self.meta_meta.optimize_reasoning()
        
        return {
            'reasoning_patterns': patterns,
            'errors_detected': len(errors),
            'biases_detected': len(biases),
            'calibration': calibration,
            'optimizations': optimizations
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'cognitive_state': self.control.current_state.value,
            'theories': len(self.meta_meta.reasoning_theories),
            'invented_methods': len(self.meta_meta.invented_methods),
            'errors': len(self.monitor.errors),
            'biases': len(self.monitor.biases),
            'active_strategies': self.control.active_strategies[-5:]
        }
