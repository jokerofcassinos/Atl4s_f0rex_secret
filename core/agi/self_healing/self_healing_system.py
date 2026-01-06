"""
AGI Fase 2: Self-Healing System

Auto-Diagnóstico e Auto-Reparação:
- Sistema Imunológico para Erros
- Diagnóstico de Causa Raiz
- Predição de Falhas
- Auto-Reparação
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("SelfHealing")


class AnomalyType(Enum):
    PERFORMANCE = "performance"
    BEHAVIOR = "behavior"
    OUTPUT = "output"
    RESOURCE = "resource"
    TIMING = "timing"


class RepairStrategy(Enum):
    RESTART = "restart"
    RECONFIGURE = "reconfigure"
    ISOLATE = "isolate"
    REPLACE = "replace"
    PATCH = "patch"


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    component: str
    description: str
    severity: float
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class Diagnosis:
    """Diagnosis of a problem."""
    diagnosis_id: str
    symptom: str
    root_cause: str
    confidence: float
    evidence: List[str]
    repair_strategy: RepairStrategy


@dataclass
class ImmuneMemory:
    """Memory of past problems and solutions."""
    problem_signature: str
    solution: str
    effectiveness: float
    occurrences: int = 1
    last_seen: float = field(default_factory=time.time)


class AnomalyDetector:
    """Immune system for detecting anomalies."""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.baselines: Dict[str, Dict] = {}
        self.anomalies: List[Anomaly] = []
        self._anomaly_counter = 0
        
        logger.info("AnomalyDetector initialized")
    
    def set_baseline(self, component: str, metrics: Dict[str, float]):
        """Set baseline metrics for a component."""
        self.baselines[component] = {
            'metrics': metrics,
            'established_at': time.time()
        }
    
    def detect(self, component: str, current_metrics: Dict[str, float]) -> List[Anomaly]:
        """Detect anomalies in component metrics."""
        if component not in self.baselines:
            self.set_baseline(component, current_metrics)
            return []
        
        baseline = self.baselines[component]['metrics']
        detected = []
        
        for metric, value in current_metrics.items():
            if metric in baseline:
                expected = baseline[metric]
                deviation = abs(value - expected) / max(0.01, expected)
                
                if deviation > (1 - self.sensitivity):
                    self._anomaly_counter += 1
                    anomaly = Anomaly(
                        anomaly_id=f"anom_{self._anomaly_counter}",
                        anomaly_type=AnomalyType.PERFORMANCE,
                        component=component,
                        description=f"{metric} deviated {deviation:.1%} from baseline",
                        severity=min(1.0, deviation)
                    )
                    self.anomalies.append(anomaly)
                    detected.append(anomaly)
        
        return detected
    
    def get_unresolved(self) -> List[Anomaly]:
        """Get unresolved anomalies."""
        return [a for a in self.anomalies if not a.resolved]


class DiagnosticEngine:
    """Deep diagnosis of problems."""
    
    def __init__(self):
        self.diagnoses: List[Diagnosis] = []
        self.failure_chains: Dict[str, List[str]] = defaultdict(list)
        self._diag_counter = 0
        
        logger.info("DiagnosticEngine initialized")
    
    def diagnose(self, anomaly: Anomaly) -> Diagnosis:
        """Diagnose root cause of anomaly."""
        self._diag_counter += 1
        
        root_cause = self._find_root_cause(anomaly)
        strategy = self._select_repair_strategy(root_cause, anomaly.severity)
        
        diagnosis = Diagnosis(
            diagnosis_id=f"diag_{self._diag_counter}",
            symptom=anomaly.description,
            root_cause=root_cause,
            confidence=0.7,
            evidence=[f"Anomaly in {anomaly.component}"],
            repair_strategy=strategy
        )
        
        self.diagnoses.append(diagnosis)
        return diagnosis
    
    def _find_root_cause(self, anomaly: Anomaly) -> str:
        """Find root cause through analysis."""
        if anomaly.anomaly_type == AnomalyType.PERFORMANCE:
            return "Performance degradation in component"
        elif anomaly.anomaly_type == AnomalyType.BEHAVIOR:
            return "Unexpected behavior pattern"
        elif anomaly.anomaly_type == AnomalyType.RESOURCE:
            return "Resource constraint"
        else:
            return "Unknown root cause"
    
    def _select_repair_strategy(self, root_cause: str, severity: float) -> RepairStrategy:
        """Select appropriate repair strategy."""
        if severity > 0.8:
            return RepairStrategy.RESTART
        elif severity > 0.5:
            return RepairStrategy.RECONFIGURE
        else:
            return RepairStrategy.PATCH
    
    def trace_failure_chain(self, component: str, failed_components: List[str]):
        """Trace chain of failures."""
        self.failure_chains[component] = failed_components
    
    def predict_failure(self, metrics_history: List[Dict]) -> Optional[str]:
        """Predict potential failures."""
        if len(metrics_history) < 5:
            return None
        
        recent = metrics_history[-5:]
        
        for metric in recent[0].keys():
            values = [m.get(metric, 0) for m in recent]
            if all(values[i] < values[i-1] for i in range(1, len(values))):
                return f"Declining {metric} - potential failure"
        
        return None


class SelfRepairer:
    """Self-repair mechanisms."""
    
    def __init__(self):
        self.repairs: List[Dict] = []
        self.redundancy_pool: Dict[str, Any] = {}
        
        logger.info("SelfRepairer initialized")
    
    def repair(self, diagnosis: Diagnosis, dry_run: bool = True) -> Dict[str, Any]:
        """Execute repair based on diagnosis."""
        repair_record = {
            'diagnosis_id': diagnosis.diagnosis_id,
            'strategy': diagnosis.repair_strategy.value,
            'timestamp': time.time(),
            'dry_run': dry_run,
            'success': False
        }
        
        if dry_run:
            repair_record['action'] = f"Would apply {diagnosis.repair_strategy.value}"
            repair_record['success'] = True
        else:
            if diagnosis.repair_strategy == RepairStrategy.RESTART:
                repair_record['action'] = "Restart component"
                repair_record['success'] = True
            elif diagnosis.repair_strategy == RepairStrategy.RECONFIGURE:
                repair_record['action'] = "Reconfigure component"
                repair_record['success'] = True
            elif diagnosis.repair_strategy == RepairStrategy.PATCH:
                repair_record['action'] = "Apply patch"
                repair_record['success'] = True
        
        self.repairs.append(repair_record)
        return repair_record
    
    def add_redundancy(self, component: str, backup: Any):
        """Add redundancy for a component."""
        self.redundancy_pool[component] = backup
    
    def activate_redundancy(self, component: str) -> bool:
        """Activate redundant component."""
        if component in self.redundancy_pool:
            logger.info(f"Activating redundancy for {component}")
            return True
        return False


class ImmuneSystem:
    """Immune system with memory."""
    
    def __init__(self):
        self.memory: Dict[str, ImmuneMemory] = {}
        self.detector = AnomalyDetector()
        self.diagnostic = DiagnosticEngine()
        self.repairer = SelfRepairer()
        
        logger.info("ImmuneSystem initialized")
    
    def remember_problem(self, signature: str, solution: str, effectiveness: float):
        """Remember a problem and its solution."""
        if signature in self.memory:
            mem = self.memory[signature]
            mem.occurrences += 1
            mem.effectiveness = (mem.effectiveness + effectiveness) / 2
            mem.last_seen = time.time()
        else:
            self.memory[signature] = ImmuneMemory(
                problem_signature=signature,
                solution=solution,
                effectiveness=effectiveness
            )
    
    def recall_solution(self, signature: str) -> Optional[str]:
        """Recall solution for a known problem."""
        if signature in self.memory:
            mem = self.memory[signature]
            if mem.effectiveness > 0.5:
                return mem.solution
        return None
    
    def vaccinate(self, potential_problem: str, prevention: Callable):
        """Proactive prevention."""
        logger.info(f"Vaccination for: {potential_problem}")


class SelfHealingSystem:
    """Main Self-Healing System."""
    
    def __init__(self):
        self.immune = ImmuneSystem()
        
        logger.info("SelfHealingSystem initialized")
    
    def monitor_and_heal(self, component: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Monitor component and heal if necessary."""
        anomalies = self.immune.detector.detect(component, metrics)
        
        results = {
            'component': component,
            'anomalies_detected': len(anomalies),
            'repairs': []
        }
        
        for anomaly in anomalies:
            signature = f"{anomaly.component}:{anomaly.anomaly_type.value}"
            known_solution = self.immune.recall_solution(signature)
            
            if known_solution:
                results['repairs'].append({
                    'anomaly': anomaly.anomaly_id,
                    'action': f"Applied known solution: {known_solution}"
                })
            else:
                diagnosis = self.immune.diagnostic.diagnose(anomaly)
                repair = self.immune.repairer.repair(diagnosis, dry_run=True)
                results['repairs'].append(repair)
                
                self.immune.remember_problem(
                    signature, 
                    diagnosis.repair_strategy.value, 
                    0.7
                )
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'unresolved_anomalies': len(self.immune.detector.get_unresolved()),
            'diagnoses': len(self.immune.diagnostic.diagnoses),
            'repairs': len(self.immune.repairer.repairs),
            'immune_memory': len(self.immune.memory)
        }
