"""
AGI Ultra: Reasoning Quality Metrics

Measures and tracks the quality of AGI reasoning:
- Reasoning depth analysis
- Coherence scoring
- Decision quality tracking
- Calibration measurement
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("ReasoningMetrics")


class ReasoningQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class ReasoningMetric:
    """A single reasoning quality measurement."""
    module_name: str
    depth: int
    coherence: float  # 0-1
    confidence: float  # 0-1
    actual_accuracy: Optional[float]  # 0-1, None if unknown
    timestamp: float = field(default_factory=time.time)
    
    def get_calibration_error(self) -> Optional[float]:
        """Get calibration error (confidence vs accuracy)."""
        if self.actual_accuracy is None:
            return None
        return abs(self.confidence - self.actual_accuracy)


@dataclass
class ModuleStats:
    """Statistics for a single module."""
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_depth: float = 0.0
    avg_coherence: float = 0.0
    avg_confidence: float = 0.0
    calibration_error: float = 0.0


class ReasoningMetricsTracker:
    """
    Track and analyze reasoning quality across AGI modules.
    
    Features:
    - Depth analysis
    - Coherence scoring
    - Confidence calibration
    - Historical tracking
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Per-module tracking
        self.module_metrics: Dict[str, deque] = {}
        self.module_stats: Dict[str, ModuleStats] = {}
        
        # Global metrics
        self.global_metrics: deque = deque(maxlen=max_history)
        
        # Calibration buckets (for reliability diagram)
        self.confidence_buckets: Dict[int, List[float]] = {
            i: [] for i in range(10)
        }
        
        logger.info("ReasoningMetricsTracker initialized")
    
    def record_reasoning(
        self,
        module_name: str,
        depth: int,
        coherence: float,
        confidence: float,
        actual_accuracy: Optional[float] = None
    ):
        """Record a reasoning event."""
        metric = ReasoningMetric(
            module_name=module_name,
            depth=depth,
            coherence=coherence,
            confidence=confidence,
            actual_accuracy=actual_accuracy
        )
        
        # Store in module history
        if module_name not in self.module_metrics:
            self.module_metrics[module_name] = deque(maxlen=self.max_history)
            self.module_stats[module_name] = ModuleStats()
        
        self.module_metrics[module_name].append(metric)
        self.global_metrics.append(metric)
        
        # Update statistics
        stats = self.module_stats[module_name]
        stats.total_predictions += 1
        
        # Running averages
        n = stats.total_predictions
        stats.avg_depth = stats.avg_depth + (depth - stats.avg_depth) / n
        stats.avg_coherence = stats.avg_coherence + (coherence - stats.avg_coherence) / n
        stats.avg_confidence = stats.avg_confidence + (confidence - stats.avg_confidence) / n
        
        # Calibration tracking
        if actual_accuracy is not None:
            if actual_accuracy > 0.5:
                stats.correct_predictions += 1
            
            # Update calibration buckets
            bucket = min(9, int(confidence * 10))
            self.confidence_buckets[bucket].append(actual_accuracy)
            
            # Update calibration error
            calib_err = metric.get_calibration_error()
            if calib_err is not None:
                stats.calibration_error = stats.calibration_error + (calib_err - stats.calibration_error) / n
    
    def get_quality_score(self, module_name: str) -> Tuple[ReasoningQuality, float]:
        """Get overall reasoning quality for a module."""
        if module_name not in self.module_stats:
            return ReasoningQuality.INVALID, 0.0
        
        stats = self.module_stats[module_name]
        
        if stats.total_predictions < 10:
            return ReasoningQuality.INVALID, 0.0
        
        # Calculate composite score
        accuracy = stats.correct_predictions / stats.total_predictions
        coherence = stats.avg_coherence
        calibration = 1 - stats.calibration_error
        depth_score = min(1.0, stats.avg_depth / 10)  # Normalize to 10
        
        # Weighted composite
        score = (
            accuracy * 0.4 +
            coherence * 0.25 +
            calibration * 0.25 +
            depth_score * 0.1
        )
        
        # Map to quality level
        if score >= 0.8:
            quality = ReasoningQuality.EXCELLENT
        elif score >= 0.6:
            quality = ReasoningQuality.GOOD
        elif score >= 0.4:
            quality = ReasoningQuality.FAIR
        else:
            quality = ReasoningQuality.POOR
        
        return quality, score
    
    def get_calibration_curve(self) -> Dict[float, float]:
        """Get calibration curve data (predicted confidence vs actual accuracy)."""
        curve = {}
        
        for bucket, accuracies in self.confidence_buckets.items():
            if accuracies:
                predicted_conf = (bucket + 0.5) / 10  # Bucket center
                actual_acc = np.mean(accuracies)
                curve[predicted_conf] = actual_acc
        
        return curve
    
    def get_module_report(self, module_name: str) -> Dict[str, Any]:
        """Get detailed report for a module."""
        if module_name not in self.module_stats:
            return {'error': 'Module not found'}
        
        stats = self.module_stats[module_name]
        quality, score = self.get_quality_score(module_name)
        
        return {
            'module': module_name,
            'quality': quality.value,
            'score': score,
            'total_predictions': stats.total_predictions,
            'accuracy': stats.correct_predictions / stats.total_predictions if stats.total_predictions > 0 else 0,
            'avg_depth': stats.avg_depth,
            'avg_coherence': stats.avg_coherence,
            'avg_confidence': stats.avg_confidence,
            'calibration_error': stats.calibration_error
        }
    
    def get_global_report(self) -> Dict[str, Any]:
        """Get global AGI reasoning report."""
        # Aggregate across all modules
        total_predictions = sum(s.total_predictions for s in self.module_stats.values())
        total_correct = sum(s.correct_predictions for s in self.module_stats.values())
        
        if total_predictions == 0:
            return {'error': 'No data'}
        
        # Module rankings
        module_scores = []
        for name in self.module_stats:
            quality, score = self.get_quality_score(name)
            module_scores.append((name, quality.value, score))
        
        module_scores.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'total_modules': len(self.module_stats),
            'total_predictions': total_predictions,
            'global_accuracy': total_correct / total_predictions,
            'calibration_curve': self.get_calibration_curve(),
            'module_rankings': module_scores[:10],
            'worst_modules': module_scores[-5:] if len(module_scores) > 5 else []
        }


class CoherenceAnalyzer:
    """
    Analyze coherence between reasoning steps.
    
    Checks for contradictions and logical consistency.
    """
    
    def __init__(self):
        self.contradiction_patterns = [
            ('BUY', 'SELL'),
            ('bullish', 'bearish'),
            ('up', 'down'),
            ('increase', 'decrease')
        ]
    
    def analyze_coherence(
        self,
        reasoning_chain: List[str],
        decisions: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Analyze coherence of reasoning chain.
        
        Returns:
            (coherence_score, list of issues found)
        """
        issues = []
        
        if not reasoning_chain:
            return 0.5, ["Empty reasoning chain"]
        
        # Check for internal contradictions
        all_text = ' '.join(reasoning_chain).lower()
        
        for term1, term2 in self.contradiction_patterns:
            if term1 in all_text and term2 in all_text:
                issues.append(f"Potential contradiction: '{term1}' vs '{term2}'")
        
        # Check decision consistency
        if decisions:
            unique_decisions = set(decisions)
            if len(unique_decisions) > 1 and 'WAIT' not in unique_decisions:
                issues.append(f"Mixed decisions: {unique_decisions}")
        
        # Check reasoning depth
        avg_length = np.mean([len(r.split()) for r in reasoning_chain])
        if avg_length < 5:
            issues.append("Shallow reasoning (avg < 5 words)")
        
        # Calculate score
        penalty = len(issues) * 0.15
        coherence = max(0.0, 1.0 - penalty)
        
        return coherence, issues
