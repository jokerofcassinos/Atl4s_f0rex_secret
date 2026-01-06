"""
AGI Ultra-Complete: Performance Analytics

Sistema de análise de performance de todos os componentes:
- PerformanceTracker: Rastreamento de métricas
- BenchmarkEngine: Benchmarking automático
- BottleneckDetector: Detecção de gargalos
- OptimizationSuggester: Sugestões de otimização
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from functools import wraps

logger = logging.getLogger("PerformanceAnalytics")


class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class Metric:
    """Performance metric."""
    name: str
    type: MetricType
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    component: str = ""


@dataclass
class Benchmark:
    """Benchmark result."""
    name: str
    iterations: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float


class PerformanceTracker:
    """Tracks performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregates: Dict[str, Dict] = {}
        
        logger.info("PerformanceTracker initialized")
    
    def record(self, name: str, value: float, 
               metric_type: MetricType = MetricType.CUSTOM,
               unit: str = "", component: str = ""):
        """Record a metric."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            unit=unit,
            component=component
        )
        
        self.metrics[name].append(metric)
        self._update_aggregate(name)
    
    def _update_aggregate(self, name: str):
        """Update aggregate statistics."""
        values = [m.value for m in self.metrics[name]]
        
        if len(values) < 2:
            return
        
        self.aggregates[name] = {
            'count': len(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }
    
    def get_metric(self, name: str) -> Optional[Dict]:
        """Get aggregate for a metric."""
        return self.aggregates.get(name)
    
    def get_recent(self, name: str, count: int = 10) -> List[Metric]:
        """Get recent metrics."""
        return list(self.metrics[name])[-count:]
    
    def time_function(self, name: str):
        """Decorator to time functions."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                self.record(name, elapsed, MetricType.LATENCY, "ms")
                return result
            return wrapper
        return decorator


class BenchmarkEngine:
    """Runs performance benchmarks."""
    
    def __init__(self):
        self.results: Dict[str, Benchmark] = {}
        self.history: List[Dict] = []
        
        logger.info("BenchmarkEngine initialized")
    
    def benchmark(self, name: str, func: Callable, 
                  iterations: int = 100, warmup: int = 10) -> Benchmark:
        """Run a benchmark."""
        for _ in range(warmup):
            func()
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        result = Benchmark(
            name=name,
            iterations=iterations,
            mean_time=statistics.mean(times),
            std_time=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            throughput=iterations / sum(times)
        )
        
        self.results[name] = result
        self.history.append({
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def compare(self, name1: str, name2: str) -> Dict[str, Any]:
        """Compare two benchmarks."""
        r1 = self.results.get(name1)
        r2 = self.results.get(name2)
        
        if not r1 or not r2:
            return {'error': 'Benchmark not found'}
        
        return {
            'speedup': r1.mean_time / r2.mean_time,
            'throughput_ratio': r2.throughput / r1.throughput,
            'better': name2 if r2.mean_time < r1.mean_time else name1
        }


class BottleneckDetector:
    """Detects performance bottlenecks."""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.bottlenecks: List[Dict] = []
        self.thresholds = {
            'latency_ms': 100,
            'error_rate': 0.05,
            'std_ratio': 0.5
        }
        
        logger.info("BottleneckDetector initialized")
    
    def analyze(self) -> List[Dict]:
        """Analyze for bottlenecks."""
        bottlenecks = []
        
        for name, aggregate in self.tracker.aggregates.items():
            if aggregate.get('mean', 0) > self.thresholds['latency_ms']:
                bottlenecks.append({
                    'type': 'high_latency',
                    'metric': name,
                    'value': aggregate['mean'],
                    'threshold': self.thresholds['latency_ms'],
                    'severity': 'high' if aggregate['mean'] > self.thresholds['latency_ms'] * 2 else 'medium'
                })
            
            mean = aggregate.get('mean', 0)
            std = aggregate.get('std', 0)
            if mean > 0 and std / mean > self.thresholds['std_ratio']:
                bottlenecks.append({
                    'type': 'high_variance',
                    'metric': name,
                    'value': std / mean,
                    'threshold': self.thresholds['std_ratio'],
                    'severity': 'medium'
                })
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def get_hotspots(self, top_n: int = 5) -> List[str]:
        """Get top performance hotspots."""
        aggregates = [(name, agg.get('mean', 0)) 
                      for name, agg in self.tracker.aggregates.items()]
        aggregates.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in aggregates[:top_n]]


class OptimizationSuggester:
    """Suggests optimizations based on metrics."""
    
    def __init__(self, detector: BottleneckDetector):
        self.detector = detector
        self.suggestions: List[Dict] = []
        
        logger.info("OptimizationSuggester initialized")
    
    def suggest(self) -> List[Dict]:
        """Generate optimization suggestions."""
        suggestions = []
        
        bottlenecks = self.detector.analyze()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'high_latency':
                suggestions.append({
                    'issue': f"High latency in {bottleneck['metric']}",
                    'suggestion': "Consider caching, parallelization, or algorithm optimization",
                    'priority': 'high' if bottleneck['severity'] == 'high' else 'medium'
                })
            
            elif bottleneck['type'] == 'high_variance':
                suggestions.append({
                    'issue': f"High variance in {bottleneck['metric']}",
                    'suggestion': "Investigate inconsistent behavior, consider adding timeouts",
                    'priority': 'medium'
                })
        
        hotspots = self.detector.get_hotspots()
        if hotspots:
            suggestions.append({
                'issue': f"Top hotspots: {', '.join(hotspots[:3])}",
                'suggestion': "Focus optimization efforts on these components",
                'priority': 'info'
            })
        
        self.suggestions = suggestions
        return suggestions


class PerformanceAnalytics:
    """Main performance analytics system."""
    
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.benchmark = BenchmarkEngine()
        self.detector = BottleneckDetector(self.tracker)
        self.suggester = OptimizationSuggester(self.detector)
        
        logger.info("PerformanceAnalytics initialized")
    
    def record_latency(self, component: str, operation: str, latency_ms: float):
        """Record latency metric."""
        self.tracker.record(
            f"{component}.{operation}",
            latency_ms,
            MetricType.LATENCY,
            "ms",
            component
        )
    
    def analyze(self) -> Dict[str, Any]:
        """Run full analysis."""
        return {
            'bottlenecks': self.detector.analyze(),
            'suggestions': self.suggester.suggest(),
            'hotspots': self.detector.get_hotspots(),
            'metrics_count': len(self.tracker.aggregates)
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard."""
        return {
            'metrics': self.tracker.aggregates,
            'benchmarks': {name: {
                'mean': r.mean_time,
                'throughput': r.throughput
            } for name, r in self.benchmark.results.items()},
            'bottlenecks': len(self.detector.bottlenecks),
            'suggestions': len(self.suggester.suggestions)
        }


performance_analytics = PerformanceAnalytics()
