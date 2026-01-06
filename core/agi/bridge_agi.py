"""
AGI Ultra-Complete: Bridge AGI Components

Sistema de Comunicação Inteligente:
- AdaptiveCommunicationProtocol: Protocolo adaptativo
- IntelligentErrorRecovery: Recuperação de erros
- LatencyOptimizationEngine: Otimização de latência
- MessagePrioritizationSystem: Priorização de mensagens
"""

import logging
import time
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import threading

logger = logging.getLogger("BridgeAGI")


class ProtocolMode(Enum):
    FAST = "fast"
    RELIABLE = "reliable"
    BALANCED = "balanced"


class MessagePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class BridgeMessage:
    """Message in the bridge."""
    id: str
    content: Any
    priority: MessagePriority
    timestamp: float = field(default_factory=time.time)
    attempts: int = 0
    acknowledged: bool = False


@dataclass
class ConnectionStats:
    """Connection statistics."""
    latency_ms: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0


class AdaptiveCommunicationProtocol:
    """Adapts communication protocol based on conditions."""
    
    def __init__(self):
        self.mode = ProtocolMode.BALANCED
        self.stats = ConnectionStats()
        self.mode_history: deque = deque(maxlen=100)
        
        self.thresholds = {
            'fast_latency_max': 50,
            'reliable_error_max': 0.01,
            'switch_cooldown': 30
        }
        self.last_switch = 0
        
        logger.info("AdaptiveCommunicationProtocol initialized")
    
    def evaluate_and_adapt(self, latency: float, error_rate: float):
        """Evaluate conditions and adapt protocol."""
        self.stats.latency_ms = latency
        self.stats.error_rate = error_rate
        
        if time.time() - self.last_switch < self.thresholds['switch_cooldown']:
            return
        
        old_mode = self.mode
        
        if error_rate > self.thresholds['reliable_error_max']:
            self.mode = ProtocolMode.RELIABLE
        elif latency < self.thresholds['fast_latency_max'] and error_rate < 0.001:
            self.mode = ProtocolMode.FAST
        else:
            self.mode = ProtocolMode.BALANCED
        
        if self.mode != old_mode:
            self.last_switch = time.time()
            logger.info(f"Protocol mode changed: {old_mode.value} -> {self.mode.value}")
            
            self.mode_history.append({
                'from': old_mode.value,
                'to': self.mode.value,
                'timestamp': time.time()
            })
    
    def get_config(self) -> Dict[str, Any]:
        """Get current protocol configuration."""
        configs = {
            ProtocolMode.FAST: {
                'timeout': 1.0,
                'retries': 1,
                'compression': False,
                'batch_size': 100
            },
            ProtocolMode.RELIABLE: {
                'timeout': 10.0,
                'retries': 5,
                'compression': True,
                'batch_size': 10
            },
            ProtocolMode.BALANCED: {
                'timeout': 5.0,
                'retries': 3,
                'compression': True,
                'batch_size': 50
            }
        }
        return configs[self.mode]


class IntelligentErrorRecovery:
    """Intelligent error recovery."""
    
    def __init__(self):
        self.error_history: deque = deque(maxlen=100)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.success_rates: Dict[str, float] = defaultdict(lambda: 0.5)
        
        logger.info("IntelligentErrorRecovery initialized")
    
    def register_strategy(self, error_type: str, strategy: Callable):
        """Register recovery strategy."""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, error_type: str, context: Dict = None) -> Tuple[bool, str]:
        """Handle an error with intelligent recovery."""
        self.error_history.append({
            'type': error_type,
            'context': context,
            'timestamp': time.time()
        })
        
        if error_type in self.recovery_strategies:
            try:
                result = self.recovery_strategies[error_type](context)
                self.success_rates[error_type] = (
                    self.success_rates[error_type] * 0.9 + 0.1
                )
                return True, "Recovery successful"
            except Exception as e:
                self.success_rates[error_type] = (
                    self.success_rates[error_type] * 0.9
                )
                return False, f"Recovery failed: {e}"
        
        return False, "No recovery strategy"
    
    def get_best_strategy(self, error_type: str) -> Optional[str]:
        """Get best strategy for error type."""
        if error_type in self.recovery_strategies:
            return error_type
        
        similar = [et for et in self.recovery_strategies if et in error_type or error_type in et]
        if similar:
            return max(similar, key=lambda x: self.success_rates[x])
        
        return None


class LatencyOptimizationEngine:
    """Optimizes communication latency."""
    
    def __init__(self):
        self.latency_samples: deque = deque(maxlen=1000)
        self.optimizations: List[Dict] = []
        
        logger.info("LatencyOptimizationEngine initialized")
    
    def record_latency(self, operation: str, latency_ms: float):
        """Record latency sample."""
        self.latency_samples.append({
            'operation': operation,
            'latency': latency_ms,
            'timestamp': time.time()
        })
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze latency patterns."""
        if not self.latency_samples:
            return {}
        
        samples = list(self.latency_samples)
        
        by_operation = defaultdict(list)
        for s in samples:
            by_operation[s['operation']].append(s['latency'])
        
        analysis = {}
        for op, latencies in by_operation.items():
            analysis[op] = {
                'mean': sum(latencies) / len(latencies),
                'max': max(latencies),
                'min': min(latencies),
                'count': len(latencies)
            }
        
        return analysis
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest latency optimizations."""
        analysis = self.analyze()
        suggestions = []
        
        for op, stats in analysis.items():
            if stats['mean'] > 100:
                suggestions.append(f"High latency in {op}: consider caching")
            if stats['max'] > stats['mean'] * 10:
                suggestions.append(f"High variance in {op}: investigate spikes")
        
        self.optimizations = suggestions
        return suggestions


class MessagePrioritizationSystem:
    """Prioritizes messages intelligently."""
    
    def __init__(self):
        self.queues: Dict[MessagePriority, queue.Queue] = {
            p: queue.Queue() for p in MessagePriority
        }
        self.stats: Dict[MessagePriority, Dict] = {
            p: {'processed': 0, 'avg_wait': 0.0} for p in MessagePriority
        }
        
        self._message_counter = 0
        
        logger.info("MessagePrioritizationSystem initialized")
    
    def enqueue(self, content: Any, priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Enqueue a message."""
        self._message_counter += 1
        
        msg = BridgeMessage(
            id=f"msg_{self._message_counter}",
            content=content,
            priority=priority
        )
        
        self.queues[priority].put(msg)
        return msg.id
    
    def dequeue(self) -> Optional[BridgeMessage]:
        """Dequeue highest priority message."""
        for priority in MessagePriority:
            if not self.queues[priority].empty():
                msg = self.queues[priority].get_nowait()
                
                wait_time = time.time() - msg.timestamp
                stats = self.stats[priority]
                stats['processed'] += 1
                stats['avg_wait'] = (stats['avg_wait'] * 0.9) + (wait_time * 0.1)
                
                return msg
        
        return None
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue sizes."""
        return {p.value: self.queues[p].qsize() for p in MessagePriority}


class ConnectionPoolManager:
    """Manages connection pool."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, Dict] = {}
        self.available: List[str] = []
        
        logger.info("ConnectionPoolManager initialized")
    
    def acquire(self) -> Optional[str]:
        """Acquire a connection."""
        if self.available:
            conn_id = self.available.pop()
            self.connections[conn_id]['in_use'] = True
            return conn_id
        
        if len(self.connections) < self.max_connections:
            conn_id = f"conn_{len(self.connections)}"
            self.connections[conn_id] = {
                'created': time.time(),
                'in_use': True,
                'requests': 0
            }
            return conn_id
        
        return None
    
    def release(self, conn_id: str):
        """Release a connection."""
        if conn_id in self.connections:
            self.connections[conn_id]['in_use'] = False
            self.available.append(conn_id)


class BridgeAGI:
    """Main Bridge AGI System."""
    
    def __init__(self):
        self.protocol = AdaptiveCommunicationProtocol()
        self.recovery = IntelligentErrorRecovery()
        self.latency = LatencyOptimizationEngine()
        self.priority = MessagePrioritizationSystem()
        self.pool = ConnectionPoolManager()
        
        logger.info("BridgeAGI initialized")
    
    def send(self, content: Any, priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Send a message."""
        return self.priority.enqueue(content, priority)
    
    def receive(self) -> Optional[BridgeMessage]:
        """Receive next message."""
        return self.priority.dequeue()
    
    def adapt_protocol(self, latency: float, error_rate: float):
        """Adapt protocol based on conditions."""
        self.protocol.evaluate_and_adapt(latency, error_rate)
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            'protocol_mode': self.protocol.mode.value,
            'queues': self.priority.get_queue_status(),
            'connections': len(self.pool.connections),
            'optimizations': self.latency.suggest_optimizations()
        }
