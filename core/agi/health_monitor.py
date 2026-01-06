"""
AGI Ultra: Health Monitor

Monitors AGI system health, performance, and detects anomalies:
- Infinite loop detection in reasoning chains
- Performance degradation alerts
- Memory pressure monitoring
- Auto-correction triggers
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("HealthMonitor")


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"


class AlertType(Enum):
    """Types of health alerts."""
    INFINITE_LOOP = "infinite_loop"
    HIGH_LATENCY = "high_latency"
    MEMORY_PRESSURE = "memory_pressure"
    LOW_ACCURACY = "low_accuracy"
    STALE_DATA = "stale_data"
    CONFLICT_STORM = "conflict_storm"


@dataclass
class HealthAlert:
    """A system health alert."""
    alert_type: AlertType
    severity: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: float
    memory_mb: float
    last_activity: float
    error_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


class HealthMonitor:
    """
    AGI Ultra: System Health Monitor.
    
    Monitors:
    - Reasoning loop detection
    - Component latencies
    - Memory usage
    - Decision accuracy
    - Data freshness
    
    Provides:
    - Automatic alerts
    - Auto-correction triggers
    - Health dashboards
    """
    
    def __init__(
        self,
        loop_detection_window: int = 100,
        latency_threshold_ms: float = 1000,
        memory_threshold_mb: float = 4096,
        check_interval_seconds: float = 5.0
    ):
        self.loop_detection_window = loop_detection_window
        self.latency_threshold_ms = latency_threshold_ms
        self.memory_threshold_mb = memory_threshold_mb
        self.check_interval = check_interval_seconds
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {}
        
        # Alert tracking
        self.active_alerts: List[HealthAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Loop detection
        self.reasoning_hashes: deque = deque(maxlen=loop_detection_window)
        self.loop_count = 0
        
        # Latency tracking
        self.latencies: Dict[str, deque] = {}
        
        # Auto-correction handlers
        self.correction_handlers: Dict[AlertType, Callable] = {}
        
        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info(f"HealthMonitor initialized: latency_threshold={latency_threshold_ms}ms")
    
    # -------------------------------------------------------------------------
    # COMPONENT REGISTRATION
    # -------------------------------------------------------------------------
    def register_component(self, name: str, initial_status: HealthStatus = HealthStatus.HEALTHY):
        """Register a component for monitoring."""
        self.components[name] = ComponentHealth(
            name=name,
            status=initial_status,
            latency_ms=0,
            memory_mb=0,
            last_activity=time.time()
        )
        self.latencies[name] = deque(maxlen=100)
        logger.debug(f"Registered component: {name}")
    
    def update_component(
        self,
        name: str,
        latency_ms: Optional[float] = None,
        memory_mb: Optional[float] = None,
        status: Optional[HealthStatus] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update component health metrics."""
        if name not in self.components:
            self.register_component(name)
        
        comp = self.components[name]
        comp.last_activity = time.time()
        
        if latency_ms is not None:
            comp.latency_ms = latency_ms
            self.latencies[name].append(latency_ms)
            
            # Check for high latency
            if latency_ms > self.latency_threshold_ms:
                self._raise_alert(
                    AlertType.HIGH_LATENCY,
                    HealthStatus.WARNING,
                    f"{name} latency too high: {latency_ms:.1f}ms",
                    {'component': name, 'latency_ms': latency_ms}
                )
        
        if memory_mb is not None:
            comp.memory_mb = memory_mb
            
            if memory_mb > self.memory_threshold_mb:
                self._raise_alert(
                    AlertType.MEMORY_PRESSURE,
                    HealthStatus.WARNING,
                    f"{name} memory pressure: {memory_mb:.1f}MB",
                    {'component': name, 'memory_mb': memory_mb}
                )
        
        if status is not None:
            comp.status = status
        
        if metrics:
            comp.metrics.update(metrics)
    
    def record_error(self, component_name: str, error_msg: str):
        """Record an error for a component."""
        if component_name not in self.components:
            self.register_component(component_name)
        
        self.components[component_name].error_count += 1
        logger.warning(f"Error in {component_name}: {error_msg}")
    
    # -------------------------------------------------------------------------
    # LOOP DETECTION
    # -------------------------------------------------------------------------
    def check_for_loop(self, reasoning_state: Dict[str, Any]) -> bool:
        """
        Check if the reasoning is caught in an infinite loop.
        
        Args:
            reasoning_state: Current reasoning state
            
        Returns:
            True if a loop is detected
        """
        # Hash the reasoning state
        state_hash = hash(str(sorted(reasoning_state.items())))
        
        # Check for repetition
        if state_hash in self.reasoning_hashes:
            self.loop_count += 1
            
            if self.loop_count >= 3:
                self._raise_alert(
                    AlertType.INFINITE_LOOP,
                    HealthStatus.CRITICAL,
                    f"Infinite loop detected in reasoning (count={self.loop_count})",
                    {'state_hash': state_hash, 'loop_count': self.loop_count}
                )
                return True
        else:
            self.loop_count = 0
        
        self.reasoning_hashes.append(state_hash)
        return False
    
    def clear_loop_state(self):
        """Clear loop detection state."""
        self.reasoning_hashes.clear()
        self.loop_count = 0
    
    # -------------------------------------------------------------------------
    # ALERTING
    # -------------------------------------------------------------------------
    def _raise_alert(
        self,
        alert_type: AlertType,
        severity: HealthStatus,
        message: str,
        details: Dict[str, Any]
    ):
        """Raise a health alert."""
        # Check for duplicate active alerts
        for alert in self.active_alerts:
            if alert.alert_type == alert_type and not alert.resolved:
                # Update existing alert
                alert.timestamp = time.time()
                alert.details.update(details)
                return
        
        alert = HealthAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        logger.warning(f"Health alert: [{severity.value}] {message}")
        
        # Trigger auto-correction if handler exists
        if alert_type in self.correction_handlers:
            try:
                self.correction_handlers[alert_type](alert)
                alert.resolved = True
                alert.resolution_time = time.time()
                logger.info(f"Auto-correction applied for {alert_type.value}")
            except Exception as e:
                logger.error(f"Auto-correction failed: {e}")
    
    def resolve_alert(self, alert_type: AlertType):
        """Mark alerts of a type as resolved."""
        for alert in self.active_alerts:
            if alert.alert_type == alert_type and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = time.time()
        
        self.active_alerts = [a for a in self.active_alerts if not a.resolved]
    
    def register_correction_handler(self, alert_type: AlertType, handler: Callable):
        """Register an auto-correction handler for an alert type."""
        self.correction_handlers[alert_type] = handler
        logger.debug(f"Registered correction handler for {alert_type.value}")
    
    # -------------------------------------------------------------------------
    # HEALTH ASSESSMENT
    # -------------------------------------------------------------------------
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if any(a.severity == HealthStatus.CRITICAL for a in self.active_alerts if not a.resolved):
            return HealthStatus.CRITICAL
        
        if any(a.severity == HealthStatus.WARNING for a in self.active_alerts if not a.resolved):
            return HealthStatus.WARNING
        
        # Check components
        for comp in self.components.values():
            if comp.status == HealthStatus.CRITICAL:
                return HealthStatus.CRITICAL
            if comp.status == HealthStatus.DEGRADED:
                return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        overall = self.get_overall_health()
        
        report = {
            'overall_status': overall.value,
            'timestamp': time.time(),
            'components': {},
            'active_alerts': [],
            'metrics': {
                'total_components': len(self.components),
                'healthy_components': sum(1 for c in self.components.values() if c.status == HealthStatus.HEALTHY),
                'active_alert_count': len([a for a in self.active_alerts if not a.resolved]),
                'loop_detections': self.loop_count
            }
        }
        
        # Component details
        for name, comp in self.components.items():
            avg_latency = 0
            if name in self.latencies and self.latencies[name]:
                avg_latency = sum(self.latencies[name]) / len(self.latencies[name])
            
            report['components'][name] = {
                'status': comp.status.value,
                'latency_ms': comp.latency_ms,
                'avg_latency_ms': avg_latency,
                'memory_mb': comp.memory_mb,
                'last_activity_seconds_ago': time.time() - comp.last_activity,
                'error_count': comp.error_count,
                'metrics': comp.metrics
            }
        
        # Active alerts
        for alert in self.active_alerts:
            if not alert.resolved:
                report['active_alerts'].append({
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'age_seconds': time.time() - alert.timestamp
                })
        
        return report
    
    # -------------------------------------------------------------------------
    # BACKGROUND MONITORING
    # -------------------------------------------------------------------------
    def start_background_monitor(self):
        """Start background health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Background health monitor started")
    
    def stop_background_monitor(self):
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("Background health monitor stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_stale_components()
                self._cleanup_old_alerts()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_stale_components(self):
        """Check for stale (inactive) components."""
        now = time.time()
        stale_threshold = 60  # 1 minute
        
        for name, comp in self.components.items():
            if now - comp.last_activity > stale_threshold:
                self._raise_alert(
                    AlertType.STALE_DATA,
                    HealthStatus.DEGRADED,
                    f"{name} has not reported activity for {int(now - comp.last_activity)}s",
                    {'component': name, 'last_activity': comp.last_activity}
                )
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff = time.time() - 3600  # 1 hour
        self.active_alerts = [
            a for a in self.active_alerts
            if not a.resolved or (a.resolution_time and a.resolution_time > cutoff)
        ]
