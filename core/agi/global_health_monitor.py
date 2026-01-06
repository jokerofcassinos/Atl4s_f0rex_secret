"""
AGI Ultra-Complete: Global Health Monitor

Sistema de monitoramento de saúde de todos os componentes:
- ComponentHealthChecker: Verifica saúde
- AlertSystem: Sistema de alertas
- DiagnosticEngine: Diagnóstico automático
- RecoveryManager: Gerenciamento de recuperação
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("GlobalHealthMonitor")


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    latency_ms: float
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class ComponentHealthChecker:
    """Checks health of components."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.thresholds = {
            'latency_warning_ms': 500,
            'latency_critical_ms': 2000,
            'error_rate_warning': 0.05,
            'error_rate_critical': 0.2
        }
        
        logger.info("ComponentHealthChecker initialized")
    
    def register_check(self, component: str, check_func: Callable):
        """Register a health check."""
        self.checks[component] = check_func
    
    def check(self, component: str) -> HealthCheck:
        """Run health check for component."""
        start = time.perf_counter()
        
        if component not in self.checks:
            return HealthCheck(
                component=component,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                message="No health check registered"
            )
        
        try:
            result = self.checks[component]()
            latency = (time.perf_counter() - start) * 1000
            
            status = HealthStatus.HEALTHY
            message = "OK"
            
            if latency > self.thresholds['latency_critical_ms']:
                status = HealthStatus.CRITICAL
                message = f"Latency critical: {latency:.0f}ms"
            elif latency > self.thresholds['latency_warning_ms']:
                status = HealthStatus.DEGRADED
                message = f"Latency high: {latency:.0f}ms"
            
            if isinstance(result, dict):
                if result.get('error_rate', 0) > self.thresholds['error_rate_critical']:
                    status = HealthStatus.CRITICAL
                    message = f"Error rate critical: {result['error_rate']:.0%}"
                elif result.get('error_rate', 0) > self.thresholds['error_rate_warning']:
                    status = HealthStatus.DEGRADED
                    message = f"Error rate high: {result['error_rate']:.0%}"
            
            return HealthCheck(
                component=component,
                status=status,
                latency_ms=latency,
                message=message,
                metrics=result if isinstance(result, dict) else {}
            )
            
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return HealthCheck(
                component=component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=f"Check failed: {str(e)}"
            )
    
    def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        for component in self.checks:
            results[component] = self.check(component)
        return results


class AlertSystem:
    """Manages system alerts."""
    
    def __init__(self):
        self.alerts: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        
        self._alert_counter = 0
        
        logger.info("AlertSystem initialized")
    
    def add_handler(self, handler: Callable):
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def raise_alert(self, severity: AlertSeverity, component: str, 
                    message: str, details: Dict = None) -> Alert:
        """Raise an alert."""
        self._alert_counter += 1
        
        alert = Alert(
            id=f"alert_{self._alert_counter}",
            severity=severity,
            component=component,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        self.active_alerts[alert.id] = alert
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(f"Alert raised: [{severity.value}] {component}: {message}")
        return alert
    
    def acknowledge(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
    
    def resolve(self, alert_id: str):
        """Resolve an alert."""
        self.active_alerts.pop(alert_id, None)
    
    def get_active(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts."""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts


class DiagnosticEngine:
    """Automatic system diagnostics."""
    
    def __init__(self, checker: ComponentHealthChecker):
        self.checker = checker
        self.diagnostics: List[Dict] = []
        self.known_issues: Dict[str, Dict] = {}
        
        logger.info("DiagnosticEngine initialized")
    
    def diagnose(self, component: str = None) -> Dict[str, Any]:
        """Run diagnostics."""
        if component:
            checks = {component: self.checker.check(component)}
        else:
            checks = self.checker.check_all()
        
        issues = []
        recommendations = []
        
        for comp, check in checks.items():
            if check.status == HealthStatus.CRITICAL:
                issues.append({
                    'component': comp,
                    'severity': 'critical',
                    'message': check.message
                })
                recommendations.append(f"Immediate attention required for {comp}")
            
            elif check.status == HealthStatus.UNHEALTHY:
                issues.append({
                    'component': comp,
                    'severity': 'high',
                    'message': check.message
                })
                recommendations.append(f"Investigate issues with {comp}")
            
            elif check.status == HealthStatus.DEGRADED:
                issues.append({
                    'component': comp,
                    'severity': 'medium',
                    'message': check.message
                })
        
        result = {
            'timestamp': time.time(),
            'components_checked': len(checks),
            'healthy': sum(1 for c in checks.values() if c.status == HealthStatus.HEALTHY),
            'issues': issues,
            'recommendations': recommendations
        }
        
        self.diagnostics.append(result)
        return result
    
    def learn_issue(self, pattern: str, solution: str, success: bool):
        """Learn from issue resolutions."""
        if pattern not in self.known_issues:
            self.known_issues[pattern] = {'solutions': {}, 'count': 0}
        
        self.known_issues[pattern]['count'] += 1
        
        if solution not in self.known_issues[pattern]['solutions']:
            self.known_issues[pattern]['solutions'][solution] = {'success': 0, 'total': 0}
        
        self.known_issues[pattern]['solutions'][solution]['total'] += 1
        if success:
            self.known_issues[pattern]['solutions'][solution]['success'] += 1


class RecoveryManager:
    """Manages system recovery."""
    
    def __init__(self):
        self.recovery_actions: Dict[str, Callable] = {}
        self.recovery_history: List[Dict] = []
        
        logger.info("RecoveryManager initialized")
    
    def register_recovery(self, component: str, action: Callable):
        """Register recovery action."""
        self.recovery_actions[component] = action
    
    def attempt_recovery(self, component: str) -> bool:
        """Attempt to recover a component."""
        if component not in self.recovery_actions:
            logger.warning(f"No recovery action for {component}")
            return False
        
        try:
            self.recovery_actions[component]()
            
            self.recovery_history.append({
                'component': component,
                'success': True,
                'timestamp': time.time()
            })
            
            logger.info(f"Recovery successful for {component}")
            return True
            
        except Exception as e:
            self.recovery_history.append({
                'component': component,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            })
            
            logger.error(f"Recovery failed for {component}: {e}")
            return False


class GlobalHealthMonitor:
    """Main global health monitor."""
    
    def __init__(self):
        self.checker = ComponentHealthChecker()
        self.alerts = AlertSystem()
        self.diagnostic = DiagnosticEngine(self.checker)
        self.recovery = RecoveryManager()
        
        self.status_history: deque = deque(maxlen=100)
        
        logger.info("GlobalHealthMonitor initialized")
    
    def register_component(self, component: str, 
                           health_check: Callable,
                           recovery_action: Callable = None):
        """Register a component for monitoring."""
        self.checker.register_check(component, health_check)
        if recovery_action:
            self.recovery.register_recovery(component, recovery_action)
    
    def check_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        checks = self.checker.check_all()
        
        for component, check in checks.items():
            if check.status == HealthStatus.CRITICAL:
                self.alerts.raise_alert(
                    AlertSeverity.CRITICAL,
                    component,
                    check.message
                )
            elif check.status == HealthStatus.UNHEALTHY:
                self.alerts.raise_alert(
                    AlertSeverity.ERROR,
                    component,
                    check.message
                )
        
        healthy_count = sum(1 for c in checks.values() if c.status == HealthStatus.HEALTHY)
        total = len(checks)
        
        overall_status = HealthStatus.HEALTHY
        if healthy_count < total * 0.5:
            overall_status = HealthStatus.CRITICAL
        elif healthy_count < total * 0.8:
            overall_status = HealthStatus.DEGRADED
        
        result = {
            'overall': overall_status.value,
            'components': {c: ch.status.value for c, ch in checks.items()},
            'healthy': healthy_count,
            'total': total,
            'active_alerts': len(self.alerts.active_alerts),
            'timestamp': time.time()
        }
        
        self.status_history.append(result)
        return result
    
    def auto_heal(self) -> Dict[str, bool]:
        """Attempt to auto-heal unhealthy components."""
        checks = self.checker.check_all()
        results = {}
        
        for component, check in checks.items():
            if check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                results[component] = self.recovery.attempt_recovery(component)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        return {
            'components': len(self.checker.checks),
            'active_alerts': len(self.alerts.active_alerts),
            'recent_recoveries': len([r for r in self.recovery.recovery_history[-10:] if r.get('success')])
        }


global_health_monitor = GlobalHealthMonitor()
