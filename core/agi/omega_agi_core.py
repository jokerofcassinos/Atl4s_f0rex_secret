"""
AGI Ultra-Complete: OmegaSystem Components

Componentes AGI para o Sistema Principal:
- MetaExecutionLoop: Loop que raciocina sobre si
- AdaptiveScheduler: Agendamento adaptativo
- AdvancedStateMachine: Estados inteligentes
- PerformanceMonitor: Monitoramento contínuo
"""

import logging
import time
import datetime
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("OmegaAGI")


class SystemState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    ANALYZING = "analyzing"
    TRADING = "trading"
    WAITING = "waiting"
    HEALING = "healing"
    EVOLVING = "evolving"
    SHUTDOWN = "shutdown"


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    decisions_made: int = 0
    trades_executed: int = 0
    profitable_trades: int = 0
    win_rate: float = 0.0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    errors: int = 0
    recoveries: int = 0


@dataclass  
class ExecutionContext:
    """Context for meta-execution loop."""
    iteration: int
    timestamp: float
    state: SystemState
    metrics: PerformanceMetrics
    recent_decisions: List[str]
    market_conditions: Dict[str, Any]


class MetaExecutionLoop:
    """
    Loop that reasons about its own execution.
    
    Optimizes its own behavior based on performance.
    """
    
    def __init__(self):
        self.iteration = 0
        self.start_time = time.time()
        self.history: deque = deque(maxlen=1000)
        
        self.optimization_rules: List[Dict] = []
        self.current_strategy = "balanced"
        
        self._init_optimization_rules()
        logger.info("MetaExecutionLoop initialized")
    
    def _init_optimization_rules(self):
        """Initialize self-optimization rules."""
        self.optimization_rules = [
            {
                'condition': lambda ctx: ctx.metrics.avg_latency_ms > 500,
                'action': 'reduce_analysis_depth',
                'description': 'High latency detected, reducing analysis depth'
            },
            {
                'condition': lambda ctx: ctx.metrics.win_rate < 0.4 and ctx.metrics.trades_executed > 10,
                'action': 'switch_to_conservative',
                'description': 'Low win rate, switching to conservative mode'
            },
            {
                'condition': lambda ctx: ctx.metrics.errors > 5,
                'action': 'enable_healing',
                'description': 'Multiple errors, enabling healing mode'
            },
            {
                'condition': lambda ctx: len(ctx.recent_decisions) > 50 and ctx.recent_decisions.count("WAIT") / len(ctx.recent_decisions) > 0.8,
                'action': 'switch_to_aggressive',
                'description': 'Too many WAIT signals, may be missing opportunities'
            }
        ]
    
    def pre_iteration(self, context: ExecutionContext) -> Dict[str, Any]:
        """Pre-iteration reasoning."""
        self.iteration += 1
        
        adjustments = {}
        
        for rule in self.optimization_rules:
            if rule['condition'](context):
                adjustments[rule['action']] = True
                logger.info(f"MetaLoop Adjustment: {rule['description']}")
        
        return adjustments
    
    def post_iteration(self, context: ExecutionContext, result: Dict):
        """Post-iteration learning."""
        self.history.append({
            'iteration': self.iteration,
            'state': context.state.value,
            'result': result,
            'timestamp': time.time()
        })
        
        if self.iteration % 100 == 0:
            self._self_evaluate()
    
    def _self_evaluate(self):
        """Evaluate own performance."""
        if len(self.history) < 10:
            return
        
        recent = list(self.history)[-100:]
        
        success_rate = sum(1 for h in recent if h['result'].get('success', False)) / len(recent)
        
        if success_rate < 0.3:
            logger.warning(f"MetaLoop: Low success rate ({success_rate:.0%}), need optimization")
        elif success_rate > 0.7:
            logger.info(f"MetaLoop: High success rate ({success_rate:.0%}), current strategy working")
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time


class AdaptiveScheduler:
    """
    Schedules operations based on learned patterns.
    
    Adapts to market conditions and performance.
    """
    
    def __init__(self):
        self.schedules: Dict[str, Dict] = {}
        self.learned_patterns: Dict[str, List] = {}
        self.current_mode = "normal"
        
        self._init_default_schedules()
        logger.info("AdaptiveScheduler initialized")
    
    def _init_default_schedules(self):
        """Initialize default schedules."""
        self.schedules = {
            'market_hours': {
                'forex': [(0, 5), (8, 12), (13, 17)],
                'crypto': [(0, 24)],
                'gold': [(8, 12), (13, 17)]
            },
            'analysis_frequency': {
                'high_volatility': 0.1,
                'normal': 0.5,
                'low_volatility': 2.0
            }
        }
    
    def should_trade(self, symbol: str, current_time: datetime.datetime) -> Tuple[bool, str]:
        """Determine if should trade based on learned patterns."""
        hour = current_time.hour
        weekday = current_time.weekday()
        
        if weekday >= 5:
            if 'USD' not in symbol and 'BTC' not in symbol and 'ETH' not in symbol:
                return False, "Weekend - market closed"
        
        if symbol in self.learned_patterns:
            patterns = self.learned_patterns[symbol]
            best_hours = [p['hour'] for p in patterns if p['performance'] > 0.6]
            if best_hours and hour not in best_hours:
                return False, f"Hour {hour} not in best hours for {symbol}"
        
        return True, "Trading allowed"
    
    def get_sleep_duration(self, volatility: float) -> float:
        """Get adaptive sleep duration."""
        if volatility > 0.02:
            return self.schedules['analysis_frequency']['high_volatility']
        elif volatility < 0.005:
            return self.schedules['analysis_frequency']['low_volatility']
        return self.schedules['analysis_frequency']['normal']
    
    def learn_from_result(self, symbol: str, hour: int, success: bool):
        """Learn trading patterns from results."""
        if symbol not in self.learned_patterns:
            self.learned_patterns[symbol] = []
        
        existing = next((p for p in self.learned_patterns[symbol] if p['hour'] == hour), None)
        
        if existing:
            existing['total'] += 1
            if success:
                existing['successes'] += 1
            existing['performance'] = existing['successes'] / existing['total']
        else:
            self.learned_patterns[symbol].append({
                'hour': hour,
                'successes': 1 if success else 0,
                'total': 1,
                'performance': 1.0 if success else 0.0
            })


class AdvancedStateMachine:
    """
    Intelligent state machine with transition reasoning.
    """
    
    def __init__(self):
        self.state = SystemState.INITIALIZING
        self.previous_state = None
        self.state_history: deque = deque(maxlen=100)
        self.transition_rules: Dict[SystemState, List[SystemState]] = {}
        
        self._init_transitions()
        logger.info("AdvancedStateMachine initialized")
    
    def _init_transitions(self):
        """Initialize valid state transitions."""
        self.transition_rules = {
            SystemState.INITIALIZING: [SystemState.READY, SystemState.SHUTDOWN],
            SystemState.READY: [SystemState.ANALYZING, SystemState.WAITING, SystemState.HEALING, SystemState.SHUTDOWN],
            SystemState.ANALYZING: [SystemState.TRADING, SystemState.WAITING, SystemState.HEALING],
            SystemState.TRADING: [SystemState.READY, SystemState.ANALYZING, SystemState.HEALING],
            SystemState.WAITING: [SystemState.READY, SystemState.ANALYZING, SystemState.EVOLVING],
            SystemState.HEALING: [SystemState.READY, SystemState.SHUTDOWN],
            SystemState.EVOLVING: [SystemState.READY, SystemState.ANALYZING],
            SystemState.SHUTDOWN: []
        }
    
    def can_transition(self, new_state: SystemState) -> bool:
        """Check if transition is valid."""
        if new_state == self.state:
            return True
        return new_state in self.transition_rules.get(self.state, [])
    
    def transition(self, new_state: SystemState, reason: str = "") -> bool:
        """Attempt state transition."""
        if not self.can_transition(new_state):
            logger.warning(f"Invalid transition: {self.state.value} -> {new_state.value}")
            return False
        
        self.previous_state = self.state
        self.state = new_state
        
        self.state_history.append({
            'from': self.previous_state.value,
            'to': new_state.value,
            'reason': reason,
            'timestamp': time.time()
        })
        
        logger.info(f"State: {self.previous_state.value} -> {new_state.value} ({reason})")
        return True
    
    def get_state_duration(self) -> float:
        """Get time in current state."""
        if not self.state_history:
            return 0.0
        
        last = self.state_history[-1]
        return time.time() - last['timestamp']
    
    def should_evolve(self) -> bool:
        """Determine if system should enter evolution state."""
        if self.state == SystemState.WAITING:
            if self.get_state_duration() > 300:
                return True
        return False


class PerformanceMonitor:
    """
    Continuous performance monitoring with auto-adjustments.
    """
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.latencies: deque = deque(maxlen=100)
        self.start_time = time.time()
        self.alerts: List[Dict] = []
        
        logger.info("PerformanceMonitor initialized")
    
    def record_decision(self, decision: str, latency_ms: float):
        """Record a decision."""
        self.metrics.decisions_made += 1
        self.latencies.append(latency_ms)
        self.metrics.avg_latency_ms = sum(self.latencies) / len(self.latencies)
    
    def record_trade(self, profit: float):
        """Record a trade result."""
        self.metrics.trades_executed += 1
        if profit > 0:
            self.metrics.profitable_trades += 1
        
        self.metrics.win_rate = self.metrics.profitable_trades / max(1, self.metrics.trades_executed)
    
    def record_error(self, error: str):
        """Record an error."""
        self.metrics.errors += 1
        
        if self.metrics.errors > 10:
            self.alerts.append({
                'type': 'high_error_rate',
                'message': f'High error rate: {self.metrics.errors} errors',
                'timestamp': time.time()
            })
    
    def record_recovery(self):
        """Record a recovery."""
        self.metrics.recoveries += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        self.metrics.uptime_seconds = time.time() - self.start_time
        return self.metrics
    
    def get_health_score(self) -> float:
        """Calculate overall health score."""
        score = 1.0
        
        if self.metrics.avg_latency_ms > 500:
            score -= 0.2
        
        if self.metrics.win_rate < 0.4:
            score -= 0.2
        
        if self.metrics.errors > 5:
            score -= 0.2
        
        return max(0.0, score)
    
    def needs_healing(self) -> bool:
        """Check if system needs healing."""
        return self.get_health_score() < 0.5


class SelfHealingManager:
    """
    Manages self-healing for the OmegaSystem.
    """
    
    def __init__(self):
        self.healing_history: List[Dict] = []
        self.known_issues: Dict[str, Dict] = {}
        
        logger.info("SelfHealingManager initialized")
    
    def diagnose(self, metrics: PerformanceMetrics) -> List[str]:
        """Diagnose issues from metrics."""
        issues = []
        
        if metrics.errors > 5:
            issues.append("high_error_rate")
        
        if metrics.avg_latency_ms > 1000:
            issues.append("high_latency")
        
        if metrics.win_rate < 0.3 and metrics.trades_executed > 20:
            issues.append("poor_performance")
        
        return issues
    
    def heal(self, issues: List[str]) -> Dict[str, Any]:
        """Apply healing actions."""
        actions = {}
        
        for issue in issues:
            if issue == "high_error_rate":
                actions['reduce_complexity'] = True
                actions['enable_safe_mode'] = True
            
            elif issue == "high_latency":
                actions['reduce_analysis_depth'] = True
                actions['increase_cache'] = True
            
            elif issue == "poor_performance":
                actions['switch_mode'] = 'conservative'
                actions['reduce_position_size'] = True
        
        self.healing_history.append({
            'issues': issues,
            'actions': actions,
            'timestamp': time.time()
        })
        
        logger.info(f"Healing applied: {actions}")
        return actions
    
    def remember_issue(self, issue: str, solution: str, success: bool):
        """Remember how to solve an issue."""
        if issue not in self.known_issues:
            self.known_issues[issue] = {'solutions': {}, 'count': 0}
        
        self.known_issues[issue]['count'] += 1
        
        if solution not in self.known_issues[issue]['solutions']:
            self.known_issues[issue]['solutions'][solution] = {'success': 0, 'total': 0}
        
        self.known_issues[issue]['solutions'][solution]['total'] += 1
        if success:
            self.known_issues[issue]['solutions'][solution]['success'] += 1


class OmegaAGICore:
    """
    Core AGI components for OmegaSystem.
    
    Integrates all AGI capabilities.
    """
    
    def __init__(self):
        self.meta_loop = MetaExecutionLoop()
        self.scheduler = AdaptiveScheduler()
        self.state_machine = AdvancedStateMachine()
        self.monitor = PerformanceMonitor()
        self.healer = SelfHealingManager()
        
        self.recent_decisions: deque = deque(maxlen=100)
        
        logger.info("OmegaAGICore initialized")
    
    def pre_tick(self, tick: Dict, config: Dict) -> Dict[str, Any]:
        """Pre-tick AGI processing."""
        context = ExecutionContext(
            iteration=self.meta_loop.iteration,
            timestamp=time.time(),
            state=self.state_machine.state,
            metrics=self.monitor.get_metrics(),
            recent_decisions=list(self.recent_decisions),
            market_conditions=tick
        )
        
        adjustments = self.meta_loop.pre_iteration(context)
        
        if self.monitor.needs_healing():
            issues = self.healer.diagnose(self.monitor.metrics)
            if issues:
                healing = self.healer.heal(issues)
                adjustments.update(healing)
                self.state_machine.transition(SystemState.HEALING, "Auto-healing triggered")
        
        return adjustments
    
    def post_tick(self, decision: str, result: Dict):
        """Post-tick AGI learning."""
        self.recent_decisions.append(decision)
        
        context = ExecutionContext(
            iteration=self.meta_loop.iteration,
            timestamp=time.time(),
            state=self.state_machine.state,
            metrics=self.monitor.get_metrics(),
            recent_decisions=list(self.recent_decisions),
            market_conditions={}
        )
        
        self.meta_loop.post_iteration(context, result)
        
        if result.get('trade_executed'):
            self.monitor.record_trade(result.get('profit', 0))
        
        if self.state_machine.should_evolve():
            self.state_machine.transition(SystemState.EVOLVING, "Time to evolve")
    
    def should_trade_now(self, symbol: str) -> Tuple[bool, str]:
        """Check if should trade now."""
        return self.scheduler.should_trade(symbol, datetime.datetime.now())
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGI core status."""
        return {
            'state': self.state_machine.state.value,
            'iteration': self.meta_loop.iteration,
            'uptime': self.meta_loop.get_uptime(),
            'health_score': self.monitor.get_health_score(),
            'metrics': {
                'decisions': self.monitor.metrics.decisions_made,
                'trades': self.monitor.metrics.trades_executed,
                'win_rate': self.monitor.metrics.win_rate,
                'errors': self.monitor.metrics.errors
            }
        }
