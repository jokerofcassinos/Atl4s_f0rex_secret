"""
AGI Ultra-Complete: Config AGI Components

Sistema de Configuração Auto-Adaptativo:
- DynamicConfigurationEngine: Config dinâmica
- ContextAwareConfigManager: Config context-aware
- ConfigPerformanceLearner: Aprende configs ideais
- MultiObjectiveConfigOptimizer: Otimização multi-objetivo
- AutomaticParameterTuner: Auto-tuning
"""

import logging
import time
import copy
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

logger = logging.getLogger("ConfigAGI")


class MarketContext(Enum):
    HIGH_VOLATILITY = "high_volatility"
    NORMAL = "normal"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"
    CRISIS = "crisis"


@dataclass
class ConfigSnapshot:
    """Configuration snapshot."""
    config: Dict[str, Any]
    context: MarketContext
    performance: float
    timestamp: float


@dataclass
class ConfigRecommendation:
    """Configuration recommendation."""
    parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float
    reason: str


class DynamicConfigurationEngine:
    """Manages dynamic configuration changes."""
    
    def __init__(self, base_config: Dict[str, Any] = None):
        self.base_config = base_config or {}
        self.current_config = copy.deepcopy(self.base_config)
        self.config_history: deque = deque(maxlen=1000)
        
        self.constraints: Dict[str, Dict] = {}
        self._init_constraints()
        
        logger.info("DynamicConfigurationEngine initialized")
    
    def _init_constraints(self):
        """Initialize parameter constraints."""
        self.constraints = {
            'virtual_sl': {'min': 1.0, 'max': 100.0, 'type': float},
            'virtual_tp': {'min': 0.5, 'max': 50.0, 'type': float},
            'spread_limit': {'min': 0.001, 'max': 0.1, 'type': float},
            'max_positions': {'min': 1, 'max': 20, 'type': int},
            'confidence_threshold': {'min': 0.5, 'max': 0.99, 'type': float}
        }
    
    def update(self, parameter: str, value: Any) -> bool:
        """Update configuration parameter."""
        if parameter in self.constraints:
            constraint = self.constraints[parameter]
            
            try:
                value = constraint['type'](value)
            except:
                logger.warning(f"Invalid type for {parameter}: {value}")
                return False
            
            if value < constraint['min'] or value > constraint['max']:
                logger.warning(f"Value {value} out of range for {parameter}")
                return False
        
        old_value = self.current_config.get(parameter)
        self.current_config[parameter] = value
        
        self.config_history.append({
            'parameter': parameter,
            'old': old_value,
            'new': value,
            'timestamp': time.time()
        })
        
        logger.info(f"Config updated: {parameter} = {value}")
        return True
    
    def get(self, parameter: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        return self.current_config.get(parameter, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return copy.deepcopy(self.current_config)
    
    def reset(self):
        """Reset to base configuration."""
        self.current_config = copy.deepcopy(self.base_config)


class ContextAwareConfigManager:
    """Manages context-aware configuration."""
    
    def __init__(self):
        self.context_configs: Dict[MarketContext, Dict[str, Any]] = {}
        self.current_context = MarketContext.NORMAL
        
        self._init_context_configs()
        
        logger.info("ContextAwareConfigManager initialized")
    
    def _init_context_configs(self):
        """Initialize context-specific configurations."""
        self.context_configs = {
            MarketContext.HIGH_VOLATILITY: {
                'virtual_sl': 30.0,
                'virtual_tp': 5.0,
                'confidence_threshold': 0.85,
                'max_positions': 3
            },
            MarketContext.NORMAL: {
                'virtual_sl': 15.0,
                'virtual_tp': 3.0,
                'confidence_threshold': 0.75,
                'max_positions': 5
            },
            MarketContext.LOW_VOLATILITY: {
                'virtual_sl': 8.0,
                'virtual_tp': 1.5,
                'confidence_threshold': 0.70,
                'max_positions': 7
            },
            MarketContext.TRENDING: {
                'virtual_sl': 20.0,
                'virtual_tp': 8.0,
                'confidence_threshold': 0.70,
                'max_positions': 5
            },
            MarketContext.RANGING: {
                'virtual_sl': 10.0,
                'virtual_tp': 2.0,
                'confidence_threshold': 0.85,
                'max_positions': 3
            },
            MarketContext.CRISIS: {
                'virtual_sl': 50.0,
                'virtual_tp': 10.0,
                'confidence_threshold': 0.95,
                'max_positions': 1
            }
        }
    
    def detect_context(self, volatility: float, trend_strength: float = 0.0) -> MarketContext:
        """Detect market context."""
        if volatility > 0.03:
            return MarketContext.HIGH_VOLATILITY
        elif volatility < 0.005:
            return MarketContext.LOW_VOLATILITY
        elif abs(trend_strength) > 0.7:
            return MarketContext.TRENDING
        elif abs(trend_strength) < 0.2:
            return MarketContext.RANGING
        return MarketContext.NORMAL
    
    def get_config_for_context(self, context: MarketContext = None) -> Dict[str, Any]:
        """Get configuration for context."""
        context = context or self.current_context
        return copy.deepcopy(self.context_configs.get(context, {}))
    
    def update_context(self, context: MarketContext):
        """Update current context."""
        if context != self.current_context:
            logger.info(f"Context changed: {self.current_context.value} -> {context.value}")
            self.current_context = context


class ConfigPerformanceLearner:
    """Learns which configurations work best."""
    
    def __init__(self):
        self.performance_data: Dict[str, List[Dict]] = defaultdict(list)
        self.optimal_configs: Dict[str, Any] = {}
        
        logger.info("ConfigPerformanceLearner initialized")
    
    def record_performance(self, config: Dict[str, Any], context: MarketContext, performance: float):
        """Record configuration performance."""
        key = f"{context.value}"
        
        self.performance_data[key].append({
            'config': copy.deepcopy(config),
            'performance': performance,
            'timestamp': time.time()
        })
        
        if len(self.performance_data[key]) > 100:
            self.performance_data[key] = self.performance_data[key][-100:]
        
        self._update_optimal(key)
    
    def _update_optimal(self, key: str):
        """Update optimal configuration for key."""
        if len(self.performance_data[key]) < 5:
            return
        
        best = max(self.performance_data[key], key=lambda x: x['performance'])
        self.optimal_configs[key] = best['config']
    
    def get_optimal_config(self, context: MarketContext) -> Optional[Dict[str, Any]]:
        """Get optimal configuration for context."""
        key = f"{context.value}"
        return self.optimal_configs.get(key)
    
    def get_recommendations(self, current_config: Dict, context: MarketContext) -> List[ConfigRecommendation]:
        """Get configuration recommendations."""
        recommendations = []
        
        optimal = self.get_optimal_config(context)
        if not optimal:
            return recommendations
        
        for param, value in optimal.items():
            current = current_config.get(param)
            if current != value:
                recommendations.append(ConfigRecommendation(
                    parameter=param,
                    current_value=current,
                    recommended_value=value,
                    confidence=0.7,
                    reason=f"Optimal for {context.value}"
                ))
        
        return recommendations


class AutomaticParameterTuner:
    """Automatic parameter tuning."""
    
    def __init__(self):
        self.tuning_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.current_direction: Dict[str, int] = {}
        
        logger.info("AutomaticParameterTuner initialized")
    
    def tune(self, parameter: str, current_value: float, 
             performance: float, step_size: float = 0.1) -> float:
        """Tune parameter based on performance."""
        self.tuning_history[parameter].append({
            'value': current_value,
            'performance': performance,
            'timestamp': time.time()
        })
        
        if len(self.tuning_history[parameter]) < 3:
            return current_value
        
        recent = list(self.tuning_history[parameter])[-3:]
        
        performances = [r['performance'] for r in recent]
        
        if performances[-1] > performances[-2]:
            direction = self.current_direction.get(parameter, 1)
            self.current_direction[parameter] = direction
            return current_value + (step_size * direction)
        elif performances[-1] < performances[-2] * 0.95:
            self.current_direction[parameter] = -self.current_direction.get(parameter, 1)
            return current_value + (step_size * self.current_direction[parameter])
        
        return current_value


class ConfigAGI:
    """
    Main Config AGI System.
    """
    
    def __init__(self, base_config: Dict[str, Any] = None):
        self.dynamic = DynamicConfigurationEngine(base_config)
        self.context = ContextAwareConfigManager()
        self.learner = ConfigPerformanceLearner()
        self.tuner = AutomaticParameterTuner()
        
        self.snapshots: deque = deque(maxlen=100)
        
        logger.info("ConfigAGI initialized")
    
    def get_adaptive_config(self, volatility: float = 0.01, 
                           trend_strength: float = 0.0) -> Dict[str, Any]:
        """Get adaptive configuration."""
        detected = self.context.detect_context(volatility, trend_strength)
        self.context.update_context(detected)
        
        base = self.dynamic.get_all()
        context_config = self.context.get_config_for_context(detected)
        
        for param, value in context_config.items():
            base[param] = value
        
        optimal = self.learner.get_optimal_config(detected)
        if optimal:
            for param, value in optimal.items():
                if param in base:
                    base[param] = (base[param] + value) / 2
        
        return base
    
    def record_performance(self, performance: float):
        """Record performance for learning."""
        snapshot = ConfigSnapshot(
            config=self.dynamic.get_all(),
            context=self.context.current_context,
            performance=performance,
            timestamp=time.time()
        )
        self.snapshots.append(snapshot)
        
        self.learner.record_performance(
            snapshot.config,
            snapshot.context,
            performance
        )
    
    def get_recommendations(self) -> List[ConfigRecommendation]:
        """Get configuration recommendations."""
        return self.learner.get_recommendations(
            self.dynamic.get_all(),
            self.context.current_context
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get AGI status."""
        return {
            'current_config': self.dynamic.get_all(),
            'context': self.context.current_context.value,
            'snapshots': len(self.snapshots),
            'optimal_configs': len(self.learner.optimal_configs)
        }
