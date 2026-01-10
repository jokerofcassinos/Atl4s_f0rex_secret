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
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from core.agi.big_beluga.correlation import CorrelationSynapse
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
                'condition': lambda ctx: ctx.metrics.win_rate < 0.25 and ctx.metrics.trades_executed > 20,
                'action': 'switch_to_conservative',
                'description': 'Critical Low win rate, switching to conservative mode'
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
                # logger.debug(f"MetaLoop Adjustment: {rule['description']}")
        
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
        
        # Filter for actual Trade Attempts (ignore idle ticks)
        # Key: We look for 'trade_action' in the result dict
        active_attempts = [h for h in recent if h['result'].get('trade_action', False)]
        
        if not active_attempts:
             return
             
        success_rate = sum(1 for h in active_attempts if h['result'].get('success', False)) / len(active_attempts)
        
        if len(active_attempts) >= 5: # Only judge if we have a sample
            if success_rate < 0.3:
                logger.warning(f"MetaLoop: Low active success rate ({success_rate:.0%}), need optimization")
            elif success_rate > 0.7:
                logger.info(f"MetaLoop: High active success rate ({success_rate:.0%}), current strategy working")
    
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


from core.agi.infinite_why_engine import InfiniteWhyEngine

from .simulation_system_agi import SimulationSystemAGI
from .temporal import FractalTimeScaleIntegrator, ChronosPattern, QuarterlyCycle # Phase 7
from .abstraction import AbstractPatternSynthesizer # Phase 7
from .synergy import AlphaSynergySwarm # Phase 8
from .quantum import QuantumProbabilityCollapser # Phase 9
from .fuzzy_logic import FuzzyLogicEngine
from .neuro_plasticity import NeuroPlasticityEngine
from .causal_nexus import CausalNexus
from .metacognition import RecursiveReflectionLoop
from .active_inference.generative_model import GenerativeModel
from .active_inference.free_energy import FreeEnergyMinimizer
from .resonance import ResonanceEngine
from .neuro_linguistics import NeuroLinguisticDriver
from .zero_shot import ZeroShotAnalyst
from .conflict_resolution import SwarmConflictResolver # Phase 155 (System #9)

# Verified AGI Modules
from .logic import SymbolicReasoningModule # Phase 9
from .plasticity import SelfModificationHeuristic # Phase 10
from .metacognition import ConfidenceCalibrator # Phase 10

# Phase 150+: Session & Liquidity Fusion
from analysis.session_liquidity_fusion import (
    TransitionDynamics, VoidNavigator, FootprintSynthesizer,
    FlowHolography, RegimeAdaptation
)
from .learning.ssl_engine import SelfSupervisedLearningEngine
from .metacognition.ontological_nuance_processor import OntologicalNuanceProcessor
from .symbiosis.neural_resonance_bridge import NeuralResonanceBridge
from .metacognition.regime_detector import RegimeDetector # Phase 5

# Core Dependencies
from core.mcts_planner import MCTSPlanner
from core.agi.self_healing.self_healing_system import SelfHealingSystem
from core.holographic_memory import HolographicMemory
from core.hyper_dimensional import HyperDimensionalEngine

logger = logging.getLogger("OmegaAGICore")


from core.agi.microstructure.flux_heatmap import FluxHeatmap
from core.agi.microstructure.liquidity_blackhole import LiquidityBlackHole
from core.agi.microstructure.volume_resonance import VolumeResonance
from core.agi.microstructure.elastic_spread import ElasticSpreadDynamics
from core.agi.microstructure.chrono_session import ChronoSessionOverlap

from core.agi.big_beluga.market_echo import MarketEchoScreener
from core.agi.big_beluga.regime_filter import RegimeFilter
from core.agi.big_beluga.fractal_trend import FractalTrend
from core.agi.big_beluga.volume_delta import VolumeDelta
from core.agi.big_beluga.liquidity_spectrum import LiquiditySpectrum

from core.agi.pan_cognitive.infinite_reflection import InfiniteRecursiveReflection
from core.agi.pan_cognitive.empathic_bridge import HighFidelityResonance
from core.agi.big_beluga.correlation import CorrelationSynapse
from core.agi.big_beluga.range_scanner import RangeScanner # Phase 17: Range Master
from core.agi.physics.time_distortion import TimeDistortionEngine # Phase 19: Market Physics
from core.agi.big_beluga.power_of_3 import PowerOf3Analyzer # Phase 19: AMD
from core.agi.big_beluga.power_of_3 import PowerOf3Analyzer # Phase 19: AMD
from core.agi.pan_cognitive.composite_operator import CompositeOperator # Phase 20: The Institute
from core.agi.big_beluga.snr_matrix import SNRMatrix # Phase 21: SNR
from core.agi.big_beluga.msnr_alchemist import MSNRAlchemist # Phase 22: MSNR
from core.agi.pan_cognitive.causal_inference import CausalInferenceEngine
from core.agi.pan_cognitive.neuro_plasticity_v2 import NeuroPlasticityV2

class OmegaAGICore:
    """
    Núcleo Central da AGI (Omega Protocol v5.0 - Symbiotic)
    """
    def __init__(self, infinite_why_engine=None, simulation_system=None, memory_file: str = None):
        self.infinite_why_engine = infinite_why_engine
        self.simulation_system = simulation_system
        self.memory_file = memory_file # Store override
        self.meta_loop = MetaExecutionLoop()
        
        # Core Memory & Vector Engines (Required for Generative Model)
        self.holographic_memory = HolographicMemory(memory_file=memory_file) if memory_file else HolographicMemory()
        self.scheduler = AdaptiveScheduler()
        self.state_machine = AdvancedStateMachine()
        self.monitor = PerformanceMonitor()
        self.correlation_synapse = CorrelationSynapse()
        
        # --- AGI Expansion: Phase 6 & 7 (Singularity & Deep Logic) ---
        # Microstructure Systems (Group A & B)
        self.flux_heatmap = FluxHeatmap()
        self.liquidity_blackhole = LiquidityBlackHole()
        self.volume_resonance = VolumeResonance()
        self.elastic_spread = ElasticSpreadDynamics()
        self.chrono_session = ChronoSessionOverlap()
        
        # BigBeluga Integrated Systems (Group C)
        self.market_echo = MarketEchoScreener()
        self.regime_filter = RegimeFilter()
        self.fractal_trend = FractalTrend()
        self.volume_delta = VolumeDelta()
        self.liquidity_spectrum = LiquiditySpectrum()
        
        # Deep AGI Pan-Cognitive Systems (Group D)
        self.infinite_reflection = InfiniteRecursiveReflection()
        self.empathic_resonance = HighFidelityResonance()
        self.causal_inference = CausalInferenceEngine()
        self.neuro_plasticity = NeuroPlasticityV2()
        
        logger.info("OmegaAGICore initialized with InfiniteWhyEngine & SimulationSystem")
        logger.info("Singularity Expansion: Microstructure & BigBeluga Systems Online.")
        # Phase 13: Active Inference
        from core.agi.active_inference.generative_model import GenerativeModel
        from core.agi.active_inference.free_energy import FreeEnergyMinimizer
        self.generative_model = GenerativeModel()
        self.free_energy = FreeEnergyMinimizer()

        logger.info("Deep AGI: Pan-Cognitive Matrix (Reflection, Resonance, Causal, Neuroplasticity) Active.")
        self.healer = SelfHealingSystem()
        
        # Project Awakening: The Causal Engine
        self.why_engine = InfiniteWhyEngine(
            max_depth=16, # Start with reasonable depth
            parallel_workers=4,
            enable_meta_reasoning=True
        )
        self.simulation = SimulationSystemAGI()
        # from core.agi.temporal import FractalTimeScaleIntegrator, ChronosPattern, QuarterlyCycle (Moved to top)
        self.temporal = FractalTimeScaleIntegrator()
        self.chronos = ChronosPattern(utc_offset_hours=2) 
        self.quarterly = QuarterlyCycle(utc_offset_hours=2)
        self.abstraction = AbstractPatternSynthesizer()
        self.synergy = AlphaSynergySwarm()
        self.quantum = QuantumProbabilityCollapser()
        self.fuzzy = FuzzyLogicEngine()


        # ... (rest of method)
        

        self.hyper_dimensional = HyperDimensionalEngine()

        # Phase 130+ Upgrades
        self.neuroplasticity = NeuroPlasticityEngine()
        self.causal_nexus = CausalNexus()
        self.metacognition = RecursiveReflectionLoop()
        # Pass dependencies to Generative Model

        # self.generative_model already initialized above
        self.resonance = ResonanceEngine()

        self.voice = NeuroLinguisticDriver()
        self.zero_shot = ZeroShotAnalyst()
        
        # Innovation Level 9/10: Logic & Plasticity
        self.logic = SymbolicReasoningModule()
        self.plasticity = SelfModificationHeuristic()
        self.calibrator = ConfidenceCalibrator()
        
        # Innovation Level 11: Civil War Resolution (System #9)
        self.conflict_resolver = SwarmConflictResolver(
            causal_nexus=self.causal_nexus,
            temporal_integrator=self.temporal,
            symbolic_logic=self.logic
        )
        
        # --- PHASE 150+: HYPER-COMPLEX FUSION (50+ Sub-systems) ---
        self.transition_dynamics = TransitionDynamics()
        self.void_navigator = VoidNavigator()
        self.footprint_synthesizer = FootprintSynthesizer()
        self.flow_holography = FlowHolography()
        self.regime_adaptation = RegimeAdaptation()
        
        self.ssl_engine = SelfSupervisedLearningEngine()
        self.ontology_nuance = OntologicalNuanceProcessor()
        self.resonance_bridge = NeuralResonanceBridge()
        self.regime_detector = RegimeDetector() # Phase 5
        
        self.quarterly_cycle = QuarterlyCycle()
        self.correlation = CorrelationSynapse()
        self.range_scanner = RangeScanner() # Phase 17
        self.time_engine = TimeDistortionEngine() # Phase 19
        self.power_of_3 = PowerOf3Analyzer() # Phase 19
        self.composite_operator = CompositeOperator() # Phase 20
        self.snr_matrix = SNRMatrix() # Phase 21
        self.msnr_alchemist = MSNRAlchemist() # Phase 22
        self.recent_decisions: deque = deque(maxlen=100)
        self.causal_engine = CausalInferenceEngine() # (Phase 6)
        
        logger.info("OmegaAGI Core v5.0 (Symbiotic) Initialized.")
        
        logger.info("OmegaAGICore initialized with InfiniteWhyEngine & SimulationSystem")
        self.last_state = None
        self.learning = None # HistoryLearningEngine reference
        self.resonance_bridge = None # NeuralResonanceBridge reference (Phase 150)

    def connect_learning_engine(self, engine):
        """
        Connects the HistoryLearningEngine to the AGI Core.
        """
        self.learning = engine
        logger.info("AGI Core connected to HistoryLearningEngine.")

    def post_tick(self, decision: str, feedback: Dict[str, Any]):
        """
        Called after decision execution to update learning.
        """
        # Update Meta-Execution Loop
        self.meta_loop.update(feedback)
        
        # Update History Learning
        if self.learning:
            self.learning.update_active_trades(feedback)
            
    def pre_tick(self, tick: Dict, config: Dict, market_data_map: Dict = None) -> Dict[str, Any]:
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

        # --- PHASE 8: SENSORY INTEGRATION (The Optic Nerve) ---
        
        # 0. CHRONOS TIME FRACTAL (NY Session + Quarterly IPDA)
        chronos_context = {}
        if market_data_map:
             sym = config.get('symbol', 'XAUUSD')
             # Session
             chronos_data = self.chronos.analyze_session_fractal(tick, market_data_map, symbol=sym)
             # Quarterly (IPDA)
             quarterly_data = self.quarterly.analyze_cycle(tick)
             
             # Merge
             chronos_context = {**chronos_data, **quarterly_data}
             adjustments['chronos_narrative'] = chronos_context
             
        # 3. Time Fractal Analysis (Chronos)
        chronos_context = self.chronos.analyze_session_fractal(tick, market_data_map, config.get('symbol', 'XAUUSD'))
        adjustments['chronos_narrative'] = chronos_context

        # 4. Range Analysis (The Impossible - Phase 17)
        range_data = {}
        if market_data_map and 'M5' in market_data_map:
             range_data = self.range_scanner.analyze(market_data_map['M5'])
             adjustments['range_analysis'] = range_data
             
             # --- SNIPER PROTOCOL v4.0: GATE 4 (REGIME CHECK) ---
             # Detect DANGEROUS_CHOP to prevent trading in noise.
             # 1. Candles pequenos (Dojis)
             # 2. Alternância de cores (Indecisão)
             df_m5 = market_data_map['M5']
             if len(df_m5) >= 10:
                  last_5 = df_m5.iloc[-5:]
                  bodies = abs(last_5['close'] - last_5['open'])
                  ranges = last_5['high'] - last_5['low']
                  avg_body_ratio = (bodies / ranges.replace(0, 0.0001)).mean()
                  
                  colors = (last_5['close'] > last_5['open']).astype(int)
                  alternations = abs(colors.diff()).sum()
                  
                  # Relaxed Thresholds v4.1
                  # Body ratio < 0.25 (was 0.35) -> Only block Extreme/Micro Dojis
                  # Alternations >= 4 (was 3) -> Require almost perfect alternating colors
                  if avg_body_ratio < 0.25 and alternations >= 4:
                       logger.warning("AGI REGIME: DANGEROUS_CHOP Detected. Blocking Trades.")
                       adjustments['regime_block'] = True
                       adjustments['regime_reason'] = "DANGEROUS_CHOP"

             if range_data['status'] == 'RANGING':
                  logger.info(f"RANGE SCANNER: Market is RANGING (Str: {range_data['strength']:.2f}). Bias: {range_data['proximity']}")
                  # Add Ping Pong Bias to adjustments
                  if range_data['proximity'] == 'RANGE_LOW':
                       adjustments['bias_override'] = 'BUY'
                  elif range_data['proximity'] == 'RANGE_HIGH':
                       adjustments['bias_override'] = 'SELL'
                       
        # 5. Physics & Institute Analysis (Phase 19/20)
        time_data = self.time_engine.process_tick(tick)
        adjustments['time_physics'] = time_data
        
        amd_data = {}
        if market_data_map and 'M5' in market_data_map:
             amd_data = self.power_of_3.analyze(market_data_map['M5'])
             adjustments['amd_structure'] = amd_data
             
        operator_profile = self.composite_operator.profile_market_behavior(tick, market_data_map, range_data)
        adjustments['operator_profile'] = operator_profile
        
        
        if time_data['time_state'] in ['WARP_EVENT', 'HFT_ACTIVITY']:
             logger.warning(f"PHYSICS: TIME WARP DETECTED! Velocity: {time_data['velocity']:.1f} TPS. Factor: {time_data['warp_factor']:.1f}x")
             
        # 6. SNR Matrix & MSNR Alchemist (Phase 21/22)
        if market_data_map and 'M5' in market_data_map:
             raw_levels = self.snr_matrix.scan_structure(market_data_map['M5'])
             golden_zones = self.msnr_alchemist.transmute(raw_levels, tick.get('bid', 0))
             
             confluence = self.msnr_alchemist.detect_confluence(golden_zones, tick.get('bid', 0))
             adjustments['structure_confluence'] = confluence
             
             if confluence['in_zone']:
                  logger.info(f"MSNR ALCHEMIST: Price inside GOLDEN ZONE (Score: {confluence['nearest_zone_score']:.1f}). Prepare for Impact.")
        
        # 7. Global Correlation
        if market_data_map and 'global_basket' in market_data_map:
             sym = config.get('symbol', 'XAUUSD')
             adjustments['symbol'] = sym # Save for later
             corr_data = self.correlation.analyze_correlations(sym, market_data_map['global_basket'])
             adjustments['global_risk'] = corr_data

        # 0.2 CAUSAL INFERENCE (Why?)
        sentiment_score = 0.5; high_impact_prob = 0.0
        causal_events = {'sentiment_score': sentiment_score, 'impact': high_impact_prob}
        if hasattr(self, 'causal_inference'):
             causa = self.causal_inference.infer_cause(tick, causal_events, chronos_context=chronos_context)
             ontological_narrative = causa.get('ontological_layer', 'STANDARD')
             if ontological_narrative != "STANDARD_MECHANICS":
                  logger.info(f"AGI DEEP THOUGHT: [{ontological_narrative}] -> {causa['confidence']:.2f}")

             adjustments['causal_root'] = causa['root_cause']
             adjustments['causal_chain'] = causa['causal_chain']
             adjustments['ontological_nuance'] = ontological_narrative


        # 1. Microstructure Analysis
        heatmap_metrics = self.flux_heatmap.update(tick)
        liquidity_metrics = self.liquidity_blackhole.analyze(tick, heatmap_metrics)
        
        # 2. BigBeluga Resonance
        # Assuming we have access to historical arrays via 'market_data_map' or similar if needed
        # For now, we pass the tick to update internal state of Beluga systems
        beluga_signals = {}
        if market_data_map:
             # Example: Extract close prices for fractal analysis
             # This assumes market_data_map has 'close', 'high', 'low' arrays
             pass 
             
        # --- PHASE 12: PROFITABILITY FRICTION CHECK ---
        # (Implicitly handled by BigBeluga or GreatFilter via context)
        
        # --- PHASE 13: ACTIVE INFERENCE (DREAMING) ---
        # 1. Dream: What do we expect the market to look like?
        # Note: In a real loop, we'd dream BEFORE the tick. Here we simulate the cycle.
        # We model "Next Tick" based on "Previous Beliefs"
        current_price = tick.get('bid', 1.0)
        dream = self.generative_model.dream_next_tick(current_price) 
        
        # 2. Reality Check: Calculate Surprise (Free Energy)
        # For prototype, we compare Dream vs Current Tick (Self-Correction)
        surprise = self.free_energy.calculate_surprise(dream, tick)
        
        # 3. Model Update (Perceptual Learning)
        self.generative_model.update_beliefs(tick, surprise)
        
        if surprise > 5.0:
            logger.warning(f"HIGH SURPRISE ({surprise:.2f})! Plasticity Triggered.")
            # Trigger rapid neuroplasticity
            if self.neuroplasticity:
                self.neuroplasticity.adapt({'surprise_shock': surprise})
        
        context.active_inference = {
            "dream": dream,
            "surprise": surprise,
            "beliefs": self.generative_model.beliefs
        }

        # --- PHASE 11: HOLOGRAPHIC RECALL ---
        # "Deja Vu" - Check if we have seen this before
        memory_echo = self.holographic_memory.retrieve_associative_memory(
            {**tick, 'metrics': context.metrics}, k=5
        )
        if memory_echo['confidence'] > 0.7:
            logger.info(f"HOLOGRAPHIC RECALL: Similar scenario found! Exp Outcome: {memory_echo['expected_outcome']:.2f}")
             
        # Inject into Context for GrandMaster
        context.microstructure = {**heatmap_metrics, **liquidity_metrics}
        context.big_beluga = beluga_signals
        context.memory_echo = memory_echo
        
        # --- PHASE 10: PLASTICITY CHECK ---
        # The Brain rewrites itself based on pain/pleasure (metrics)
        plasticity_mods = self.plasticity.evaluate_and_adapt(context.metrics.__dict__, config)
        if plasticity_mods:
             # Apply modifications to the 'adjustments' packet for main.py to handle
             # or apply directly if we had a config object reference
             adjustments.update(plasticity_mods)
             
        # --- PHASE 7: DEEP AGI NEUROPLASTICITY V2 ---
        # Real-time synaptic re-weighting
        neuro_mods = self.neuro_plasticity.adapt(context.metrics.__dict__)
        if neuro_mods.get('weight_adjustments'):
             logger.info(f"NEUROPLASTICITY: Rewired Synapses -> {neuro_mods['weight_adjustments']}")
             logger.info(f"AGI PLASTICITY: Self-Modification Triggered: {plasticity_mods}")

        # --- PHASE 7: SINGULARITY REASONING (Temporal & Abstract) ---
        if market_data_map:
            # 1. Temporal Coherence
            coherence = self.temporal.calculate_fractal_coherence(market_data_map)
            # If Fractals Disagree, reduce risk
            if abs(coherence) < 0.2:
                 adjustments['reduce_position_size'] = True
                 # logger.debug(f"Fractal Incoherence ({coherence:.2f}). Reducing Size.")
            
            # 2. Time Dilation (Fake Breakout Check)
            dilation = self.temporal.detect_temporal_dilation(market_data_map)
            if dilation == "DILATION_FAST":
                 # M1 exploding, H1 sleeping -> High Volatility Noise
                 adjustments['widen_stops'] = True
                 logger.info("TEMPORAL DILATION DETECTED: Time is speeding up (M1 >> H1). Widening stops.")

            # 3. Abstract Pattern Recognition
            # Use M1 Close prices for now
            if '1m' in market_data_map:
                df_m1 = market_data_map['1m']
                if not df_m1.empty:
                    prices = df_m1['close'].tail(15).tolist() # Last 15 candles
                    sig = self.abstraction.synthesize_signature(prices)
                    pattern, score = self.abstraction.find_structural_similarity(sig)
                    
                    if pattern == "COMPRESSION" and score > 0.7:
                        adjustments['prepare_breakout'] = True
                        logger.info(f"ABSTRACT PATTERN: {pattern} detected (Score {score:.2f}). Preparing for Move.")
        
        if self.monitor.needs_healing():
            issues = self.healer.diagnose(self.monitor.metrics)
            if issues:
                healing = self.healer.heal(issues)
                adjustments.update(healing)
                self.state_machine.transition(SystemState.HEALING, "Auto-healing triggered")
                
                self.state_machine.transition(SystemState.HEALING, "Auto-healing triggered")

        # --- PHASE 6: EMPIRICAL HISTORY CHECK ---
        if self.learning:
            stats = self.learning.analyze_patterns()
            if stats['win_rate'] < 0.25 and stats['total_trades'] > 20:
                 adjustments['risk_mode'] = 'CONSERVATIVE'
                 adjustments['switch_mode'] = 'CONSERVATIVE' # Pivot to safety
                 logger.warning(f"AGI HISTORY WARNING: Critical Low Win Rate ({stats['win_rate']:.2f}). Pivot -> CONSERVATIVE.")
                 
            elif stats['win_rate'] > 0.70 and stats['total_trades'] > 15:
                 adjustments['switch_mode'] = 'WOLF_PACK' # Reward high performance
                 logger.info(f"AGI HISTORY: High Win Rate ({stats['win_rate']:.2f}). Pivot -> WOLF_PACK.")

        # --- AGI DREAMER: MENTAL SIMULATION ---
        # "Imagine the next 10 minutes..."
        try:
            current_state = {
                'price': tick.get('bid', 0),
                'value': tick.get('bid', 0), # Generic value
            }
            if current_state['price'] > 0:
                sim_result = self.simulation.full_mental_simulation(current_state)
                adjustments['premonition'] = {
                    'direction': sim_result['recommendation'],
                    'optimistic_outcome': sim_result['scenarios']['optimistic'],
                    'pessimistic_outcome': sim_result['scenarios']['pessimistic'],
                    'monte_carlo_prob': sim_result['monte_carlo']['positive_probability']
                }
        except Exception as e:
            logger.warning(f"Dreamer Failure: {e}")

        # --- PHASE 150+: HYPER-COMPLEX ANALYSIS (Satisfy Metacognition) ---
        try:
            logger.info("OmegaAGICore: Starting Hyper-Complex Analysis...")
            # 1. Session Timing Analysis
            current_session = config.get('session_name', 'UNKNOWN')
            next_session = config.get('next_session', 'UNKNOWN')
            session_meta = self.transition_dynamics.analyze_transition(
                current_session, next_session, tick.get('liquidity_map', {})
            )
            adjustments['session_analysis'] = session_meta
            
            # 2. Liquidity Analysis (Void & Flow)
            l_matrix = tick.get('liquidity_matrix', np.zeros((10, 2)))
            void_meta = self.void_navigator.navigate(l_matrix, tick.get('bid', 0))
            flow_meta = self.flow_holography.analyze_flow(
                tick.get('delta_stream', []), tick
            )
            adjustments['liquidity_check'] = {
                'void_metrics': void_meta,
                'flow_metrics': flow_meta,
                'score': void_meta.get('refill_probability', 0.5)
            }
            
            # 3. Spread Verification
            spread = tick.get('ask', 0) - tick.get('bid', 0)
            adjustments['spread_check'] = {
                'current_spread': spread,
                'is_tight': spread < config.get('max_spread', 10.0)
            }
            
            # 4. Ontological Nuance & SSL Discovery
            nuance = self.ontology_nuance.process_nuance(tick, adjustments)
            adjustments['ontological_nuance'] = nuance
            
            # logger.info(f"OmegaAGICore: Hyper-Complex Analysis Complete. Context Keys: {list(adjustments.keys())}")

            if self.meta_loop.iteration % 100 == 0:
                raw_data = np.array(tick.get('delta_stream', [0.0] * 10))
                self.ssl_engine.discover_features(raw_data)
                
        except Exception as e:
            logger.error(f"Hyper-Complex Analysis Failure: {e}", exc_info=True)
        
        return adjustments

    def synthesize_singularity_decision(self, swarm_signal: Any, market_data_map: Dict[str, Any] = None, agi_context: Dict[str, Any] = None, open_positions: List[Dict] = None) -> Any:
        # --- ADAPTER LAYER (Handle Main.py Dict Input) ---
        is_dict = isinstance(swarm_signal, dict)
        if is_dict:
            from collections import namedtuple
            # Define flexible namedtuple that supports _replace
            GenericSignal = namedtuple('GenericSignal', ['signal_type', 'confidence', 'meta_data'])
            
            # Map input dict to object
            raw_dir = swarm_signal.get('direction', 0)
            s_type = "BUY" if raw_dir == 1 else "SELL" if raw_dir == -1 else "WAIT"
            
            internal_signal = GenericSignal(
                signal_type=s_type, 
                confidence=swarm_signal.get('confidence', 0.0),
                meta_data={}
            )
        else:
            internal_signal = swarm_signal

        # --- CORE LOGIC (Uses internal_signal) ---
        if not self.synergy: return self._format_output(internal_signal, is_dict)
        
        inputs = {
            'swarm_consensus': internal_signal,
        }
        
        # 2. Fractal Trend (Ontological Truth)
        trend_info = {}
        if self.fractal_trend and market_data_map:
             trend_info = self.fractal_trend.analyze({'bid': 0}, market_data_map)
             
        # 3. Metacognitive Reflection (The Mirror)
        # We reflect on the SWARM'S decision
        
        # Merge Pre-Tick Context (Session, Liquidity) with Inputs
        reflection_context = {
                'swarm_votes': inputs.get('swarm_votes', {}),
                'market_state': inputs.get('market_state', {}),
                'trend_info': trend_info,
                'spread_ok': agi_context.get('spread_check', {}).get('is_tight', True) if agi_context else True
        }
        
        if agi_context:
            reflection_context.update(agi_context) # Inject Session, Liquidity, Nuance
        
        reflection_result = self.metacognition.reflect(
            decision={
                'direction': internal_signal.signal_type,
                'confidence': internal_signal.confidence,
                'factors': internal_signal.meta_data.get('factors', [])
            },
            context=reflection_context
        )
        
        # --- PHASE 9: SINGULARITY VETO & INVERSION ---
        final_signal = internal_signal
        
        # 1. Counter-Trend Check (Ontological Paradox)
        trend_score = trend_info.get('composite_score', 0.0)
        
        # Determine if we have a strong macro trend conflict
        macro_conflict = False
        signal_val = 1 if internal_signal.signal_type == "BUY" else -1 if internal_signal.signal_type == "SELL" else 0
        
        # Stricter Threshold: 0.3 (was 0.4)
        if abs(trend_score) > 0.3:
            if (trend_score > 0 and signal_val < 0) or (trend_score < 0 and signal_val > 0):
                macro_conflict = True

        if "Counter-Trend Violation" in str(reflection_result.blind_spots_detected) or macro_conflict:
             # --- NEOGENESIS V2: ELASTIC SNAP LOGIC ---
             # Instead of Vetoing, we check if the Counter-Trend is a valid "Rubber Band" Reversion.
             
             allow_scalp = False
             inversion_candidate = False
             reason = "Safety"
             
             # 1. Check Cycle (Reversion Zones allow Scalps)
             current_minute = int((time.time() / 60) % 60)
             q_cycle = (current_minute // 15) + 1
             
             if q_cycle in [2, 4]: # Q2 (Shift) or Q4 (Fix)
                  allow_scalp = True
                  reason = f"Cycle Q{q_cycle} (Reversion Zone)"
                  
             # 2. Check Entropy (Chaos allows Scalps)
             # Try to get entropy from context, default to 0.5
             entropy = agi_context.get('market_entropy', 0.5) if agi_context else 0.5
             if entropy > 0.85:
                  allow_scalp = True
                  reason = "High Entropy (Chaos)"
                  
             # 3. Extreme Extension (Rubber Band)
             if abs(trend_score) > 0.92:
                  allow_scalp = True
                  reason = "Extreme Extension (Snap Back)"
             
             if allow_scalp:
                  # REDUCE RISK but ALLOW TRADE
                  logger.info(f"NEOGENESIS: Overriding Counter-Trend Veto. {internal_signal.signal_type} Allowed. Reason: {reason}")
                  final_signal = internal_signal._replace(confidence=min(internal_signal.confidence, 75.0))
                  if not internal_signal.meta_data: internal_signal = internal_signal._replace(meta_data={})
                  final_signal.meta_data['method'] = "ELASTIC_SCALP"
                  
             else:
                  # If we cannot scalp (Trend is strong and time is Q1/Q3), we INVERT (Aikido).
                  # "Buying the Dip"
                  if internal_signal.signal_type == "SELL" and trend_score > 0:
                       logger.info(f"NEOGENESIS: Trend Aikido (Inversion). SELL -> BUY. Reason: Strong Trend + Q{q_cycle}")
                       final_signal = internal_signal._replace(signal_type="BUY", confidence=0.85) # High confidence trend follow
                       if not internal_signal.meta_data: internal_signal = internal_signal._replace(meta_data={})
                       final_signal.meta_data['method'] = "TREND_AIKIDO_V2"
                  
                  elif internal_signal.signal_type == "BUY" and trend_score < 0:
                       logger.info(f"NEOGENESIS: Trend Aikido (Inversion). BUY -> SELL. Reason: Strong Bear + Q{q_cycle}")
                       final_signal = internal_signal._replace(signal_type="SELL", confidence=0.85)
                       final_signal.meta_data['method'] = "TREND_AIKIDO_V2"
                  
                  else:
                       # Fallback Safety
                       logger.warning(f"NEOGENESIS: Vetoed Counter-Trend {internal_signal.signal_type} - Trend Score: {trend_score:.2f}. Safety First.")
                       final_signal = internal_signal._replace(signal_type="WAIT", confidence=0.0)
        
        # 2. Low Quality Veto
        elif reflection_result.reasoning_quality < 0.5:
             logger.warning(f"NEOGENESIS: Vetoed {internal_signal.signal_type} - Poor Reasoning Quality")
             final_signal = internal_signal._replace(signal_type="WAIT", confidence=0.0)
        
        # 3. GLOBAL CORRELATION VETO & SYMPATHY (The Web)
        # Improved Logic: Uses direct correlation matrix + open positions
        elif open_positions:
             sym = market_data_map.get('symbol', 'UNKNOWN')
             
             # A. Direct Conflict Check
             conflict, reason = self.correlation_synapse.check_correlation_conflict(sym, final_signal.signal_type, open_positions)
             if conflict:
                  logger.warning(f"CORRELATION GUARD: Vetoing {final_signal.signal_type} -> {reason}")
                  final_signal = final_signal._replace(signal_type="WAIT", confidence=0.0)
                  
             # B. Sympathy Play (Boost)
             elif final_signal.signal_type != "WAIT":
                  symp, reason = self.correlation_synapse.scan_sympathy_opportunities(sym, open_positions)
                  if symp:
                       logger.info(f"SYMPATHY PLAY: Boosting {final_signal.signal_type} -> {reason}")
                       final_signal = final_signal._replace(confidence=min(1.0, final_signal.confidence * 1.25))
                       if not final_signal.meta_data: final_signal = final_signal._replace(meta_data={})
                       final_signal.meta_data['sympathy'] = True

        # Fallback: Macro Risk Check (If no specific positions to conflict with, or just general safety)
        elif agi_context and 'global_risk' in agi_context:
            risk_data = agi_context['global_risk']
            sentiment = risk_data.get('global_risk_sentiment', 'NEUTRAL')
            
            sym = agi_context.get('symbol', 'XAUUSD')
            
            # --- PHASE 17: RANGE EXCEPTION ---
            # If we are in "Ping Pong Mode" (Ranging), we ignore Trend Vetoes
            # But we still respect GLOBAL RISK if it's extreme.
            range_info = agi_context.get('range_analysis', {})
            is_ranging = range_info.get('status') == 'RANGING'
            range_bias = agi_context.get('bias_override')
            
            if is_ranging and range_bias:
                 # Logic: We are Ranging. We WANT to trade Reversals.
                 # If Swarm says SELL and Range says SELL (Top of Range) -> APPROVE (Ping Pong)
                 # We bypass typical "Counter-Trend" blocks here because Range Trading IS Counter-Trend locally.
                 
                 if final_signal.signal_type == "BUY" and range_bias == "BUY":
                      logger.info("SINGULARITY: Approving PING-PONG BUY (Range Low Support).")
                      final_signal = final_signal._replace(signal_type="BUY", confidence=0.85)
                      return self._format_output(final_signal, is_dict)
                      
                 if final_signal.signal_type == "SELL" and range_bias == "SELL":
                      logger.info("SINGULARITY: Approving PING-PONG SELL (Range High Resistance).")
                      final_signal = final_signal._replace(signal_type="SELL", confidence=0.85)
                      return self._format_output(final_signal, is_dict)

            # Standard Logic (Trend Followers)
            if final_signal.signal_type == "BUY":
                if sentiment == "RISK_OFF" and sym in ["AUDUSD", "NZDUSD", "BTCUSD", "ETHUSD", "SPX500"]:
                     logger.warning(f"NEOGENESIS: Vetoed BUY on {sym} - Sentiment is {sentiment} (Risk-Off). Protection Active.")
                     final_signal = final_signal._replace(signal_type="WAIT", confidence=0.0)

            elif final_signal.signal_type == "SELL":
                # PROTECTION: Don't SELL USD Pairs if DXY is Strong (Risk Off usually means DXY UP)
                # If Sentiment is RISK_OFF -> DXY is Rising -> USDXXX goes UP.
                # So Selling USDCHF/USDJPY is suicide.
                if sentiment == "RISK_OFF" and sym in ["USDCHF", "USDJPY", "USDCAD"]:
                     logger.warning(f"NEOGENESIS: Vetoed SELL on {sym} - DXY is Strong (Risk-Off). Protection Active.")
                     final_signal = final_signal._replace(signal_type="WAIT", confidence=0.0)
                     
        # 4. Boost Confidence
        elif reflection_result.reasoning_quality > 0.8:
             final_signal = final_signal._replace(confidence=min(1.0, final_signal.confidence * 1.1))

        if reflection_result.total_depth > 0:
             logger.info(f"AGI REFLECTION: {reflection_result.synthesis} (Quality: {reflection_result.reasoning_quality:.2f})")

        return self._format_output(final_signal, is_dict)

    def _format_output(self, signal_obj: Any, return_dict: bool) -> Any:
        if return_dict:
            return {
                'verdict': signal_obj.signal_type,
                'confidence': signal_obj.confidence,
                'score': 0.0
            }
        return signal_obj
        # Assuming we can get a quick sentiment from Causal (simplification)
        
        # 2. Temporal Input
        if hasattr(self, 'temporal'):
             pass

        # 3. History
        if self.learning:
             learning_stats = self.learning.analyze_patterns()
             # Placeholder for atr, entropy, trend_strength - these would come from market_data_map or other analysis
             atr = 0.0 # Example value
             entropy = 0.0 # Example value
             
             perf_metrics = {
                 'win_rate_10': learning_stats.get('win_rate', 0.5),
                 'drawdown': self.monitor.metrics.max_drawdown # Assuming monitor has max_drawdown
             }
             
             # Phase 5: Regime Detection
             regime_data = self.regime_detector.detect_regime(
                 market_metrics={'atr': atr, 'entropy': entropy, 'trend_strength': 50.0}, # Trend TBD
                 performance_metrics=perf_metrics
             )
             adjustments.update(regime_data) # Inject 'regime' and 'threshold_modifier'
             
             if regime_data['regime'] == "CRITICAL" and config.get('mode') != 'CONSERVATIVE': # Use config from pre_tick args
                  logger.critical("REGIME: Switching to CONSERVATIVE due to Critical State.")
                  adjustments['switch_mode'] = 'CONSERVATIVE'
             if learning_stats['win_rate'] > 0.6:
                 inputs['history_bias'] = {'direction': 1, 'confidence': 0.8}
             elif learning_stats['win_rate'] < 0.4:
                 inputs['history_bias'] = {'direction': -1, 'confidence': 0.8}

        decision_packet = self.synergy.synthesize_singularity_vector(inputs)
        
        # --- PHASE 9: LOGIC GATE ---
        # "Logic is the beginning of wisdom, not the end." - Spock
        # But we must start with Logic.
        # We need a context for logic (Mocking for now, ideally passed in)
        logic_context = {'spread': 0.1, 'volatility': 1.0, 'trend_score': 0} # Placeholder
        
        # In real implementation, we'd grab these from 'market_data_map' or 'tick' if available in this scope.
        # For now, we trust the swarm unless Logic is explicitly called elsewhere.
        # Let's just return the decision.
        
        return decision_packet

    def resolve_conflict(self, buyers: float, sellers: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wraps the SwarmConflictResolver.
        Called by SwarmOrchestrator when Civil War is detected.
        """
        return self.conflict_resolver.resolve_civil_war(buyers, sellers, context)
    
    def post_tick(self, decision: str, result: Dict):
        """Post-tick AGI learning."""
        self.recent_decisions.append(decision)
        
        # --- AGI CAUSAL REFLECTION ---
        # The Infinite Why: Reason about the decision
        try:
            event = self.why_engine.capture_event(
                symbol="XAUUSD", # TODO: Pass dynamically
                timeframe="M1", # TODO: Pass dynamically
                market_state=result.get('market_state', {}),
                analysis_state={},
                decision=decision,
                decision_score=0.0,
                decision_meta=result,
                module_name="OmegaCore"
            )
            
            # Deep Scan on interesting events (Trades or unexpected Wait)
            if decision in ["BUY", "SELL"] or (decision == "WAIT" and self.meta_loop.iteration % 100 == 0):
                reflection = self.why_engine.deep_scan_recursive(
                    module_name="OmegaCore",
                    query_event=event,
                    max_depth=5 # Fast scan
                )
                logger.debug(f"InfiniteWhy Reflection: {len(reflection.get('why_root', {}).get('children', []))} branches analyzed")
                
        except Exception as e:
            logger.warning(f"InfiniteWhy Failure: {e}")

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
