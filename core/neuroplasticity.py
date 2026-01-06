"""
AGI Ultra: Neuroplasticity Engine - Adaptive Weight Learning

Features:
- Real-time weight adjustment based on outcomes
- Context-aware learning (per market regime)
- Synaptic pruning for underperforming connections
- Hierarchical plasticity (different learning rates per layer)
- Hebbian learning: "Neurons that fire together, wire together"
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("Neuroplasticity")


@dataclass
class SynapticConnection:
    """A connection between two neural components."""
    source: str
    target: str
    weight: float
    base_weight: float
    learning_rate: float = 0.1
    last_updated: float = field(default_factory=time.time)
    activation_count: int = 0
    success_count: int = 0
    
    def get_efficacy(self) -> float:
        """Get connection efficacy (success rate)."""
        if self.activation_count == 0:
            return 0.5
        return self.success_count / self.activation_count


@dataclass
class RegimeContext:
    """Learning context for a specific market regime."""
    regime_name: str
    weights: Dict[str, float]
    performance: Dict[str, float]
    sample_count: int = 0
    last_active: float = field(default_factory=time.time)


class NeuroPlasticityEngine:
    """
    AGI Ultra: Advanced Neuroplasticity Engine.
    
    Features:
    - Real-time weight adaptation
    - Context-aware learning (per market regime)
    - Synaptic pruning for underperformers
    - Hierarchical learning rates
    - Hebbian co-activation strengthening
    """
    
    # Default base weights for all swarms/agents
    DEFAULT_WEIGHTS = {
        # Core Analysis Swarms
        'Trending_Swarm': 1.0,
        'Sniper_Swarm': 1.0,
        'Quant_Swarm': 1.0,
        'Kinematic_Swarm': 2.0,
        'Veto_Swarm': 2.0,
        'Whale_Swarm': 1.5,
        'Quantum_Grid_Swarm': 1.2,
        'Game_Swarm': 1.0,
        'Chaos_Swarm': 0.8,
        'Macro_Swarm': 1.0,
        'Fractal_Vision_Swarm': 1.0,
        'Oracle_Swarm': 1.8,
        'Reservoir_Swarm': 1.2,
        'Sentiment_Swarm': 1.3,
        'Order_Flow_Swarm': 1.5,
        'Liquidity_Map_Swarm': 1.6,
        'Causal_Graph_Swarm': 1.4,
        'Counterfactual_Engine': 1.2,
        'Spectral_Swarm': 1.3,
        'Wavelet_Swarm': 1.5,
        'Topological_Swarm': 1.4,
        'Manifold_Swarm': 1.4,
        'Associative_Swarm': 1.3,
        'Path_Integral_Swarm': 1.6,
        # Ultra Advanced Swarms
        'Singularity_Swarm': 2.5,
        'Hyperdimensional_Swarm': 2.2,
        'Vortex_Swarm': 2.1,
        'DNA_Swarm': 2.0,
        'Schrodinger_Swarm': 1.9,
        'Antimatter_Swarm': 1.5,
        'Heisenberg_Swarm': 2.6,
        'Navier_Stokes_Swarm': 1.6,
        'Dark_Matter_Swarm': 1.8,
        'Holographic_Swarm': 1.7,
        'Superluminal_Swarm': 2.1,
        'Event_Horizon_Swarm': 1.8,
        'Lorentz_Swarm': 2.2,
        'Minkowski_Swarm': 1.9,
        'Higgs_Swarm': 2.0,
        'Boltzmann_Swarm': 1.8,
        'Fermi_Swarm': 2.1,
        'Bose_Einstein_Swarm': 2.3,
        'Schrodinger_Newton_Swarm': 1.9,
        'Tachyon_Swarm': 2.5,
        'Feynman_Swarm': 2.4,
        'Maxwell_Swarm': 2.2,
        'Riemann_Swarm': 2.7,
        'Penrose_Swarm': 2.8,
        'Godel_Swarm': 3.0,
        # AGI Modules
        'InfiniteWhyEngine': 2.5,
        'MetaLearning': 2.0,
        'UnifiedReasoning': 2.5,
        'HolographicMemory': 2.0,
    }
    
    def __init__(
        self,
        base_learning_rate: float = 0.1,
        pruning_threshold: float = 0.3,
        history_window: int = 100,
        regime_memory_size: int = 10
    ):
        self.base_learning_rate = base_learning_rate
        self.pruning_threshold = pruning_threshold
        self.history_window = history_window
        self.regime_memory_size = regime_memory_size
        
        # Initialize weights
        self.base_weights = self.DEFAULT_WEIGHTS.copy()
        self.dynamic_weights = self.DEFAULT_WEIGHTS.copy()
        
        # Hierarchical learning rates per layer
        self.layer_learning_rates = {
            'core': 0.05,      # Core swarms - learn slowly (stability)
            'physics': 0.08,   # Physics-based - medium learning
            'quantum': 0.1,    # Quantum - faster adaptation
            'meta': 0.15,      # Meta-reasoning - fastest learning
        }
        
        # Performance tracking
        self.performance: Dict[str, float] = {k: 0.5 for k in self.base_weights}
        self.history: Dict[str, List[int]] = {k: [] for k in self.base_weights}
        
        # Synaptic connections for Hebbian learning
        self.connections: Dict[Tuple[str, str], SynapticConnection] = {}
        
        # Context-aware regime learning
        self.regime_contexts: Dict[str, RegimeContext] = {}
        self.current_regime: str = "unknown"
        
        # Pruned swarms (disabled due to poor performance)
        self.pruned_swarms: Dict[str, float] = {}  # swarm -> time pruned
        
        # Statistics
        self.total_updates = 0
        self.successful_predictions = 0
        
        logger.info(f"NeuroPlasticityEngine initialized: lr={base_learning_rate}, prune={pruning_threshold}")
    
    # -------------------------------------------------------------------------
    # CONTEXT-AWARE LEARNING
    # -------------------------------------------------------------------------
    def set_regime(self, regime: str):
        """Set current market regime for context-aware learning."""
        self.current_regime = regime
        
        if regime not in self.regime_contexts:
            # Initialize new regime context
            self.regime_contexts[regime] = RegimeContext(
                regime_name=regime,
                weights=self.base_weights.copy(),
                performance={k: 0.5 for k in self.base_weights}
            )
        
        # Load regime-specific weights
        self.dynamic_weights = self.regime_contexts[regime].weights.copy()
        
        logger.debug(f"Switched to regime: {regime}")
    
    def _get_regime_context(self) -> RegimeContext:
        """Get or create current regime context."""
        if self.current_regime not in self.regime_contexts:
            self.set_regime(self.current_regime)
        return self.regime_contexts[self.current_regime]
    
    # -------------------------------------------------------------------------
    # OUTCOME REGISTRATION
    # -------------------------------------------------------------------------
    def register_outcome(
        self,
        signals: Dict[str, str],
        actual_move: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Register trade/prediction outcome and update weights.
        
        Args:
            signals: Agent signals {'Swarm_Name': 'BUY'/'SELL'/'WAIT'}
            actual_move: Actual price movement
            context: Additional context (regime, volatility, etc.)
        """
        self.total_updates += 1
        direction = 1 if actual_move > 0 else -1
        
        # Track co-activations for Hebbian learning
        co_activated = [(name, sig) for name, sig in signals.items() if sig in ['BUY', 'SELL']]
        
        for agent_name, signal in signals.items():
            if agent_name not in self.base_weights:
                continue
            if agent_name in self.pruned_swarms:
                continue
            
            # Calculate score
            score = 0
            if signal == "BUY":
                score = 1 if direction == 1 else 0
            elif signal == "SELL":
                score = 1 if direction == -1 else 0
            else:
                continue
            
            if score == 1:
                self.successful_predictions += 1
            
            # Update history
            self.history[agent_name].append(score)
            if len(self.history[agent_name]) > self.history_window:
                self.history[agent_name].pop(0)
            
            # Calculate performance
            if self.history[agent_name]:
                avg_acc = sum(self.history[agent_name]) / len(self.history[agent_name])
            else:
                avg_acc = 0.5
            
            self.performance[agent_name] = avg_acc
            
            # Get learning rate for this agent
            lr = self._get_learning_rate(agent_name)
            
            # Update dynamic weight
            self._update_weight(agent_name, avg_acc, lr)
            
            # Update regime context
            regime_ctx = self._get_regime_context()
            regime_ctx.performance[agent_name] = avg_acc
            regime_ctx.weights[agent_name] = self.dynamic_weights[agent_name]
            regime_ctx.sample_count += 1
            regime_ctx.last_active = time.time()
        
        # Hebbian learning: strengthen connections between co-successful agents
        if len(co_activated) >= 2:
            self._hebbian_update(co_activated, direction, actual_move)
        
        # Periodic pruning check
        if self.total_updates % 100 == 0:
            self._check_pruning()
    
    def _update_weight(self, agent_name: str, accuracy: float, learning_rate: float):
        """Update a single agent's weight."""
        base_w = self.base_weights.get(agent_name, 1.0)
        
        # Plasticity formula with momentum
        # Weight = Base * (Accuracy / 0.5)^2
        accuracy = max(0.1, min(0.9, accuracy))
        plasticity = pow(accuracy / 0.5, 2)
        
        # Smooth update with learning rate
        old_w = self.dynamic_weights.get(agent_name, base_w)
        new_w = base_w * plasticity
        
        # Exponential moving average
        self.dynamic_weights[agent_name] = old_w * (1 - learning_rate) + new_w * learning_rate
    
    def _get_learning_rate(self, agent_name: str) -> float:
        """Get hierarchical learning rate for an agent."""
        # Determine layer based on agent name
        name_lower = agent_name.lower()
        
        if any(x in name_lower for x in ['meta', 'infinite', 'unified', 'learning']):
            return self.layer_learning_rates['meta']
        elif any(x in name_lower for x in ['quantum', 'schrodinger', 'heisenberg', 'bose']):
            return self.layer_learning_rates['quantum']
        elif any(x in name_lower for x in ['navier', 'maxwell', 'lorentz', 'riemann', 'feynman']):
            return self.layer_learning_rates['physics']
        else:
            return self.layer_learning_rates['core']
    
    # -------------------------------------------------------------------------
    # HEBBIAN LEARNING
    # -------------------------------------------------------------------------
    def _hebbian_update(
        self,
        co_activated: List[Tuple[str, str]],
        direction: int,
        magnitude: float
    ):
        """
        Hebbian learning: strengthen connections between agents that fire together
        and succeed together.
        """
        for i in range(len(co_activated)):
            for j in range(i + 1, len(co_activated)):
                source, sig1 = co_activated[i]
                target, sig2 = co_activated[j]
                
                # Both predicted same direction
                if sig1 == sig2:
                    key = (source, target) if source < target else (target, source)
                    
                    if key not in self.connections:
                        self.connections[key] = SynapticConnection(
                            source=key[0],
                            target=key[1],
                            weight=1.0,
                            base_weight=1.0
                        )
                    
                    conn = self.connections[key]
                    conn.activation_count += 1
                    
                    # Did they both predict correctly?
                    predicted_buy = sig1 == 'BUY'
                    was_correct = (predicted_buy and direction > 0) or (not predicted_buy and direction < 0)
                    
                    if was_correct:
                        conn.success_count += 1
                        # Strengthen connection
                        conn.weight = min(3.0, conn.weight * 1.05)
                    else:
                        # Weaken connection
                        conn.weight = max(0.5, conn.weight * 0.95)
                    
                    conn.last_updated = time.time()
    
    def get_connection_bonus(self, agent1: str, agent2: str) -> float:
        """Get the Hebbian connection bonus between two agents."""
        key = (agent1, agent2) if agent1 < agent2 else (agent2, agent1)
        
        if key in self.connections:
            conn = self.connections[key]
            if conn.activation_count >= 10:
                return conn.weight * conn.get_efficacy()
        
        return 1.0
    
    # -------------------------------------------------------------------------
    # SYNAPTIC PRUNING
    # -------------------------------------------------------------------------
    def _check_pruning(self):
        """Check and prune consistently underperforming agents."""
        for agent_name, perf in self.performance.items():
            if agent_name in self.pruned_swarms:
                continue
            
            # Must have significant history
            if len(self.history.get(agent_name, [])) < 50:
                continue
            
            if perf < self.pruning_threshold:
                self.pruned_swarms[agent_name] = time.time()
                logger.warning(f"Pruned underperforming agent: {agent_name} (accuracy={perf:.1%})")
    
    def restore_pruned(self, agent_name: str):
        """Restore a pruned agent."""
        if agent_name in self.pruned_swarms:
            del self.pruned_swarms[agent_name]
            self.history[agent_name] = []
            self.performance[agent_name] = 0.5
            self.dynamic_weights[agent_name] = self.base_weights.get(agent_name, 1.0)
            logger.info(f"Restored pruned agent: {agent_name}")
    
    # -------------------------------------------------------------------------
    # WEIGHT ACCESS
    # -------------------------------------------------------------------------
    def get_dynamic_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """Get current optimized weights, optionally for a specific regime."""
        if regime and regime in self.regime_contexts:
            return self.regime_contexts[regime].weights.copy()
        return self.dynamic_weights.copy()
    
    def get_weight(self, agent_name: str) -> float:
        """Get weight for a specific agent."""
        if agent_name in self.pruned_swarms:
            return 0.0
        return self.dynamic_weights.get(agent_name, 1.0)
    
    def get_performance_ranking(self) -> List[Tuple[str, float]]:
        """Get agents ranked by performance."""
        rankings = [
            (name, perf) for name, perf in self.performance.items()
            if name not in self.pruned_swarms
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get neuroplasticity statistics."""
        active_agents = len(self.base_weights) - len(self.pruned_swarms)
        
        return {
            'total_updates': self.total_updates,
            'successful_predictions': self.successful_predictions,
            'success_rate': self.successful_predictions / self.total_updates if self.total_updates > 0 else 0,
            'active_agents': active_agents,
            'pruned_agents': len(self.pruned_swarms),
            'active_regimes': len(self.regime_contexts),
            'current_regime': self.current_regime,
            'hebbian_connections': len(self.connections),
            'top_performers': self.get_performance_ranking()[:5]
        }
