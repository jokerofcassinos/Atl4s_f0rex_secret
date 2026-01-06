
import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from .consciousness_bus import ConsciousnessBus
from .interfaces import SwarmSignal, SubconsciousUnit
from core.memory.holographic import HolographicMemory  # Phase 117
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

# --- SWARM MODULE IMPORTS ---
# Core / Legacy
from analysis.swarm.trending_swarm import TrendingSwarm
from analysis.swarm.sniper_swarm import SniperSwarm
from analysis.swarm.quant_swarm import QuantSwarm
from analysis.swarm.veto_swarm import VetoSwarm
from analysis.swarm.whale_swarm import WhaleSwarm
from analysis.swarm.quantum_grid_swarm import QuantumGridSwarm
from analysis.swarm.game_swarm import GameSwarm
from analysis.swarm.chaos_swarm import ChaosSwarm
from analysis.swarm.macro_swarm import MacroSwarm
from analysis.swarm.fractal_vision_swarm import FractalVisionSwarm

# Phase 10-18 Engines
from analysis.swarm.oracle_swarm import OracleSwarm
from analysis.swarm.reservoir_swarm import ReservoirSwarm
from analysis.swarm.sentiment_swarm import SentimentSwarm
from analysis.swarm.order_flow_swarm import OrderFlowSwarm
from analysis.swarm.liquidity_map_swarm import LiquidityMapSwarm
from analysis.swarm.causal_graph_swarm import CausalGraphSwarm
from analysis.swarm.counterfactual_engine import CounterfactualEngine
from analysis.swarm.spectral_swarm import SpectralSwarm
from analysis.swarm.wavelet_swarm import WaveletSwarm
from analysis.swarm.topological_swarm import TopologicalSwarm
from analysis.swarm.manifold_swarm import ManifoldSwarm
from analysis.swarm.associative_swarm import AssociativeSwarm
from analysis.swarm.path_integral_swarm import PathIntegralSwarm
from analysis.swarm.hybrid_scalper_swarm import HybridScalperSwarm
from analysis.swarm.architect_swarm import ArchitectSwarm
from analysis.swarm.time_knife_swarm import TimeKnifeSwarm
from analysis.swarm.harvester_swarm import HarvesterSwarm
from analysis.swarm.nexus_swarm import NexusSwarm
from analysis.swarm.apex_swarm import ApexSwarm
from analysis.swarm.weaver_swarm import WeaverSwarm
from analysis.swarm.laplace_swarm import LaplaceSwarm
from analysis.swarm.physarum_swarm import PhysarumSwarm
from analysis.swarm.singularity_swarm import SingularitySwarm
from analysis.swarm.neural_lace import NeuralLace
from analysis.swarm.causal_swarm import CausalSwarm
from analysis.swarm.news_swarm import NewsSwarm
from analysis.swarm.zero_point_swarm import ZeroPointSwarm
from analysis.swarm.active_inference_swarm import ActiveInferenceSwarm
from analysis.swarm.hawking_swarm import HawkingSwarm
# Phase 115
from analysis.swarm.riemann_swarm import RiemannSwarm
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm
from analysis.swarm.gravity_swarm import GravitySwarm
from analysis.swarm.holographic_swarm import HolographicSwarm
from analysis.swarm.strange_attractor_swarm import StrangeAttractorSwarm
from analysis.swarm.bayesian_swarm import BayesianSwarm
from analysis.swarm.akashic_swarm import AkashicSwarm
from analysis.swarm.gaussian_process_swarm import GaussianProcessSwarm
from analysis.swarm.thermodynamic_swarm import ThermodynamicSwarm
from analysis.swarm.hyperdimensional_swarm import HyperdimensionalSwarm
from analysis.swarm.mirror_swarm import MirrorSwarm
from analysis.swarm.interference_swarm import InterferenceSwarm
from analysis.swarm.kinematic_swarm import KinematicSwarm
from analysis.swarm.vortex_swarm import VortexSwarm
from analysis.swarm.dna_swarm import DNASwarm
from analysis.swarm.schrodinger_swarm import SchrodingerSwarm
from analysis.swarm.antimatter_swarm import AntimatterSwarm
from analysis.swarm.heisenberg_swarm import HeisenbergSwarm
from analysis.swarm.navier_stokes_swarm import NavierStokesSwarm
from analysis.swarm.dark_matter_swarm import DarkMatterSwarm
from analysis.swarm.superluminal_swarm import SuperluminalSwarm
from analysis.swarm.lorentz_swarm import LorentzSwarm
from analysis.swarm.minkowski_swarm import MinkowskiSwarm
from analysis.swarm.higgs_swarm import HiggsSwarm
from analysis.swarm.boltzmann_swarm import BoltzmannSwarm
from analysis.swarm.fermi_swarm import FermiSwarm
from analysis.swarm.bose_einstein_swarm import BoseEinsteinSwarm
from analysis.swarm.schrodinger_newton_swarm import SchrodingerNewtonSwarm
from analysis.swarm.tachyon_swarm import TachyonSwarm
from analysis.swarm.feynman_swarm import FeynmanSwarm
from analysis.swarm.maxwell_swarm import MaxwellSwarm
from analysis.swarm.penrose_swarm import PenroseSwarm
from analysis.swarm.godel_swarm import GodelSwarm
from analysis.swarm.cinematics_swarm import CinematicsSwarm
from analysis.swarm.attention_swarm import AttentionSwarm
from analysis.swarm.unified_field_swarm import UnifiedFieldSwarm
from analysis.swarm.black_swan_swarm import BlackSwanSwarm
from analysis.swarm.council_swarm import CouncilSwarm
from analysis.swarm.overlord_swarm import OverlordSwarm
from analysis.swarm.sovereign_swarm import SovereignSwarm
from .neuroplasticity import NeuroPlasticityEngine
from .mcts_planner import MCTSPlanner
from .hyper_dimensional import HyperDimensionalEngine
from .transformer_lite import TransformerLite
from .genetics import EvolutionEngine

logger = logging.getLogger("SwarmOrchestrator")

class SwarmOrchestrator:
    def __init__(self, bus: ConsciousnessBus, evolution: EvolutionEngine, neuroplasticity: NeuroPlasticityEngine, attention: TransformerLite, grandmaster: MCTSPlanner):
        self.bus = bus
        self.evolution = evolution
        self.neuroplasticity = neuroplasticity
        self.attention = attention
        self.grandmaster = MCTSPlanner()
        
        self.active_agents = []
        self.alpha_threshold = 60.0 # Default
        self.short_term_memory = {'market_state': {}}
        self.synaptic_buffer = []

    async def initialize_swarm(self):
        logger.info("--- GENESIS PROTOCOL: AWAKENING SWARM ---")
        self.active_agents = []
        
        # 1. Safety & Core
        self.active_agents.append(VetoSwarm()) 
        self.active_agents.append(TrendingSwarm())
        self.active_agents.append(SniperSwarm())
        self.active_agents.append(QuantSwarm())
        self.active_agents.append(WhaleSwarm())
        self.active_agents.append(QuantumGridSwarm())
        
        # 2. Hyper-Cognition
        self.active_agents.append(GameSwarm())
        self.active_agents.append(ChaosSwarm())
        self.active_agents.append(MacroSwarm())
        self.active_agents.append(FractalVisionSwarm())
        
        # 3. Projections & Intuition
        self.active_agents.append(OracleSwarm())
        self.active_agents.append(ReservoirSwarm())
        self.active_agents.append(SentimentSwarm())
        
        # 4. Institutional Matrix
        self.active_agents.append(OrderFlowSwarm())
        self.active_agents.append(LiquidityMapSwarm())
        
        # 5. Causal Nexus
        self.active_agents.append(CausalGraphSwarm())
        self.active_agents.append(CounterfactualEngine())
        
        # 6. Spectral Engine
        self.active_agents.append(SpectralSwarm())
        self.active_agents.append(WaveletSwarm())
        
        # 7. Geometric Mind
        self.active_agents.append(TopologicalSwarm())
        self.active_agents.append(ManifoldSwarm())
        
        # 8. Omniscient Lattice & Feynman Machine
        self.active_agents.append(AssociativeSwarm())
        self.active_agents.append(PathIntegralSwarm())
        self.active_agents.append(HybridScalperSwarm())
        self.active_agents.append(ArchitectSwarm())
        self.active_agents.append(TimeKnifeSwarm())
        self.active_agents.append(HarvesterSwarm())
        self.active_agents.append(NexusSwarm())
        self.active_agents.append(ApexSwarm())
        self.active_agents.append(EventHorizonSwarm())
        self.active_agents.append(GravitySwarm())
        
        self.active_agents.append(WeaverSwarm(self.bus))
        self.active_agents.append(LaplaceSwarm())
        self.active_agents.append(PhysarumSwarm())
        self.active_agents.append(SingularitySwarm())
        self.active_agents.append(NeuralLace(self.bus))
        self.active_agents.append(CausalSwarm())

        # Phase 50-90: Physics & Exotic Agents (Restored)
        self.active_agents.append(StrangeAttractorSwarm())
        self.active_agents.append(BayesianSwarm())
        self.active_agents.append(AkashicSwarm())
        self.active_agents.append(GaussianProcessSwarm())
        self.active_agents.append(ThermodynamicSwarm())
        self.active_agents.append(HyperdimensionalSwarm())
        self.active_agents.append(MirrorSwarm())
        self.active_agents.append(InterferenceSwarm())
        self.active_agents.append(KinematicSwarm())
        self.active_agents.append(VortexSwarm())
        self.active_agents.append(DNASwarm())
        self.active_agents.append(SchrodingerSwarm())
        self.active_agents.append(AntimatterSwarm())
        self.active_agents.append(HeisenbergSwarm())
        self.active_agents.append(NavierStokesSwarm())
        self.active_agents.append(DarkMatterSwarm())
        self.active_agents.append(SuperluminalSwarm())
        self.active_agents.append(LorentzSwarm())
        self.active_agents.append(MinkowskiSwarm())
        self.active_agents.append(HiggsSwarm())
        self.active_agents.append(BoltzmannSwarm())
        self.active_agents.append(FermiSwarm())
        self.active_agents.append(BoseEinsteinSwarm())
        self.active_agents.append(SchrodingerNewtonSwarm())
        self.active_agents.append(TachyonSwarm())
        self.active_agents.append(FeynmanSwarm())
        self.active_agents.append(MaxwellSwarm())
        self.active_agents.append(HolographicSwarm()) # The Original Entropy Swarm
        self.active_agents.append(PenroseSwarm())
        self.active_agents.append(GodelSwarm())
        self.active_agents.append(CinematicsSwarm())
        self.active_agents.append(AttentionSwarm())
        self.active_agents.append(UnifiedFieldSwarm())
        self.active_agents.append(BlackSwanSwarm())
        
        self.active_agents.append(CouncilSwarm())
        self.active_agents.append(OverlordSwarm())
        self.active_agents.append(SovereignSwarm())
        
        # Phase 111-115
        self.active_agents.append(NewsSwarm())
        self.active_agents.append(ZeroPointSwarm())
        self.active_agents.append(ActiveInferenceSwarm())
        self.active_agents.append(HawkingSwarm())
        self.active_agents.append(HawkingSwarm())
        self.active_agents.append(RiemannSwarm())
        
        # Phase 118: Time Crystals
        from analysis.swarm.chronos_swarm import ChronosSwarm
        self.active_agents.append(ChronosSwarm())
        
        # Phase 119: Manifold Engine
        from analysis.swarm.ricci_swarm import RicciSwarm
        self.active_agents.append(RicciSwarm())
        
        # Phase 120: Technical Omniscience
        from analysis.swarm.technical_swarm import TechnicalSwarm
        self.active_agents.append(TechnicalSwarm())
        
        # Phase 122: Smart Money (SMC) & Visuals
        from analysis.swarm.smc_swarm import SmartMoneySwarm
        self.active_agents.append(SmartMoneySwarm())

    def inject_bridge(self, bridge):
        """
        Injects the ZmqBridge into capable agents for drawing.
        """
        for agent in self.active_agents:
            if hasattr(agent, 'set_bridge'):
                agent.set_bridge(bridge)
        
        # Phase 117: Holographic Associative Memory
        self.holographic_memory = HolographicMemory()
        self.last_consensus_state = {}
        self.last_price = 0.0

        logger.info(f"Swarm Initialized with {len(self.active_agents)} Cognitive Sub-Units.")
        self.state = "CONSCIOUS"

    async def process_tick(self, tick: Dict[str, Any], data_map: Dict[str, pd.DataFrame], config: Dict = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        The Main Loop of Consciousness.
        """
        # --- PHASE 117: HOLOGRAPHIC LEARNING ---
        current_price = tick.get('bid', 0)
        
        if self.last_price > 0 and self.last_consensus_state:
            delta = current_price - self.last_price
            outcome_score = np.tanh(delta * 0.5) 
            self.holographic_memory.store_experience(self.last_consensus_state, outcome_score)
            
        # --- PHASE 125: NEUROPLASTICITY (Self-Correction) ---
        # Learning Trigger: New Candle (or every N ticks)
        # We need to access the LAST candle's close from df_m5 if available
        # or just use the rolling price delta in valid ticks.
        
        # Simplified Learning: Every 5 minutes (approx 300 seconds), check result of LAST vote.
        now_time = tick.get('time_msc', time.time()*1000)
        if not hasattr(self, 'last_learning_time'): self.last_learning_time = 0
        if not hasattr(self, 'last_vote_snapshot'): self.last_vote_snapshot = {}
        if not hasattr(self, 'last_price_snapshot'): self.last_price_snapshot = current_price
        
        if now_time - self.last_learning_time > 300000: # 5 Minutes
            price_delta = current_price - self.last_price_snapshot
            
            if self.last_vote_snapshot:
                 logger.info(f"NEUROPLASTICITY: Analyzing Past 5m. Delta: {price_delta:.2f}. Adjusting Synapses...")
                 self.neuroplasticity.register_outcome(self.last_vote_snapshot, price_delta)
                 
                 # Save Memories (Dreams) to Disk
                 self.holographic_memory.save_memory()
                 
            self.last_learning_time = now_time
            self.last_price_snapshot = current_price
            
        self.last_price = current_price

        # Context for Swarms
        df_m5 = data_map.get('M5')
        if isinstance(df_m5, pd.DataFrame) and not df_m5.empty:
             context = {
                'tick': tick,
                'data_map': data_map,
                'df_m5': df_m5,
                'market_state': {},
                'config': config if config else {}
             }
        else:
             context = {'tick': tick} # Fallback
             
        # Collect Signals 
        thoughts: List[SwarmSignal] = []
        swarm_snapshot: Dict[str, float] = {}  # For Memory

        symbol = tick.get('symbol', 'UNKNOWN')
        timeframe = config.get('timeframe', 'M5') if config else 'M5'
        market_state_base: Dict[str, Any] = {
            "price": float(current_price),
        }

        for agent in self.active_agents:
            signal = await agent.process(context)
            if signal:
                # Phase 9: Global AGI hook for ALL swarms (no file left behind)
                try:
                    adapter = AGISwarmAdapter(agent.name)
                    swarm_output = {
                        "decision": signal.signal_type,
                        "score": signal.confidence,
                        "reason": signal.meta_data.get("reason", ""),
                        "aggregated_signal": signal.signal_type,
                    }
                    # Merge generic state with per-signal meta_data for richer memory
                    market_state = {
                        **market_state_base,
                        **signal.meta_data,
                    }
                    thought: SwarmThoughtResult = adapter.think_on_swarm_output(
                        symbol=symbol,
                        timeframe=timeframe,
                        market_state=market_state,
                        swarm_output=swarm_output,
                    )
                    # Attach AGI introspection ids into meta_data for downstream modules
                    signal.meta_data.setdefault("agi_thought_root_id", thought.thought_root_id)
                    signal.meta_data.setdefault("agi_scenarios", thought.meta.get("scenario_count", 0))
                except Exception as e:
                    logger.error("Global AGI Swarm Hook error for %s: %s", agent.name, e)

                thoughts.append(signal)
                self.bus.register_thought(signal)

                # Build Consensus State for Memory
                sign = 1.0 if signal.signal_type == "BUY" else (-1.0 if signal.signal_type == "SELL" else 0.0)
                swarm_snapshot[agent.name] = signal.confidence * sign

        self.last_consensus_state = swarm_snapshot

        # Synthesize without await (Sync Method)
        # Snapshot for Learning
        self.last_vote_snapshot = {}
        for t in thoughts:
             self.last_vote_snapshot[t.source] = t.signal_type
             
        decision, score, meta = self.synthesize_thoughts(thoughts, swarm_snapshot)
        return decision, score, meta

    def synthesize_thoughts(self, thoughts: List[SwarmSignal] = None, current_state_vector: Dict = None) -> Tuple[str, float, Dict]:
        if not thoughts:
            thoughts = self.bus.get_recent_thoughts()
        
        if not thoughts: return "WAIT", 0.0, {}
        
        # Log all thoughts for visibility (Restored Feature)
        vote_strings = []
        for t in thoughts:
             vote_strings.append(f"{t.source}={t.signal_type}({t.confidence:.0f}%)")
        logger.debug(f"SWARM VOTES: {', '.join(vote_strings)}")
        
        allowed_actions = ["BUY", "SELL", "WAIT", "EXIT_ALL", "EXIT_LONG", "EXIT_SHORT", "VETO"]
        macro_bias_reason = ""
        strong_macro = False
        macro_score = 0.0
        
        # Macro Check
        for t in thoughts:
            if t.source == "News_Swarm" and hasattr(t, 'meta_data'):
                macro_score = t.meta_data.get('bias', 0.0)
                if macro_score > 0.60: 
                    allowed_actions = ["BUY", "WAIT", "EXIT_SHORT", "EXIT_ALL"]
                    macro_bias_reason = f"Extreme Bullish News ({macro_score:.2f})"
                    strong_macro = True
                elif macro_score < -0.60:
                    allowed_actions = ["SELL", "WAIT", "EXIT_LONG", "EXIT_ALL"]
                    macro_bias_reason = f"Extreme Bearish News ({macro_score:.2f})"
                    strong_macro = True

        # 1. Physics Reality Check
        physics_decision = self._resolve_physics_conflict(thoughts)
        if physics_decision:
             action, conf, meta = physics_decision
             if conf > 98.0:
                 return (action, conf, meta)
             if conf > 85.0 and not strong_macro:
                   return (action, conf, meta)
             if action not in allowed_actions:
                  logger.warning(f"PHYSICS BLOCKED BY MACRO: {action} denied.")
                  return ("WAIT", 0.0, {})
             return (action, conf, meta)

        # 2. Holographic Recall
        if current_state_vector:
             intuition = self.holographic_memory.retrieve_intuition(current_state_vector)
             if intuition < -0.3:
                  logger.warning(f"DEJA VU: Holographic Danger (Score: {intuition:.2f}). Aborting.")
                  return ("WAIT", 0.0, {})
        
        # 0. Sovereign Executive Order (Meta-Cognition)
        weights = {}
        
        # A. Neuroplasticity Weights (Long Term Memory)
        learned_weights = self.neuroplasticity.get_dynamic_weights()
        
        # B. Sovereign Weights (Regime Adaptation)
        sov_weights = {}
        for t in thoughts:
             if t.source == "Sovereign_Swarm" and t.signal_type == "META_INFO":
                 sov_weights = t.meta_data.get('weight_vector', {})
                 logger.info(f"SOVEREIGN ACT: Regime={t.meta_data.get('regime')} | Weights Applied.")
        
        # C. Merge (Sovereign * Learned)
        # Allows Sovereign to Temporarily Override long-term learning if Regime changes drastically
        all_keys = set(learned_weights.keys()).union(set(sov_weights.keys()))
        for k in all_keys:
             w_learn = learned_weights.get(k, 1.0) # Default to base
             w_sov = sov_weights.get(k, 1.0)
             weights[k] = w_learn * w_sov

        # 3. Transformer Attention Consensus (AGI Brain)
        if 'allowed_actions' not in locals(): allowed_actions = ["BUY", "SELL", "WAIT", "EXIT_LONG", "EXIT_SHORT", "EXIT_ALL"]
        
        final_decision, final_score, meta_data = self._transformer_consensus(thoughts, weights, current_state_vector, allowed_actions)
        # Harvester Override (Priority)
        for t in thoughts:
             if t.signal_type == "EXIT_ALL": return ("EXIT_ALL", 99.0, t.meta_data)
             if t.signal_type == "VETO": return ("WAIT", 0.0, {})

        return (final_decision, final_score, meta_data)

        # --- LEGACY CODE BELOW (TO BE REMOVED) ---
        score_buy = 0
        score_sell = 0
        for t in thoughts:
            if t.signal_type == "META_INFO": continue # Skip Meta signals
            
            # Apply Sovereign Weight
            w = weights.get(t.source, 1.0)
            
            # Log significant mod
            if w != 1.0:
                 logger.debug(f"Weight Mod: {t.source} * {w}")
            
            if t.signal_type == "BUY": score_buy += t.confidence * w
            elif t.signal_type == "SELL": score_sell += t.confidence * w
            
        final_decision = "WAIT"
        final_score = 0
        
        # CIVIL WAR CHECK (Both sides strong)
        if score_buy > 2000 and score_sell > 2000:
             ratio = score_buy / score_sell if score_sell > 0 else 1.0
             if 0.9 < ratio < 1.1: # Relaxed from 0.8-1.2 to 0.9-1.1 to allow more trades
                 logger.warning(f"CIVIL WAR DETECTED: Buyers({score_buy:.0f}) vs Sellers({score_sell:.0f}). Gridlock.")
                 return "WAIT", 0.0, {}
                 
        if score_buy > score_sell and score_buy > 500: # Heuristic thresh
            final_decision = "BUY"
            final_score = 80
        elif score_sell > score_buy and score_sell > 500:
            final_decision = "SELL"
            final_score = 80
            
        if final_decision not in allowed_actions:
             final_decision = "WAIT"
             
        # Harvester Override
        for t in thoughts:
             if t.signal_type == "EXIT_ALL": return ("EXIT_ALL", 99.0, t.meta_data)
             if t.signal_type == "VETO": return ("WAIT", 0.0, {})

        return (final_decision, final_score, {})

    def _resolve_physics_conflict(self, thoughts):
        physics_agents = ['SingularitySwarm', 'HyperdimensionalSwarm', 'KinematicSwarm', 'VortexSwarm', 'Schrodinger_Newton_Swarm']
        high_command = [t for t in thoughts if t.source in physics_agents and t.confidence > 95.0]
        if high_command:
            top_signal = sorted(high_command, key=lambda x: x.confidence, reverse=True)[0]
            return (top_signal.signal_type, top_signal.confidence, top_signal.meta_data)
        return None

    def _transformer_consensus(self, thoughts, weights, current_state_vector, allowed_actions):
        """
        Phase 117: Transformer Attention Consensus (AGI Brain).
        Replaces simple voting with Self-Attention.
        """
        d_model = 64
        vectors = []
        
        # Vector 0: Holographic Memory Context (The "RAG" Token)
        rag_vec = np.zeros(d_model)
        if current_state_vector:
             intuition = self.holographic_memory.retrieve_intuition(current_state_vector)
             rag_vec[0] = intuition
             rag_vec[1] = 1.0 # Type: Memory
        vectors.append(rag_vec)
        
        # Vectors 1..N: Agent Signals
        for t in thoughts:
             if t.signal_type == "META_INFO": continue
             w = weights.get(t.source, 1.0)
             val = 0
             if t.signal_type == "BUY": val = 1
             elif t.signal_type == "SELL": val = -1
             
             vec = np.zeros(d_model)
             vec[0] = val * t.confidence / 100.0 * w
             vec[1] = 0.5 # Type: Agent
             vec[2] = w
             h = hash(t.source) % 100
             vec[3 + (h % 10)] = 1.0 
             vectors.append(vec)
             
        if not vectors: return "WAIT", 0.0, {}
        
        x = np.array(vectors)
        context_matrix, attn_weights = self.attention.forward(x)
        swarm_vec = np.mean(context_matrix[1:], axis=0) if len(context_matrix) > 1 else context_matrix[0]
        
        decision_score = swarm_vec[0]
        final_decision = "WAIT"
        final_conf = abs(decision_score) * 100.0
        
        if decision_score > 0.05: final_decision = "BUY"
        elif decision_score < -0.05: final_decision = "SELL"
             
        final_conf = min(100.0, final_conf * 5.0) 
        
        logger.info(f"AGI ATTENTION: Decision={final_decision} (Score={decision_score:.3f}) | Memory Bias={vectors[0][0]:.3f}")

        if final_decision not in allowed_actions: final_decision = "WAIT"

        return final_decision, final_conf, {'attention_weights': attn_weights.tolist()}

