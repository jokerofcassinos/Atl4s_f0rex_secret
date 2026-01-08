
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
from analysis.swarm.fractal_swarm import FractalMicroClimax # Phase 146

from .agi.active_inference import GenerativeModel, FreeEnergyMinimizer
from .agi.metacognition import RecursiveReflectionLoop
from .agi.symbiosis.neural_resonance_bridge import NeuralResonanceBridge # Phase 150+

logger = logging.getLogger("SwarmOrchestrator")

from core.agi.omni_cortex import OmniCortex

class SwarmOrchestrator:
    def __init__(self, bus: ConsciousnessBus, evolution: EvolutionEngine, neuroplasticity: NeuroPlasticityEngine, attention: TransformerLite, grandmaster: MCTSPlanner = None, agi=None):
        self.bus = bus
        self.evolution = evolution
        self.neuroplasticity = neuroplasticity
        self.attention = attention
        self.grandmaster = grandmaster if grandmaster else MCTSPlanner() # Use passed instance
        self.agi = agi # Reference to OmegaAGICore (Brain)()
        self.omni_cortex = OmniCortex() # High-Level Reasoning Engine
        
        # --- AGI PHASE 4: HOLOGRAPHIC ACTIVE INFERENCE ---
        self.hd_engine = HyperDimensionalEngine()
        self.holographic_memory = HolographicMemory()
        self.generative_model = GenerativeModel()
        self.free_energy_minimizer = FreeEnergyMinimizer()
        
        # --- AGI PHASE 5: METACOGNITION ---
        self.metacognition = RecursiveReflectionLoop()
        self.resonance_bridge = NeuralResonanceBridge() # Phase 150+

        self.active_agents = []
        self.alpha_threshold = 60.0 # Default
        self.short_term_memory = {'market_state': {}}
        self.synaptic_buffer = []
        
        # AGI Hierarchical Clusters (Brain Regions)
        self.clusters = {
            'PHYSICS': ['Laplace_Swarm', 'Riemann_Swarm', 'Fermi_Swarm', 'Bose_Swarm', 'Thermodynamics_Swarm', 'Gravity_Swarm', 'Navier_Stokes_Swarm', 'Maxwell_Swarm'],
            'QUANTUM': ['Quantum_Grid', 'Schrodinger_Swarm', 'Heisenberg_Swarm', 'Antimatter_Swarm', 'Entanglement_Swarm', 'Zero_Point_Swarm', 'Tachyon_Swarm'],
            'PRICING': ['Trending_Swarm', 'Sniper_Swarm', 'Technical_Swarm', 'Fractal_Vision', 'Wavelet_Swarm', 'Spectral_Swarm'],
            'INSTITUTIONAL': ['Whale_Swarm', 'Order_Flow', 'Liquidity_Map', 'SMC_Swarm', 'News_Swarm', 'Apex_Swarm'],
            'META': ['Veto_Swarm', 'Red_Team_Swarm', 'Dream_Swarm', 'Reflection_Swarm', 'Zen_Swarm', 'Council_Swarm', 'Overlord_Swarm', 'Sovereign_Swarm']
        }

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

        # Phase 123: AGI Awakening (The Dreamer & The Mirror)
        from analysis.swarm.dream_swarm import DreamSwarm
        from analysis.swarm.reflection_swarm import ReflectionSwarm
        from analysis.swarm.zen_swarm import ZenSwarm
        self.active_agents.append(DreamSwarm())
        self.active_agents.append(ReflectionSwarm())
        self.active_agents.append(ZenSwarm())

        # Phase 130: The Adversary (Red Team / GAN)
        from analysis.swarm.red_team_swarm import RedTeamSwarm
        self.active_agents.append(RedTeamSwarm())

    def inject_bridge(self, bridge):
        """
        Injects the ZmqBridge into capable agents for drawing.
        """
        for agent in self.active_agents:
            if hasattr(agent, 'set_bridge'):
                agent.set_bridge(bridge)
        '                                                                               '
        # Phase 117: Holographic Associative Memory
        self.holographic_memory = HolographicMemory()
        self.last_consensus_state = {}
        self.last_price = 0.0

        logger.info(f"Swarm Initialized with {len(self.active_agents)} Cognitive Sub-Units.")
        self.state = "CONSCIOUS"

    async def process_tick(self, tick: Dict[str, Any], data_map: Dict[str, pd.DataFrame], config: Dict = None, agi_context: Dict = None) -> Tuple[str, float, Dict[str, Any]]:
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
                    
                    # --- CRITICAL FIX Check for List Corruption ---
                    # Some agents might return a list as metadata by mistake
                    if isinstance(signal.meta_data, list):
                        logger.error(f"DATA CORRUPTION: Agent {agent.name} returned LIST as metadata. Fixing...")
                        signal.meta_data = {'raw_data': signal.meta_data}
                    elif not isinstance(signal.meta_data, dict):
                         signal.meta_data = {'raw_val': str(signal.meta_data)}
                         
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

        # --- PHASE 2: AGI PERCEPTION & HYPER-COMPLEX ANALYSIS ---
        agi_adj = agi_context # Use provided context if available
        if not agi_adj and self.agi:
            # Fallback: Calculate locally if not provided
            agi_adj = self.agi.pre_tick(tick, config, data_map)
        
        if self.agi and agi_adj:
            # Merge AGI adjustments into swarm logic (e.g., premonition)
            thoughts.extend(self._process_agi_adjustments(agi_adj))
        
        # 2. Perceptual Processing (M1 Context)
        df_m1 = data_map.get('1m')
        if df_m1 is not None and not df_m1.empty:
            # ... existing logic ...
            pass

        # 3. Decision Synthesis
        # Pass agi_adj to synthesize_thoughts to enrich metadata
        decision, score, meta = self.synthesize_thoughts(thoughts, swarm_snapshot, 
                                                        mode=config.get('mode', 'SNIPER') if config else 'SNIPER',
                                                        agi_context=agi_adj,
                                                        config=config)
        return decision, score, meta

    def synthesize_thoughts(self, thoughts: List[SwarmSignal] = None, 
                         current_state_vector: Dict = None, 
                         mode: str = "SNIPER", 
                         agi_context: Dict = None,
                         config: Dict = None) -> Tuple[str, float, Dict]:
        if not thoughts:
            thoughts = self.bus.get_recent_thoughts()
        
        if not thoughts: return "WAIT", 0.0, {}
        
        # ===== PRIORITY OVERRIDES (Check FIRST, before any other logic) =====
        # 1. VETO signals have absolute authority
        for t in thoughts:
            if t.signal_type == "VETO":
                logger.warning(f"VETO OVERRIDE: {t.meta_data.get('reason', 'Unknown')} - Trade Blocked")
                return ("WAIT", 0.0, {"blocked_by": "VETO", "reason": t.meta_data.get('reason', '')})
        
        # 2. EXIT_ALL signals (emergency close)
        for t in thoughts:
            if t.signal_type == "EXIT_ALL":
                logger.critical(f"EMERGENCY EXIT: {t.meta_data.get('reason', 'Unknown')}")
                return ("EXIT_ALL", 99.0, t.meta_data)
        
        # Log all thoughts for visibility (Restored Feature)
        vote_strings = []
        for t in thoughts:
             vote_strings.append(f"{t.source}={t.signal_type}({t.confidence:.0f}%)")
        logger.info(f"SWARM VOTES: {', '.join(vote_strings)}")
        
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
                 regime = t.meta_data.get('regime', 'UNKNOWN')
                 logger.info(f"SOVEREIGN ACT: Regime={regime} | Weights Applied.")
        
        # C. Merge (Sovereign * Learned)
        # Allows Sovereign to Temporarily Override long-term learning if Regime changes drastically
        all_keys = set(learned_weights.keys()).union(set(sov_weights.keys()))
        for k in all_keys:
             w_learn = learned_weights.get(k, 1.0) # Default to base
             w_sov = sov_weights.get(k, 1.0)
             w_sov = sov_weights.get(k, 1.0)
             weights[k] = w_learn * w_sov
             
        # Decide based on weighted votes

        if 'market_data' in self.short_term_memory:
             self.omni_cortex.perceive(self.short_term_memory['market_data'])

        # D. Hierarchical Cluster Boosting (The Meta-Cortex)
        # If a brain region (Cluster) is highly active/confident, boost its members.
        cluster_scores = {}
        for cluster_name, agents in self.clusters.items():
            avg_conf = 0.0
            count = 0
            for t in thoughts:
                # Approximate matching (Agent name might be formatted)
                # t.source is "Sniper_Swarm", list has "Sniper_Swarm"
                if any(start in t.source for start in agents): 
                     avg_conf += t.confidence
                     count += 1
            if count > 0:
                cluster_scores[cluster_name] = avg_conf / count
        
        # Apply Boosts
        for cluster_name, score in cluster_scores.items():
            if score > 80.0: # High Consensus in Region
                boost = 1.25
                if score > 90.0: boost = 1.5
                # logger.debug(f"Cortex Region {cluster_name} Active ({score:.1f}%). Boosting Signals.")
                for agent_key in all_keys:
                     # Check if agent belongs to this cluster
                     if any(start in agent_key for start in self.clusters[cluster_name]):
                          weights[agent_key] *= boost

        # 3. Transformer Attention Consensus (AGI Brain)
        if 'allowed_actions' not in locals(): allowed_actions = ["BUY", "SELL", "WAIT", "EXIT_LONG", "EXIT_SHORT", "EXIT_ALL"]
        if 'regime' not in locals(): regime = "UNKNOWN"
        
        # Pass agi_context into transformer to be merged immediately
        final_decision, final_score, meta_data = self._transformer_consensus(thoughts, weights, current_state_vector, allowed_actions, mode=mode, regime=regime, agi_context=agi_context)
        
        # Inject Sovereign Signals into Metadata for Main Loop Multipliers
        sovereign_state = "NEUTRAL"
        for t in thoughts:
             if t.source == "Sovereign_Swarm":
                 # Check for Singularity
                 if t.meta_data.get('decision') == "SINGULARITY_REACHED":
                      sovereign_state = "SINGULARITY"
                 elif "STRONG" in t.meta_data.get('decision', ''):
                      sovereign_state = "STRONG"
        
        meta_data['sovereign_state'] = sovereign_state
        
        # --- PHASE 150+: HYPER-COMPLEX METADATA INJECTION ---
        if agi_context:
            for key, val in agi_context.items():
                if key not in meta_data:
                    meta_data[key] = val
            meta_data.setdefault('factors', []).append("AGI_HYPER_COMPLEX_INTEGRATION")

        # Harvester Override (Priority)
        for t in thoughts:
             if t.signal_type == "EXIT_ALL": return ("EXIT_ALL", 99.0, t.meta_data)
             if t.signal_type == "VETO": return ("WAIT", 0.0, {})

        # --- TREND VETO (Counter-Trend Block) ---
        # If Fractal/Trend Swarm signals strong trend, block opposing trades.
        trend_direction = None
        trend_strength = 0.0
        
        for t in thoughts:
             if t.source in ["Fractal_Vision_Swarm", "Trending_Swarm", "Navier_Stokes_Swarm"]:
                 if t.confidence > 75.0 and t.signal_type in ["BUY", "SELL"]:
                     if trend_direction is None:
                         trend_direction = t.signal_type
                         trend_strength = t.confidence
                     elif trend_direction == t.signal_type:
                         trend_strength = max(trend_strength, t.confidence)
                     else:
                         # Conflicting Trend Signals -> No clear trend veto
                         trend_direction = None
                         break
        
        if trend_direction and final_decision != "WAIT" and final_decision != trend_direction:
             # We are opposing a Strong Trend.
             # Only allow if Reversion Swarm is VERY confident (>95%)?
             # For now, SAFETY FIRST. Block it.
             logger.warning(f"TREND VETO: Blocked {final_decision} against Strong {trend_direction} Trend (Conf: {trend_strength:.1f})")
             final_decision = "WAIT"
             meta_data['veto_reason'] = f"Counter-Trend Protection ({trend_direction})"

        # 4. Active Inference Arbitration (The Free Energy Principal)
        # If the Swarm is split or uncertain, we check which decision minimizes Free Energy (Risk + Ambiguity)
        active_inf_decision = self._resolve_active_inference(thoughts, final_decision)
        if active_inf_decision:
             final_decision, final_score, meta_new = active_inf_decision
             meta_data.update(meta_new)
             logger.info(f"ACTIVE INFERENCE: Override -> {final_decision} (Minimizes Free Energy)")

        # 5. Metacognitive Reflection (Self-Correction)
        # "Before I act, let me think about why I am acting."
        reflection = self.metacognition.reflect(final_decision, final_score, meta_data, {'config': {'mode': mode}})
        final_score = reflection['adjusted_confidence']
        if reflection['notes']:
            meta_data['reflection_notes'] = reflection['notes']
            # If validation failed (score -> 0), switch to WAIT
            # Relaxed Threshold: 35.0 (was 50.0) to allow "Moderate" reasoning (0.7 quality * 0.6 conf = 42)
            if final_score < 35.0 and final_decision != "EXIT_ALL":
                final_decision = "WAIT"
                logger.info(f"METACOGNITION: Vetoed Decision. Score {final_score:.1f} < 35.0. Reason: {reflection['notes']}")
            else:
                logger.info(f"METACOGNITION: Passed. Score {final_score:.1f} (Threshold 35.0)")
        
        # 6. Neural Resonance Sync (Phase 150+)
        # Synchronize decision with user intent model (Symbiosis)
        if hasattr(self.agi, 'resonance_bridge'):
            u_intent = config.get('user_intent', {'target': 'PROFIT', 'risk': 'MODERATE'}) if config else {'target': 'PROFIT', 'risk': 'MODERATE'}
            resonance = self.agi.resonance_bridge.synchronize(u_intent, {'decision': final_decision, 'score': final_score})
            meta_data['symbiosis_sync'] = resonance['symbiosis_confidence']
            if resonance['fusion_stream'] == "SYMBIOTIC_CONSENSUS":
                logger.info(f"NEURAL RESONANCE: Absolute Symbiosis achieved ({resonance['symbiosis_confidence']:.2%})")

        return (final_decision, final_score, meta_data)

    def _resolve_active_inference(self, thoughts: List[SwarmSignal], current_consensus: str) -> Tuple[str, float, Dict]:
        """
        Calculates Expected Free Energy based on Swarm Hypotheses.
        """
        # If consensus is strong (>80%), skip expensive calculation
        # if final_score > 80: return None
        
        # We evaluate 3 Policies: BUY, SELL, HOLD (WAIT)
        policies = ["BUY", "SELL", "HOLD"]
        
        # Context: Aggregated Swarm Sentiment & Market State
        # We need to construct a context dict for the Free Energy Minimizer
        
        # Calculate Consensus Score for the Current Decision
        consensus_score = 0.0
        count = 0
        for t in thoughts:
             if t.signal_type == current_consensus:
                  consensus_score += t.confidence
                  count += 1
        if count > 0: consensus_score /= count
        
        # Get Volatility/Entropy from Short Term Memory if available
        volatility = 0.01
        entropy = 0.5
        if 'market_metrics' in self.short_term_memory:
             volatility = self.short_term_memory['market_metrics'].get('volatility', 0.01)
             entropy = self.short_term_memory['market_metrics'].get('entropy', 0.5)
             
        context = {
             'consensus_decision': current_consensus,
             'consensus_confidence': consensus_score,
             'volatility': volatility,
             'entropy': entropy
        }

        result = self.free_energy_minimizer.select_best_policy(policies, context)
        
        best_policy = result['selected_policy']
        best_G = result['best_G']
        
        # Map HOLD -> WAIT
        if best_policy == "HOLD": best_policy = "WAIT"
        
        # If Active Inference strongly disagrees with Consensus, we prefer Active Inference
        # (Generative Model is 'smarter' than simple voting)
        
        # For now, we only use it to BREAK TIES or Veto weak signals
        if current_consensus == "WAIT" and best_policy != "WAIT":
             return (best_policy, 65.0, {'active_inference_G': best_G})
             
        if current_consensus != "WAIT" and best_policy == "WAIT":
             # Active Inference suggests Holding is safer (lower Free Energy)
             # But if Consensus is VERY strong, we ignore the caution (Risk Taker)
             # MODIFIED: Lowered threshold from 80.0 to 60.0. Added Mode sensitivity.
             
             threshold = 60.0
             mode = context.get('mode', 'SNIPER')
             if mode in ['HYDRA', 'WOLF_PACK']:
                 threshold = 50.0 # Aggressive modes ignore caution more easily
                 
             if context['consensus_confidence'] > threshold:
                  logger.info(f"ACTIVE INFERENCE: Constraint OVERRIDDEN by Swarm Power ({context['consensus_confidence']} > {threshold})")
                  return (current_consensus, context['consensus_confidence'], {'active_inference_veto': False, 'note': 'High Confidence Override'})
             
             logger.info(f"ACTIVE INFERENCE: Vetoing {current_consensus} (Confidence {context['consensus_confidence']} < {threshold})")
             return ("WAIT", 0.0, {'active_inference_veto': True, 'G': best_G})
             
        return None


    def calculate_dynamic_slots(self, volatility: float, trend_strength: float, mode: str = "SNIPER") -> int:
        """
        Calculates the optimal number of operational slots (max positions).
        
        AGI Logic:
        - High Volatility (Chaos) -> Reduce Slots (Shield).
        - Strong Trend (Order) -> Increase Slots (Spear).
        - Mode Factor -> Wolf Pack gets multiplier.
        """
        # Base Slots
        base_slots = 5
        
        # 1. Mode Multiplier
        if mode == "WOLF_PACK":
            base_slots = 10
        elif mode == "HYBRID":
            base_slots = 10
        elif mode == "AGI_MAPPER":
            base_slots = 12
            
        # 2. Entropy / Volatility Damper
        # Volatility 0-100
        # If Vol > 50 (Choppy/Risky) -> Reduce
        # If Vol < 20 (Calm) -> Neutral
        entropy_factor = 1.0
        if volatility > 60:
            entropy_factor = 0.75 # Less punishment (was 0.5)
        elif volatility > 40:
            entropy_factor = 0.9 # Mild reduction (was 0.8)
            
        # 3. Trend Spear
        # Trend 0-100
        # Strong Trend -> Add slots to pyramid
        trend_factor = 1.0
        if trend_strength > 70:
            trend_factor = 1.5
        elif trend_strength > 90:
            trend_factor = 2.0
            
        final_slots = int(base_slots * entropy_factor * trend_factor)
        
        # Hard Limits
        final_slots = max(1, min(final_slots, 25))
        
        # logger.debug(f"AGI SLOTS: Base {base_slots} * Ent {entropy_factor} * Trend {trend_factor} = {final_slots}")
        
        return final_slots

    def _process_agi_adjustments(self, agi_adj: Dict) -> List[SwarmSignal]:
        """
        Converts AGI adjustments into swarm signals (e.g., Premonitions).
        Does NOT modify agi_adj in place.
        """
        signals = []
        
        # 1. Premonition Signal
        premonition = agi_adj.get('premonition', {})
        if premonition:
            direction = premonition.get('direction', 'WAIT')
            prob = premonition.get('monte_carlo_prob', 0.0)
            
            if direction in ["BUY", "SELL"] and prob > 0.6:
                # Create a synthetic signal
                sig = SwarmSignal(
                    source="Premonition_Swarm",
                    signal_type=direction,
                    confidence=prob * 100.0,
                    meta_data={'type': 'PREMONITION', 'outcome': premonition.get('optimistic_outcome', 0)}
                )
                signals.append(sig)
                
        # 2. Ontology / Nuance Signal
        nuance = agi_adj.get('ontological_nuance', {})
        if nuance:
            concept = nuance.get('high_level_concept', '')
            if "CRASH" in concept or "EXPLOSION" in concept:
                 # Extreme Event Warning
                 sig = SwarmSignal(
                    source="Ontology_Swarm",
                    signal_type="WAIT",
                    confidence=95.0,
                    meta_data={'type': 'ONTOLOGY_WARNING', 'concept': concept}
                 )
                 signals.append(sig)
                 
        return signals

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
        # CIVIL WAR CHECK (Both sides strong)
        if score_buy > 2000 and score_sell > 2000:
             # Calculate Ratio
             ratio = score_buy / score_sell if score_sell > 0 else 1.0
             
             # "Looser" Restriction: Only block if ratio is VERY tight (0.95 - 1.05)
             # User requested "less vetting".
             if 0.95 < ratio < 1.05: 
                 logger.warning(f"CIVIL WAR DETECTED: Buyers({score_buy:.0f}) vs Sellers({score_sell:.0f}). Gridlock.")
                 
                 # AGI INTERVENTION (Tie-Breaker)
                 # We summon the GrandMaster to decide the fate of the Civil War.
                 if hasattr(self.grandmaster, 'perceive_and_decide'):
                      # Create minimal context
                      gm_ctx = {'close': float(self.last_price)}
                      verdict = self.grandmaster.perceive_and_decide(gm_ctx)
                      
                      if verdict == "BUY":
                          logger.info("GRANDMASTER INTERVENTION: Overrode Civil War -> BUY")
                          return "BUY", 75.0, {"reason": "GrandMaster Tie-Break"}
                      elif verdict == "SELL":
                          logger.info("GRANDMASTER INTERVENTION: Overrode Civil War -> SELL")
                          return "SELL", 75.0, {"reason": "GrandMaster Tie-Break"}
                      else:
                          return "WAIT", 0.0, {}
                 else:
                      return "WAIT", 0.0, {}
                      
             # If ratio is slighly skewed (e.g. 1.2), allow the dominant side but penalize confidence
             # This prevents blocking valid but contested moves.
             pass

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
        # DISABLED: This was causing 99% confidence on all trades
        # Physics swarms were bypassing transformer consensus entirely
        # Now all decisions go through proper confidence calculation
        # TODO: Re-enable with proper confidence scaling if needed
        return None
        
        # Original code (kept for reference):
        # physics_agents = ['SingularitySwarm', 'HyperdimensionalSwarm', 'KinematicSwarm', 'VortexSwarm', 'Schrodinger_Newton_Swarm']
        # high_command = [t for t in thoughts if t.source in physics_agents and t.confidence > 95.0]
        # if high_command:
        #     top_signal = sorted(high_command, key=lambda x: x.confidence, reverse=True)[0]
        #     return (top_signal.signal_type, top_signal.confidence, top_signal.meta_data)
        # return None

    def _transformer_consensus(self, thoughts: List[SwarmSignal], weights: Dict, 
                               current_state_vector: Dict, allowed_actions: List[str], 
                               mode: str = "SNIPER", regime: str = "UNKNOWN",
                               agi_context: Dict = None) -> Tuple[str, float, Dict]:
        """
        Phase 4: Transformer-based Consensus Mechanism.
        Uses MCTS-style readout (simulated via heuristic attention) to find best action.
        """
        d_model = 64
        vectors = []
        
        # Track raw faction strengths (before cancellation)
        buy_strength = 0.0
        sell_strength = 0.0
        total_weight = 0.0
        
        # Initialize meta_data
        meta_data = {
            'attention_weights': {},
            'regime': regime,
            'mode': mode
        }
        
        # Merge AGI Context IMMEDIATELY if provided
        if agi_context:
             meta_data.update(agi_context)
             meta_data['agi_integrated'] = True

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
             if t.signal_type == "BUY": 
                 val = 1
                 buy_strength += t.confidence * w
             elif t.signal_type == "SELL": 
                 val = -1
                 sell_strength += t.confidence * w
             total_weight += w
             
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
        
        # === NEW: Calculate confidence from DOMINANT FACTION, not net score ===
        # This prevents BUY/SELL cancellation from killing confidence
        
        dominant_strength = max(buy_strength, sell_strength)
        total_strength = buy_strength + sell_strength
        
        # Consensus ratio: How unified is the swarm?
        if total_strength > 0:
            consensus_ratio = dominant_strength / total_strength  # 0.5 = split, 1.0 = unanimous
        else:
            consensus_ratio = 0.0
        
        # Base confidence: Average confidence of dominant faction votes
        active_votes = len([t for t in thoughts if t.signal_type in ["BUY", "SELL"]])
        if active_votes > 0:
            base_conf = dominant_strength / (active_votes * 1.5)  # Normalized to ~50-70 range
        else:
            base_conf = 0.0
        
        # Consensus boost: Unanimous swarm = 1.2x, Split swarm = 0.9x
        consensus_boost = 0.9 + (consensus_ratio - 0.5) * 0.6  # Range 0.6 to 1.2
        
        # Mode multipliers (reduced for realism)
        mode_mult = 1.0
        if mode == "WOLF_PACK": mode_mult = 1.2
        elif mode == "HYDRA": mode_mult = 1.3  # Moderate boost, not 2x
        elif mode == "SNIPER": mode_mult = 0.9  # Conservative
        
        # Calculate and CAP at 85% (100% is unrealistic)
        final_conf = min(85.0, base_conf * consensus_boost * mode_mult)
        
        # Determine direction from decision_score
        if decision_score > 0.02: 
            final_decision = "BUY"
        elif decision_score < -0.02: 
            final_decision = "SELL"
        
        logger.info(f"AGI ATTENTION: Decision={final_decision} (Score={decision_score:.3f}) | Memory Bias={rag_vec[0]:.3f}")
        
        # --- DIALECTIC SYNTHESIS (Hegelian Logic) ---
        # Instead of averaging +1 and -1 to 0, we check the DOMINANT CLUSTER.
        # If Regime is TRENDING, we ignore the Counter-Trend dissenters (Civil War Resolution).
        
        if regime == "TRENDING" and final_decision == "WAIT":
             # Check if Trend Swarms are voting BUY
             trend_vote = 0.0
             for t in thoughts:
                 if "Trending" in t.source or "Apex" in t.source or "Technical" in t.source:
                      if t.signal_type == "BUY": trend_vote += t.confidence
                      elif t.signal_type == "SELL": trend_vote -= t.confidence
             
             if trend_vote > 200: # Strong Trend Vote masked by Oscillators
                 logger.info("DIALECTIC SYNTHESIS: Regime is TRENDING. Ignoring Counter-Trend Dissenters.")
                 final_decision = "BUY"
                 final_conf = 85.0 # High Confidence Override
                 
             elif trend_vote < -200:
                 logger.info("DIALECTIC SYNTHESIS: Regime is TRENDING. Ignoring Counter-Trend Dissenters.")
                 final_decision = "SELL"
                 final_conf = 85.0

        # Wolf Pack Boost (Winner Takes All)
        if mode == "WOLF_PACK" and final_decision != "WAIT":
             # If we have a winner, boost it to overcome hesitation
             final_conf = max(final_conf, 75.0)
        
        # HYDRA Boost (Aggressive Multi-Vector)
        if mode == "HYDRA" and final_decision != "WAIT":
             final_conf = max(final_conf, 65.0)  # Minimum 65% for HYDRA
 
        
        # Boost confidence for strong consensus (capped at 85% - 100% is unrealistic)
        if final_conf > 50:
            final_conf = min(85.0, final_conf * 1.1)  # 10% boost, max 85% 
        
        if final_decision not in allowed_actions: final_decision = "WAIT"

        return final_decision, final_conf, {'attention_weights': attn_weights.tolist()}


