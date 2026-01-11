
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
from core.agi.inference.causal_engine import CausalInferenceEngine
from core.risk.entropy_harvester import EntropyHarvester # Phase 26
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
        self.last_learning_time = 0 # Phase 4.2 Update
        
        # Validation Attributes
        self.last_price = 0.0
        self.last_consensus_state = {}
        
        # AGI Hierarchical Clusters (Brain Regions)
        self.clusters = {
            'PHYSICS': ['Laplace_Swarm', 'Riemann_Swarm', 'Fermi_Swarm', 'Bose_Swarm', 'Thermodynamics_Swarm', 'Gravity_Swarm', 'Navier_Stokes_Swarm', 'Maxwell_Swarm'],
            'QUANTUM': ['Quantum_Grid', 'Schrodinger_Swarm', 'Heisenberg_Swarm', 'Antimatter_Swarm', 'Entanglement_Swarm', 'Zero_Point_Swarm', 'Tachyon_Swarm'],
            'PRICING': ['Trending_Swarm', 'Sniper_Swarm', 'Technical_Swarm', 'Fractal_Vision', 'Wavelet_Swarm', 'Spectral_Swarm'],
            'INSTITUTIONAL': ['Whale_Swarm', 'Order_Flow', 'Liquidity_Map', 'SMC_Swarm', 'News_Swarm', 'Apex_Swarm'],
            'META': ['Veto_Swarm', 'Red_Team_Swarm', 'Dream_Swarm', 'Reflection_Swarm', 'Zen_Swarm', 'Council_Swarm', 'Overlord_Swarm', 'Sovereign_Swarm']
        }
        
        # --- PHASE 200+: CAUSAL INFERENCE ---
        self.causal_engine = CausalInferenceEngine()
        
        # --- PHASE 26: ENTROPY HARVESTER (CHAOS) ---
        self.entropy_harvester = EntropyHarvester(None, None) # Standalone mode for Analysis

    async def initialize_swarm(self):
        logger.info("--- GENESIS PROTOCOL: AWAKENING SWARM (OPTIMIZED) ---")
        self.active_agents = []
        
        # ========== ESSENTIAL CORE (28 Agents) ==========
        
        # 1. Safety & Veto (3)
        self.active_agents.append(VetoSwarm()) 
        self.active_agents.append(BlackSwanSwarm())
        from analysis.swarm.red_team_swarm import RedTeamSwarm
        self.active_agents.append(RedTeamSwarm())
        
        # 2. Core Technical (5)
        self.active_agents.append(TrendingSwarm())
        self.active_agents.append(SniperSwarm())
        self.active_agents.append(QuantSwarm())
        from analysis.swarm.technical_swarm import TechnicalSwarm
        self.active_agents.append(TechnicalSwarm())
        self.active_agents.append(FractalVisionSwarm())
        
        # 3. Institutional (5)
        self.active_agents.append(WhaleSwarm())
        self.active_agents.append(OrderFlowSwarm())
        self.active_agents.append(LiquidityMapSwarm())
        from analysis.swarm.smc_swarm import SmartMoneySwarm
        self.active_agents.append(SmartMoneySwarm())
        self.active_agents.append(NewsSwarm())
        
        # 4. Physics & Chaos (5)
        self.active_agents.append(ChaosSwarm())
        self.active_agents.append(QuantumGridSwarm())
        self.active_agents.append(SingularitySwarm())
        self.active_agents.append(GravitySwarm())
        self.active_agents.append(ThermodynamicSwarm())
        
        # 5. Meta-Cognition (5)
        self.active_agents.append(OracleSwarm())
        self.active_agents.append(CouncilSwarm())
        self.active_agents.append(ActiveInferenceSwarm())
        from analysis.swarm.dream_swarm import DreamSwarm
        from analysis.swarm.reflection_swarm import ReflectionSwarm
        self.active_agents.append(DreamSwarm())
        self.active_agents.append(ReflectionSwarm())
        
        # 6. Time & Temporal (3)
        from analysis.swarm.chronos_swarm import ChronosSwarm
        self.active_agents.append(ChronosSwarm())
        self.active_agents.append(TimeKnifeSwarm())
        self.active_agents.append(EventHorizonSwarm())
        
        # 7. Causal & Bayesian (2)
        self.active_agents.append(CausalSwarm())
        self.active_agents.append(BayesianSwarm())
        
        # ========== DEACTIVATED (Commented for Performance) ==========
        # Phase 50-90 Physics: Schrodinger, Heisenberg, Fermi, Bose, Navier, etc.
        # Geometric: Topological, Manifold, Ricci, Minkowski, Penrose, Godel
        # Meta Overlap: Overlord, Sovereign, Apex, Zen, NeuralLace
        # See implementation_plan.md for full list
        
        logger.info(f"Swarm Initialized with {len(self.active_agents)} Optimized Agents (was 93).")

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
                                                        config=config,
                                                        data_context={'df': df_m5, 'tick': tick})
        return decision, score, meta

    def synthesize_thoughts(self, thoughts: List[SwarmSignal] = None, 
                         current_state_vector: Dict = None, 
                         mode: str = "SNIPER", 
                         agi_context: Dict = None,
                         config: Dict = None,
                         data_context: Dict = None) -> Tuple[str, float, Dict]:
                         
        # Phase 5: Regime-Based Threshold Adjustment
        base_threshold = self.alpha_threshold # Default 60.0
        regime_mod = 0.0
        
        if agi_context and 'threshold_modifier' in agi_context:
             regime_mod = agi_context['threshold_modifier']
             base_threshold += regime_mod
             # logger.info(f"REGIME ADJUSTMENT: Threshold {self.alpha_threshold} -> {base_threshold} (Mod: {regime_mod:+.1f})")
             
        if not thoughts:
            thoughts = self.bus.get_recent_thoughts()
        
        if not thoughts: return "WAIT", 0.0, {}
        
        # ===== PRIORITY OVERRIDES (Converted to PENALTY SCORES in Phase 2) =====
        penalty_score = 0.0
        penalty_reasons = []

        # 1. VETO signals -> Heavy Penalty
        for t in thoughts:
            if t.signal_type == "VETO":
                reason = t.meta_data.get('reason', 'Unknown')
                logger.warning(f"VETO SIGNAL: {reason} from {t.source} - Applying Penalty.")
                penalty_score += 25.0  # Phase 2: Penalty instead of hard block
                penalty_reasons.append(f"VETO:{t.source}")
        
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

        # --- SNIPER PROTOCOL v4.0: GATE 3 (TIER 1 UNANIMITY) ---
        # The "Technical Reality Check"
        # Fractal Vision (Structure) AND Liquidity Map (Targets) MUST agree.
        # If one says BUY and other says SELL, we are in a CHAOS zone.
        
        fractal_vote = None
        liquidity_vote = None
        
        for t in thoughts:
            if t.source == "Fractal_Vision_Swarm":
                fractal_vote = t.signal_type
            elif t.source == "Liquidity_Map_Swarm":
                liquidity_vote = t.signal_type
                
        # Only enforce if both are present (some swarms might be asleep)
        if fractal_vote and liquidity_vote:
            if fractal_vote in ["BUY", "SELL"] and liquidity_vote in ["BUY", "SELL"]:
                if fractal_vote != liquidity_vote:
                    # ✅ PHASE 0 FIX #2: Convert hard block to penalty
                    logger.warning(f"GATE 3: Tier 1 Conflict (Fractal={fractal_vote} vs Liquidity={liquidity_vote}). Applying Penalty +15.")
                    penalty_score += 15.0
                    penalty_reasons.append(f"GATE3_CONFLICT({fractal_vote}vs{liquidity_vote})")
        
        # -------------------------------------------------------

        # 1. Physics Reality Check (Tier 1 Check)
        physics_decision = self._resolve_physics_conflict(thoughts)
        if physics_decision:
             action, conf, meta = physics_decision
             if action not in allowed_actions and not strong_macro:
                   logger.warning(f"PHYSICS CONFLICT: {action} denied. Applying Penalty.")
                   penalty_score += 15.0
                   penalty_reasons.append("PHYSICS_CONFLICT")

        # 2. Holographic Recall (Tier 2 Check)
        if current_state_vector:
             intuition = self.holographic_memory.retrieve_intuition(current_state_vector)
             if intuition < -30.0:  
                  logger.warning(f"DEJA VU: Holographic Danger (Score: {intuition:.2f}). Applying Penalty.")
                  penalty_score += 20.0
                  penalty_reasons.append(f"HOLOGRAPHIC_DANGER({intuition:.1f})")
        
        # 0. Sovereign Executive Order (Meta-Cognition)
        meta_data = {} # Initialize meta_data
        weights = {}
        
        # A. Neuroplasticity Weights (Long Term Memory)
        learned_weights = self.neuroplasticity.get_dynamic_weights()
        
        # B. Sovereign Weights (Regime Adaptation)
        sov_weights = {}
        for t in thoughts:
             if t.source == "Sovereign_Swarm" and t.signal_type == "META_INFO":
                 sov_weights = t.meta_data.get('weight_vector', {})
                 regime = t.meta_data.get('regime', 'UNKNOWN')
                 # logger.debug(f"SOVEREIGN ACT: Regime={regime} | Weights Applied.")
        
        # C. Merge (Sovereign * Learned)
        # Allows Sovereign to Temporarily Override long-term learning if Regime changes drastically
        all_keys = set(learned_weights.keys()).union(set(sov_weights.keys()))
        for k in all_keys:
             w_learn = learned_weights.get(k, 1.0) # Default to base
             w_sov = sov_weights.get(k, 1.0)
             w_sov = sov_weights.get(k, 1.0)
             weights[k] = w_learn * w_sov
             
        # D. Dynamic Performance Weighting (Phase 4.2)
        # Fetch Stats from Learning Engine
        benchwarmers = []
        if self.agi and hasattr(self.agi, 'learning') and self.agi.learning:
             perf_map = self.agi.learning.get_all_performances()
             
             for k in weights.keys():
                  # Match swarm name key (fuzzy match)
                  # If agent name is "TrendingSwarm", key might be "Trending_Swarm"
                  # Performance map uses sources like "Trending_Swarm" directly from thoughts.
                  
                  stats = perf_map.get(k)
                  if stats:
                       wins = stats.get('wins', 0)
                       losses = stats.get('losses', 0)
                       total = wins + losses
                       
                       if total > 5: # Min sample size
                            win_rate = wins / total
                            
                            if win_rate < 0.40: # Benchwarmer
                                 weights[k] = 0.0
                                 benchwarmers.append(f"{k}({win_rate:.0%})")
                            elif win_rate > 0.60: # Star Player
                                 weights[k] *= 1.2
                                 
                            # Phase 4.3: Confidence Calibration
                            # Bias = Predicted Conf - Actual Win Rate
                            bias = self.agi.learning.get_calibration_bias(k)
                            if bias > 15.0: # Overconfident (e.g. Says 90%, Wins 50% -> Bias 40)
                                 weights[k] *= 0.8
                                 # logger.debug(f"CALIBRATION: Penalizing Overconfident {k} (Bias {bias:.1f})")
                            elif bias < -15.0: # Underconfident (e.g. Says 60%, Wins 90% -> Bias -30)
                                 weights[k] *= 1.15
                                 # logger.debug(f"CALIBRATION: Boosting Underconfident {k} (Bias {bias:.1f})")
                                 
        if benchwarmers:
             logger.info(f"COACH: Benchwarming Underperformers: {', '.join(benchwarmers)}")
             
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
        
        # --- PHASE 2: TIER CONSOLIDATION ---
        # Tier 1: Perceptor (Physics + Quantum + Pricing)
        t1_sources = [cluster_scores.get('PHYSICS', 0), cluster_scores.get('QUANTUM', 0), cluster_scores.get('PRICING', 0)]
        tier_1_score = sum(t1_sources) / 3.0
        
        # Tier 2: Synthesizer (Institutional + Meta)
        t2_sources = [cluster_scores.get('INSTITUTIONAL', 0), cluster_scores.get('META', 0)]
        tier_2_score = sum(t2_sources) / 2.0
        
        meta_data['tier_scores'] = {'perceptor': tier_1_score, 'synthesizer': tier_2_score}
        logger.info(f"TIER SCORES: Perceptor={tier_1_score:.1f}% | Synthesizer={tier_2_score:.1f}%")

        # Apply Boosts based on Tiers
        # Use Tier 1 to filter Tier 2? Or just weighted boost?
        # If Perceptor is weak (< 50), dampen everything.
        if tier_1_score < 50.0:
             logger.info("TIER 1 WEAK: Dampening weights.")
             for k in weights: weights[k] *= 0.8
             
        # If Synthesizer is strong, boost confidence
        if tier_2_score > 80.0:
             for k in weights: weights[k] *= 1.2

        # 3. Transformer Attention Consensus (AGI Brain)
        if 'allowed_actions' not in locals(): allowed_actions = ["BUY", "SELL", "WAIT", "EXIT_LONG", "EXIT_SHORT", "EXIT_ALL"]
        if 'regime' not in locals(): regime = "UNKNOWN"
        
        # Pass agi_context into transformer to be merged immediately
        final_decision, final_score, meta_data = self._transformer_consensus(thoughts, weights, current_state_vector, allowed_actions, mode=mode, regime=regime, agi_context=agi_context)
        
        # --- PHASE 2: APPLY PENALTIES ---
        if penalty_score > 0:
            original_score = final_score
            final_score = max(0.0, final_score - penalty_score)
            logger.info(f"PENALTY APPLIED: {original_score:.1f} -> {final_score:.1f} (Reasons: {penalty_reasons})")
            
            # If score drops too low, force WAIT
            if final_score < 20.0 and final_decision != "EXIT_ALL":
                 final_decision = "WAIT"
                 meta_data['veto_reason'] = f"Penalties: {','.join(penalty_reasons)}"
        

        
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
            
            # --- PHASE 155: ANTI-SCHIZOPHRENIA GUARD (Directional Lock) ---
            # Prevents opening BUY and SELL on the same symbol (unless Hedging Mode enabled).
            # User reported bot opening disjointed orders in the same candle.
            
            if final_decision in ["BUY", "SELL"]:
                 open_positions = agi_context.get('open_positions', [])
                 if open_positions:
                      # Check for opposite positions on SAME symbol
                      has_buy = any(p['symbol'] == agi_context.get('symbol', 'UNKNOWN') and p['type'] == 'BUY' for p in open_positions)
                      has_sell = any(p['symbol'] == agi_context.get('symbol', 'UNKNOWN') and p['type'] == 'SELL' for p in open_positions)
                      
                      # Note: Dictionary usually has type as String "BUY"/"SELL" or int 0/1. 
                      # main.py passes trades_json where 'type' is likely INT (0=Buy, 1=Sell).
                      # Let's handle both just in case.
                      
                      current_symbol = agi_context.get('symbol')
                      
                      # Re-scan with robust type check
                      has_buy = False
                      has_sell = False
                      
                      for p in open_positions:
                           if p.get('symbol') != current_symbol: continue
                           
                           t_type = p.get('type')
                           if t_type == 0 or t_type == "BUY": has_buy = True
                           if t_type == 1 or t_type == "SELL": has_sell = True
                      
                      # ELASTIC SNAP LOGIC (Smart Inversion)
                      # If we have a High Conviction (>75%) signal in the opposite direction,
                      # we admit the previous trade was wrong (or trend changed) -> CLOSE IT -> OPEN REVERSE.
                      # If signal is weak, we VETO (Anti-Hedge).
                      
                      is_high_conviction = final_score >= 75.0
                      
                      if final_decision == "SELL" and has_buy:
                           if is_high_conviction:
                                logger.warning(f"ELASTIC SNAP: High Confidence Short ({final_score:.1f}%). Closing Longs to Reverse.")
                                # Return EXIT_LONG to Execution Engine (which will close buys)
                                # Next tick, the SELL signal will persist (Or we can force it here? No, let's just exit first for safety/margin).
                                # Actually, better to switch signal to "EXIT_LONG".
                                # The loop usually processes one action.
                                final_decision = "EXIT_LONG" 
                                meta_data['reason'] = "Elastic Snap Reverse (High Conf Short)"
                           else:
                                logger.warning(f"DIRECTIONAL LOCK: Blocked SELL on {current_symbol} (Conf {final_score:.1f}% < 75%) because we hold BUYs.")
                                final_decision = "WAIT"
                                meta_data['veto_reason'] = "Directional Lock (Anti-Hedge)"
                           
                      elif final_decision == "BUY" and has_sell:
                           if is_high_conviction:
                                logger.warning(f"ELASTIC SNAP: High Confidence Long ({final_score:.1f}%). Closing Shorts to Reverse.")
                                final_decision = "EXIT_SHORT"
                                meta_data['reason'] = "Elastic Snap Reverse (High Conf Long)"
                           else:
                                logger.warning(f"DIRECTIONAL LOCK: Blocked BUY on {current_symbol} (Conf {final_score:.1f}% < 75%) because we hold SELLs.")
                                final_decision = "WAIT"
                                meta_data['veto_reason'] = "Directional Lock (Anti-Hedge)"

        # Harvester Override (Priority)
        for t in thoughts:
             if t.signal_type == "EXIT_ALL": return ("EXIT_ALL", 99.0, t.meta_data)
             if t.signal_type == "VETO": return ("WAIT", 0.0, {})

        # --- TREND VETO (Counter-Trend Block) ---
        # If Fractal/Trend Swarm signals strong trend, block opposing trades.
        trend_direction = None
        trend_strength = 0.0
        is_exhausted = False
        
        for t in thoughts:
             if t.source in ["Fractal_Vision_Swarm", "Trending_Swarm", "Navier_Stokes_Swarm"]:
                 # Check for Exhaustion FIRST
                 if t.meta_data.get('exhaustion'):
                     is_exhausted = True
                     logger.warning(f"TREND EXHAUSTION DETECTED: {t.meta_data.get('exhaustion_reason')}")
                     
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
             # TREND AIKIDO: If Trend is POWERFUL (>80), we don't just Veto. We Join.
             # The Counter-Trend signal provided the Volatility Trigger, but the Macro Trend dictates direction.
             if trend_strength > 80.0 and not is_exhausted:
                  logger.warning(f"TREND AIKIDO: Inverting {final_decision} -> {trend_direction} (Trend Conf: {trend_strength:.1f})")
                  
                  # Invert Decision
                  final_decision = trend_direction 
                  
                  # Boost confidence (Trend Support + Volatility Trigger)
                  final_score = (final_score + trend_strength) / 2.0
                  
                  meta_data['method'] = "TREND_AIKIDO"
                  meta_data['aikido_flip'] = True
             else:
                  # Weak Trend? 
                  # OLD LOGIC: Standard Safety Veto.
                  # NEW LOGIC: User requested "More Reasoning".
                  # If Trend is WEAK (e.g. 77.8), a strong Counter-Signal might be a Reversal or Correction.
                  
                  # Check if the Counter-Signal is strong enough to justify a fight
                  if final_score > 70.0:
                       # Allow the trade but mark as "Reversal"
                       logger.info(f"TREND COUNTER-ATTACK: Allowing {final_decision} against Weak {trend_direction} (Trend Conf: {trend_strength:.1f} < 80). Reason: High Conviction Reversal.")
                       meta_data['method'] = "REVERSAL_SCALP"
                       meta_data['trend_conflict'] = True
                       # Mild penalty to confidence for fighting trend
                       final_score *= 0.9
                  else:
                       # Weak Signal vs Weak Trend -> Chop. Stay out.
                       logger.warning(f"TREND VETO: Blocked Weak {final_decision} ({final_score:.1f}) against Weak {trend_direction} Trend.")
                       final_decision = "WAIT"
                       meta_data['veto_reason'] = f"Chop Protection ({trend_direction})"

        # 4. Active Inference Arbitration (The Free Energy Principal)
        # If the Swarm is split or uncertain, we check which decision minimizes Free Energy (Risk + Ambiguity)
        active_inf_decision = self._resolve_active_inference(thoughts, final_decision, weights)
        if active_inf_decision:
             final_decision, final_score, meta_new = active_inf_decision
             meta_data.update(meta_new)
             logger.info(f"ACTIVE INFERENCE: Override -> {final_decision} (Minimizes Free Energy)")

        # 5. Metacognitive Reflection (Self-Correction)
        # "Before I act, let me think about why I am acting."
        
        # --- PHASE 26: ENTROPY STATE INJECTION ---
        # Calculate Market Entropy (Chaos)
        if data_context and 'df' in data_context:
             df_entropy = data_context['df']
             if 'close' in df_entropy.columns:
                 market_entropy = self.entropy_harvester.calculate_shannon_entropy(df_entropy['close'])
                 meta_data['market_entropy'] = market_entropy
                 
                 # 1. Start with Entropy (0.0 to 1.0)
                 # > 0.8 = High Chaos (Random) -> Favor Mean Reversion? Risk Off?
                 # < 0.3 = Order (Trend) -> Favor Trend?
                 
                 if market_entropy > 0.85:
                      logger.warning(f"ENTROPY WARNING: High Chaos ({market_entropy:.2f}). Market is Random.")
                      # Penalize Trend Trades
                      if meta_data.get('method') == "TREND_AIKIDO" or trend_strength > 50:
                           final_score *= 0.7
                           meta_data['entropy_penalty'] = True
                           
                 elif market_entropy < 0.30:
                      # High Order -> Trend likely stable
                      if meta_data.get('method') == "TREND_AIKIDO" or trend_strength > 50:
                           final_score *= 1.15
                           meta_data['entropy_boost'] = True
        
        # --- PHASE 200: CAUSAL TRUTH FILTER ---
        # Before Metacognition, we check "Ontological Truth".
        # If Price is Hallucinating (No Causal support), we kill it.
        if final_decision in ["BUY", "SELL"]:
             df_causal = data_context.get('df') if data_context else None
             causal_analysis = self.causal_engine.analyze_causality(df_causal) 
             meta_data['causal_truth'] = causal_analysis['truth_score']
             
             if causal_analysis['truth_score'] < 0.4:
                  # Major Causal Disconnect (Hallucination)
                  logger.warning(f"CAUSAL VETO: Truth Score {causal_analysis['truth_score']:.2f} < 0.4. Blocking Hallucination.")
                  final_decision = "WAIT"
                  meta_data['veto_reason'] = "Causal Disconnect (Hallucination)"
        
        reflection = self.metacognition.reflect(final_decision, final_score, meta_data, {'config': {'mode': mode}})
        final_score = reflection['adjusted_confidence']
        if reflection['notes']:
            meta_data['reflection_notes'] = reflection['notes']
            # If validation failed (score -> 0), switch to WAIT
            # Relaxed Threshold: 38.0 (was 47.0) per Phase 1 Stabilization Plan
            if final_score < 38.0 and final_decision != "EXIT_ALL":
                # NEW LOGIC: Attempt to rescue the decision via Deep Reasoning
                rescued_decision, rescued_score, rescue_reason = self._attempt_deep_reasoning_fix(
                    final_decision, final_score, reflection['notes'], meta_data
                )
                
                if rescued_decision != "WAIT":
                    final_decision = rescued_decision
                    # Boost score slightly above threshold to let it pass execution filters
                    final_score = max(rescued_score, 50.0) 
                    meta_data['rescue_reason'] = rescue_reason
                    meta_data['deep_reasoning'] = True
                    logger.info(f"DEEP REASONING: Rescued 'Poor Reasoning' decision. {rescue_reason}")
                else:
                    final_decision = "WAIT"
                    logger.info(f"METACOGNITION: Vetoed Decision. Score {final_score:.1f} < 47.0. Reason: {reflection['notes']}")
            else:
                logger.info(f"METACOGNITION: Passed. Score {final_score:.1f} (Threshold 47.0)")
        
        # --- PHASE 250: TIME CONFIRMATION BIAS (The Golden Third) ---
        # "Entering at 3:40 (220s) avoids noise and closing algos."
        
        # STRICT GATE: Use System Time as requested ("horario do meu computador")
        # Ensure we are synchronized with the 5-minute grid.
        system_time = time.time()
        seconds_in_bar = int(system_time % 300)
        
        # Phase 5.1: Dynamic Time Gate based on Regime
        regime = meta_data.get('regime', 'UNKNOWN')
        if regime == 'OPTIMAL':
            time_gate = 120 # Earlier entry allowed
        elif regime in ['CRITICAL', 'CHOPPY']:
            time_gate = 180 # Stricter: wait longer
        else:
            time_gate = 150 # Default
        
        if final_decision in ["BUY", "SELL"]:
             meta_data['bar_time'] = seconds_in_bar
             
             # RULE 1: STRICT PRE-GATE (0 - time_gate) -> BLOCK ENTRIES
             # Dynamically adjusted by Regime (Phase 5.1)
             if seconds_in_bar < time_gate:
                  logger.info(f"TIME GATE: Too early ({seconds_in_bar}s). Waiting for Regime Gate ({time_gate}s).")
                  final_decision = "WAIT"
                  meta_data['veto_reason'] = f"Time Gate ({time_gate}s). Current: {seconds_in_bar}s"
                       
             # RULE 2: THE GOLDEN THIRD (220s - 280s) -> BOOST
             elif 150 <= seconds_in_bar <= 280:
                  final_score *= 1.25 # Significant Boost
                  final_score = min(final_score, 99.0)
                  logger.info(f"TIME BIAS: Golden Window ({seconds_in_bar}s). Perfect Entry Window. Score -> {final_score:.1f}")
                  meta_data['golden_time'] = True
                  
             # RULE 3: CLOSING ALGOS (285s+) -> DANGER
             elif seconds_in_bar > 285:
                  final_score *= 0.6
                  logger.warning(f"TIME BIAS: Closing Algo Danger ({seconds_in_bar}s). Penalizing. Score -> {final_score:.1f}")
                  if final_score < 55.0:
                       final_decision = "WAIT"
                       meta_data['veto_reason'] = "Closing Algo Risk"
                       
             # --- PHASE 255: ALGORITHMIC QUARTER CYCLES (Macro Timing) ---
             # Determines bias based on 15-minute blocks within the hour.
             # tick_time is Epoch Seconds.
             # Minute of hour:
             minute_of_hour = int((time.time() / 60) % 60)
             quarter_cycle = (minute_of_hour // 15) + 1 # 1, 2, 3, 4
             meta_data['quarter_cycle'] = quarter_cycle
              
             # Q1 (00-15): The Trap / Accumulation. Breakouts often fail.
             if quarter_cycle == 1:
                  if final_decision != "WAIT":
                       # Slight Penalty for Breakouts due to "Judas Swing" risk
                       # But if it's a "Reversal" trade (Sniper), it's good.
                       if meta_data.get('method') in ["BREAKOUT", "TREND_AIKIDO"]:
                            final_score *= 0.9
                            logger.info(f"CYCLE BIAS: Q1 (The Trap). Caution on Breakout. Score -> {final_score:.1f}")
              
             # Q2 (15-30): The Shift / Reversal. True Trend often starts here.
             elif quarter_cycle == 2:
                  # Good for Reversals and New Trends
                  final_score *= 1.1
                  logger.info(f"CYCLE BIAS: Q2 (The Shift). Boosting Validity. Score -> {final_score:.1f}")
                  
             # Q3 (30-45): The Flow. Best for Continuation.
             elif quarter_cycle == 3:
                   # Best for Trend Following
                   if meta_data.get('method') == "TREND_AIKIDO" or trend_strength > 50:
                        final_score *= 1.2
                        logger.info(f"CYCLE BIAS: Q3 (The Flow). Strong Trend Support. Score -> {final_score:.1f}")
                        
             # Q4 (45-60): The Fix / Closing. Mean Reversion.
             elif quarter_cycle == 4:
                   # High risk of closing algos.
                   final_score *= 0.9
                   logger.info(f"CYCLE BIAS: Q4 (The Fix). Caution (Closing Algos). Score -> {final_score:.1f}")
        
        # 6. Neural Resonance Sync (Phase 150+)
        # Synchronize decision with user intent model (Symbiosis)
        if hasattr(self.agi, 'resonance_bridge'):
            u_intent = config.get('user_intent', {'target': 'PROFIT', 'risk': 'MODERATE'}) if config else {'target': 'PROFIT', 'risk': 'MODERATE'}
            resonance = self.agi.resonance_bridge.synchronize(u_intent, {'decision': final_decision, 'score': final_score})
            meta_data['symbiosis_sync'] = resonance['symbiosis_confidence']
            if resonance['fusion_stream'] == "SYMBIOTIC_CONSENSUS":
                logger.info(f"NEURAL RESONANCE: Absolute Symbiosis achieved ({resonance['symbiosis_confidence']:.2%})")

        return (final_decision, final_score, meta_data)

    def _resolve_active_inference(self, thoughts: List[SwarmSignal], current_consensus: str, weights: Dict[str, float] = None) -> Tuple[str, float, Dict]:
        """
        Calculates Expected Free Energy based on Swarm Hypotheses.
        """
        if weights is None: weights = {}
        
        # We evaluate 3 Policies: BUY, SELL, HOLD (WAIT)
        policies = ["BUY", "SELL", "HOLD"]
        
        # Context: Aggregated Swarm Sentiment & Market State
        # We need to construct a context dict for the Free Energy Minimizer
        
        # Calculate Consensus Score for the Current Decision
        consensus_score = 0.0
        count = 0
        
        # Calculate Coherence (WEIGHTED Vote Ratio)
        buy_weight = 0.0
        sell_weight = 0.0
        
        for t in thoughts:
             w = weights.get(t.source, 1.0)
             
             if t.signal_type == current_consensus:
                  consensus_score += t.confidence * w
                  # Normalize later? No, raw strength is fine here.
                  count += w
             
             if t.signal_type == "BUY": buy_weight += t.confidence * w
             elif t.signal_type == "SELL": sell_weight += t.confidence * w

        if count > 0: consensus_score /= count # Normalized Weighted Average Confidence
        
        total_weight = buy_weight + sell_weight
        coherence = 0.0
        if total_weight > 0:
             coherence = max(buy_weight, sell_weight) / total_weight
             
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
             'entropy': entropy,
             'coherence': coherence
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
             # Active Inference suggests Holding is safer.
             # We only override if the Swarm is STRONGLY united.
             
             threshold = 60.0
             mode = context.get('mode', 'SNIPER')
             if mode in ['HYDRA', 'WOLF_PACK']:
                 threshold = 50.0 

             # CRITICAL FIX: Check Coherence (Vote Ratio)
             # If Swarm is split (e.g. 51% Buy vs 49% Sell), we CANNOT override safety.
             coherence = context.get('coherence', 0.5) 
             
             # If Consensus is weak (low coherence), we trust Active Inference (WAIT)
             # Relaxed Coherence: 0.48 (was 0.6) per Phase 1 Stabilization
             if coherence < 0.48 and context['consensus_confidence'] < 85.0:
                  logger.info(f"ACTIVE INFERENCE: Vetoing {current_consensus} (Low Coherence {coherence:.2f} < 0.48)")
                  return ("WAIT", 0.0, {'active_inference_veto': True, 'G': best_G})

             if context['consensus_confidence'] > threshold:
                  logger.info(f"ACTIVE INFERENCE: Constraint OVERRIDDEN by Swarm Power ({context['consensus_confidence']} > {threshold} & Coherence {coherence:.2f})")
                  return (current_consensus, context['consensus_confidence'], {'active_inference_veto': False, 'note': 'High Confidence Override'})
             
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
        # 1. Mode Multiplier
        if mode == "WOLF_PACK":
            base_slots = 20 # Boosted for Pack Logic
        elif mode == "HYBRID":
            base_slots = 30 # Increased Capacity
        elif mode == "HYDRA":
            base_slots = 60 # Machine Gun Mode (User Request)
        elif mode == "AGI_MAPPER":
            base_slots = 100 # Unlimited Power for GrandMaster
            
        # 2. Entropy / Volatility Damper
        # Volatility 0-100
        # If Vol > 50 (Choppy/Risky) -> Reduce
        # If Vol < 20 (Calm) -> Neutral
        entropy_factor = 1.0
        if volatility > 60:
            entropy_factor = 0.8 # Less punishment (was 0.5)
        elif volatility > 40:
            entropy_factor = 0.95 # Mild reduction (was 0.8)
            
        # 3. Trend Spear
        # Trend 0-100
        # Strong Trend -> Add slots to pyramid
        trend_factor = 1.0
        if trend_strength > 70:
            trend_factor = 1.5
        elif trend_strength > 90:
            trend_factor = 2.5 # Aggressive Scaling
            
        final_slots = int(base_slots * entropy_factor * trend_factor)
        
        # Hard Limits (Raised significantly for Scalability)
        final_slots = max(1, min(final_slots, 200))
        
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
            # Dynamic Score: Normalize weighted vote to 0-100 range
            final_score = min(99.0, 50.0 + (score_buy / 100.0)) # Base 50 + scaled votes
        elif score_sell > score_buy and score_sell > 500:
            final_decision = "SELL"
            final_score = min(99.0, 50.0 + (score_sell / 100.0))
        elif score_buy > score_sell: # Below threshold but still a winner
            final_decision = "BUY"
            final_score = min(60.0, 30.0 + (score_buy / 50.0)) # Lower confidence for weak signals
        elif score_sell > score_buy:
            final_decision = "SELL"
            final_score = min(60.0, 30.0 + (score_sell / 50.0))
        else:
            # True tie - keep default WAIT but with a measurable score
            final_score = 20.0 # Indicative that analysis happened but was inconclusive

            
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
        
        meta_data = {} # Initialize early for scope safety
        
        # Track Top Agents
        top_buy_agent = "Unknown"
        top_buy_score = -1.0
        top_sell_agent = "Unknown"
        top_sell_score = -1.0
        
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
                 w_conf = t.confidence * w
                 buy_strength += w_conf
                 if w_conf > top_buy_score:
                     top_buy_score = w_conf
                     top_buy_agent = t.source
                     
             elif t.signal_type == "SELL": 
                 val = -1
                 w_conf = t.confidence * w
                 sell_strength += w_conf
                 if w_conf > top_sell_score:
                     top_sell_score = w_conf
                     top_sell_agent = t.source
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
        
        # logger.debug(f"AGI ATTENTION: Decision={final_decision} (Score={decision_score:.3f}) | Memory Bias={rag_vec[0]:.3f}")
        
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
                 # logger.debug("DIALECTIC SYNTHESIS: Regime is TRENDING. Ignoring Counter-Trend Dissenters.")
                 final_decision = "BUY"
                 # Boost calculated confidence rather than forcing 85
                 final_conf = min(85.0, final_conf * 1.20)
                 
             elif trend_vote < -200:
                 # logger.debug("DIALECTIC SYNTHESIS: Regime is TRENDING. Ignoring Counter-Trend Dissenters.")
                 final_decision = "SELL"
                 final_conf = min(85.0, final_conf * 1.20)

        # --- HYDRA-ONLY MODE ENFORCEMENT ---
        # The user requested to use only HYDRA mode from now on.
        mode = "HYDRA"
        
        # HYDRA Boost (Refined: 15% boost instead of hard floor)
        if final_decision != "WAIT":
             final_conf = min(85.0, final_conf * 1.15)
 
        
        # Boost confidence for strong consensus (capped at 85% - 100% is unrealistic)
        if final_conf > 50:
            final_conf = min(85.0, final_conf * 1.1)  # 10% boost, max 85% 
            
        # Phase 5: Dynamic Threshold Check
        # Threshold was calculated at start of synthesis (base_threshold)
        # We need to pass it or recalc provided we are inside the same method... 
        # Actually this method _transformer_consensus is called by synthesize_thoughts.
        # Let's check where the threshold is used. 
        # It's not used inside _transformer_consensus strictly for "WAIT" decision unless we enforce it.
        # But score is returned. The CALLER (synthesize_thoughts or main loop) typically checks score > threshold.
        # However, SwarmOrchestrator returns (Decision, Score). Main.py decides.
        # WAIT! Main.py usually blindly trusts "BUY/SELL" if score is high enough.
        # Let's enforce the threshold HERE in the decision logic.
        
        # Enforce Dynamic Threshold
        # We need to retrieve the modifier again (or pass it down).
        # Context is in 'meta_data' (merged from agi_context)
        
        dyn_threshold = 60.0 + meta_data.get('threshold_modifier', 0.0)
        
        if final_decision != "WAIT" and final_conf < dyn_threshold:
             # logger.info(f"REGIME VETO: Score {final_conf:.1f} < Dynamic Threshold {dyn_threshold:.1f}")
             final_decision = "WAIT"
             meta_data['veto_reason'] = f"Regime Threshold ({dyn_threshold:.1f})"
        
        # --- SMART CIVIL WAR RESOLUTION ---
        # If we are gridlocked ("WAIT") but there is significant volume/conviction in the room.
        if final_decision == "WAIT" and (buy_strength + sell_strength) > 300:
             # 1. Elite Agent Arbitration (Whales & Smart Money have 2x Vote)
             elite_balance = 0.0
             elite_agents = ["Whale_Swarm", "SmartMoney_Swarm", "Apex_Swarm", "Quantum_Grid_Swarm"]
             
             for t in thoughts:
                 if t.source in elite_agents:
                     if t.signal_type == "BUY": elite_balance += t.confidence
                     elif t.signal_type == "SELL": elite_balance -= t.confidence
             
             if elite_balance > 100.0:
                 final_decision = "BUY"
                 final_conf = 70.0 # Enough to trigger, not enough for Wolf Pack
                 logger.info(f"CIVIL WAR RESOLVED: Elite Agents ({elite_balance:.0f}) broke the tie -> BUY")
                 meta_data['resolution'] = "ELITE_VOTE"
                 
             elif elite_balance < -100.0:
                 final_decision = "SELL"
                 final_conf = 70.0
                 logger.info(f"CIVIL WAR RESOLVED: Elite Agents ({elite_balance:.0f}) broke the tie -> SELL")
                 meta_data['resolution'] = "ELITE_VOTE"
        
        if final_decision not in allowed_actions: final_decision = "WAIT"

        if final_decision == "BUY":
            meta_data['winning_source'] = top_buy_agent
        elif final_decision == "SELL":
            meta_data['winning_source'] = top_sell_agent
            
        meta_data['attention_weights'] = attn_weights.tolist()
        return final_decision, final_conf, meta_data



    def _attempt_deep_reasoning_fix(self, decision: str, score: float, notes: List[str], meta_data: Dict) -> Tuple[str, float, str]:
        """
        Attempt to rescue a decision that failed Metacognition (Poor Reasoning).
        We check if the 'Unreasonable' decision is actually 'Intuitive' or 'Reflexive'.
        """
        # 0. SAFETY LOCK: Never rescue if the Swarm detected Exhaustion.
        if meta_data.get('exhaustion'):
             return "WAIT", 0.0, f"Rescue Blocked by Exhaustion ({meta_data.get('exhaustion_reason')})"

        # 1. Elite Authority Check
        # If the signal originated from an Elite Agent, we trust it more than the general consensus.
        # We need to check which agents contributed to this decision.
        # This is hard because 'thoughts' are not passed here, but we can infer from meta_data or simply trust the Elite Agents blindly for now if we had access.
        # FIX: We don't have 'thoughts' here easily. 
        # But we can check 'resolution' in meta_data. If it was an ELITE_VOTE, we should have passed!
        
        if meta_data.get('resolution') == 'ELITE_VOTE':
            return decision, 70.0, "Elite Vote Authority overrides Poor Reasoning"
            
        # 2. Volatility Reflex Exception
        # If Volatility is Extreme, standard logic (spread checks, session checks) might be too slow.
        # We assume High Entropy = High Volatility context.
        entropy = 0.5
        if 'market_metrics' in self.short_term_memory:
             entropy = self.short_term_memory['market_metrics'].get('entropy', 0.5)
             
        if entropy > 0.8: # Chaos Mode
             return decision, 60.0, "High Entropy Reflex: Speed > Logic"
             
        # 3. Trend Aikido Exception
        # If we just performed a Trend Aikido flip, it MIGHT look like poor reasoning (counter-intuitive).
        # But we know it's a valid strategy.
        if meta_data.get('aikido_flip'):
             return decision, 65.0, "Trend Aikido Strategy is Valid by Design"

        return "WAIT", 0.0, "Rescue Failed"
