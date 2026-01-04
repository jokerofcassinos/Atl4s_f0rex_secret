
import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Tuple
import pandas as pd
from .consciousness_bus import ConsciousnessBus
from .interfaces import SwarmSignal, SubconsciousUnit

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
# Phase 111-114 imports
from analysis.swarm.news_swarm import NewsSwarm
# Phase 115
from analysis.swarm.riemann_swarm import RiemannSwarm
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm
from analysis.swarm.gravity_swarm import GravitySwarm
from analysis.swarm.gravity_swarm import GravitySwarm
from analysis.swarm.holographic_swarm import HolographicSwarm
from analysis.swarm.holographic_swarm import HolographicSwarm
from analysis.swarm.strange_attractor_swarm import StrangeAttractorSwarm
from analysis.swarm.strange_attractor_swarm import StrangeAttractorSwarm
from analysis.swarm.causal_swarm import CausalSwarm
from analysis.swarm.causal_swarm import CausalSwarm
from analysis.swarm.bayesian_swarm import BayesianSwarm
from analysis.swarm.topological_swarm import TopologicalSwarm
from analysis.swarm.akashic_swarm import AkashicSwarm
from analysis.swarm.gaussian_process_swarm import GaussianProcessSwarm
from analysis.swarm.thermodynamic_swarm import ThermodynamicSwarm
from analysis.swarm.hyperdimensional_swarm import HyperdimensionalSwarm
from analysis.swarm.mirror_swarm import MirrorSwarm
from analysis.swarm.interference_swarm import InterferenceSwarm
from analysis.swarm.kinematic_swarm import KinematicSwarm
from analysis.swarm.singularity_swarm import SingularitySwarm
from analysis.swarm.hyperdimensional_swarm import HyperdimensionalSwarm
from analysis.swarm.vortex_swarm import VortexSwarm
from analysis.swarm.dna_swarm import DNASwarm
from analysis.swarm.schrodinger_swarm import SchrodingerSwarm
from analysis.swarm.antimatter_swarm import AntimatterSwarm
from analysis.swarm.heisenberg_swarm import HeisenbergSwarm
from analysis.swarm.navier_stokes_swarm import NavierStokesSwarm
from analysis.swarm.dark_matter_swarm import DarkMatterSwarm
from analysis.swarm.holographic_swarm import HolographicSwarm
from analysis.swarm.superluminal_swarm import SuperluminalSwarm
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm
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
from analysis.swarm.heisenberg_swarm import HeisenbergSwarm
from analysis.swarm.riemann_swarm import RiemannSwarm
from analysis.swarm.penrose_swarm import PenroseSwarm
from analysis.swarm.godel_swarm import GodelSwarm

# Meta-Cognition
from analysis.swarm.attention_swarm import AttentionSwarm

# Systems
from core.neuroplasticity import NeuroplasticityLoop
from core.holographic_memory import HolographicMemory
from core.genetics import EvolutionChamber
from core.quantum_bus import QuantumBus
from risk.great_filter import GreatFilter
from core.mcts_planner import MCTSPlanner

logger = logging.getLogger("SwarmOrchestrator")

class SwarmOrchestrator:
    """
    The Conscious Mind (Cortex).
    Coordinates the Subconscious Swarms and makes the Final Executive Decision.
    """
    def __init__(self):
        self.bus = ConsciousnessBus() # Internal
        self.quantum = QuantumBus(mode="PUB") # External/Inter-Process
        self.active_agents: List[SubconsciousUnit] = []
        self.attention = AttentionSwarm() # Meta-Cognition
        self.state = "BOOTING"
        
        # Memory & Adaptation
        self.short_term_memory: Dict[str, Any] = {}
        self.holographic_memory = HolographicMemory()
        self.neuroplasticity = NeuroplasticityLoop()
        
        # Evolution
        self.evolution = EvolutionChamber(population_size=10)
        self.current_dna = self.evolution.best_dna
        
        # Global Alpha (Confidence Threshold)
        self.alpha_threshold = 60.0 # Lowered to allow execution in complex environments 
        
        # System 2 Reasoning
        self.grandmaster = MCTSPlanner() 

    async def initialize_swarm(self):
        logger.info("--- GENESIS PROTOCOL: AWAKENING SWARM ---")
        
        # Initialize Swarm Modules
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
        self.active_agents.append(HolographicSwarm())
        self.active_agents.append(HolographicSwarm())
        self.active_agents.append(StrangeAttractorSwarm())
        self.active_agents.append(StrangeAttractorSwarm())
        self.active_agents.append(CausalSwarm())
        self.active_agents.append(CausalSwarm())
        self.active_agents.append(BayesianSwarm())
        self.active_agents.append(TopologicalSwarm())
        self.active_agents.append(AkashicSwarm())
        self.active_agents.append(GaussianProcessSwarm())
        self.active_agents.append(ThermodynamicSwarm())
        self.active_agents.append(HyperdimensionalSwarm())
        self.active_agents.append(MirrorSwarm())
        self.active_agents.append(InterferenceSwarm())
        self.active_agents.append(KinematicSwarm())
        self.active_agents.append(SingularitySwarm())
        self.active_agents.append(HyperdimensionalSwarm())
        self.active_agents.append(VortexSwarm())
        self.active_agents.append(DNASwarm())
        self.active_agents.append(SchrodingerSwarm())
        self.active_agents.append(AntimatterSwarm())
        self.active_agents.append(HeisenbergSwarm())
        self.active_agents.append(NavierStokesSwarm())
        self.active_agents.append(DarkMatterSwarm())
        self.active_agents.append(HolographicSwarm())
        self.active_agents.append(SuperluminalSwarm())
        self.active_agents.append(EventHorizonSwarm())
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
        self.active_agents.append(HeisenbergSwarm())
        self.active_agents.append(RiemannSwarm())
        self.active_agents.append(PenroseSwarm())
        self.active_agents.append(GameSwarm())
        self.active_agents.append(GodelSwarm())
        self.active_agents.append(AkashicSwarm())
        self.active_agents.append(WeaverSwarm(self.bus))
        self.active_agents.append(LaplaceSwarm())
        self.active_agents.append(PhysarumSwarm())
        self.active_agents.append(SingularitySwarm())
        self.active_agents.append(NeuralLace(self.bus))
        self.active_agents.append(CausalSwarm())
        self.active_agents.append(NewsSwarm())
        self.active_agents.append(ZeroPointSwarm())
        self.active_agents.append(ActiveInferenceSwarm())
        self.active_agents.append(HawkingSwarm())
        self.active_agents.append(RiemannSwarm())
        
        logger.info(f"Swarm Initialized with {len(self.active_agents)} Cognitive Sub-Units.")
        
        self.state = "CONSCIOUS"
        logger.info("State: CONSCIOUS. Waiting for Input.")

    async def process_tick(self, tick: Dict[str, Any], data_map: Dict[str, pd.DataFrame], config: Dict = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        The Main Loop of Consciousness.
        """
        # --- PHASE 47: DATA NORMALIZATION (SCALE FIX) ---
        # Detect if we are using a Proxy Data Source (e.g., BTC-USD @ 90k)
        # while trading a bridged asset (e.g., BTCXAU @ 20.00).
        # scaling_factor = 1.0
        
        df_m5 = data_map.get('M5') # Renamed from df to df_m5 for consistency with original on_tick
        if df_m5 is not None and not df_m5.empty:
            last_close = df_m5['close'].iloc[-1]
            current_price = tick.get('bid', last_close)
            
            # If mismatch is massive (e.g. > 10x diff)
            if current_price > 0 and (last_close / current_price) > 5.0:
                 scale_factor = last_close / current_price
                 # Create a SHADOW TICK for Analysis Only
                 # We do NOT modify the original tick passed to execution, 
                 # but for the Swarms, they need to see consistent numbers.
                 tick = tick.copy()
                 tick['bid'] *= scale_factor
                 tick['ask'] *= scale_factor
                 tick['scale_factor'] = scale_factor # Notify agents if they care
                 # logger.info(f"SCALE FIX: {scale_factor:.2f}x (Tick {current_price} -> {tick['bid']:.2f})")
        # ----------------------------------------------)
        if random.random() < 0.05 and df_m5 is not None: 
             self.current_dna = self.evolution.evolve(df_m5)

        # Context for Swarms
        df_h1 = data_map.get('H1') # Assuming H1 data is available in data_map
        context = {
            'tick': tick,
            'data_map': data_map,
            'df_m1': data_map.get('M1'), # Keep M1 if it was intended to be there
            'df_m5': df_m5,
            'df_h1': df_h1,
            'market_state': {}, # To be populated by specialized agents
            'config': config if config else {},
            'dna': self.current_dna 
        }
        
        # Collect Signals 
        current_signals = {}
        
        # Collect Signals (and Log them)
        signal_summary = []
        laminar_mode = False
        self.alpha_threshold = 60.0 # Reset to Default
        
        for agent in self.active_agents:
            thought = await agent.process(context)
            if thought:
                self.bus.register_thought(thought)
                current_signals[agent.name] = thought.signal_type
                signal_summary.append(f"{agent.name}={thought.signal_type}({thought.confidence:.0f}%)")
                
                # SAFETY OVERRIDE (Phase 64 Fixing Phase 58 failure)
                # If Harvester signals EXIT (Emergency or Strategic), we obey IMMEDIATELY.
                if agent.name == "HarvesterSwarm" and "EXIT" in thought.signal_type:
                     logger.warning(f"🚨 HARVESTER OVERRIDE: {thought.signal_type} ({thought.meta_data.get('reason')})")
                     return (thought.signal_type, thought.confidence, thought.meta_data)
                
                # Phase 54: Laminar Flow Engine
                if agent.name == "Chaos_Swarm":
                    entropy = thought.meta_data.get('entropy', 1.0)
                    lyapunov = thought.meta_data.get('lyapunov', 0.0)
                    
                    # Laminar Flow Condition (Low Entropy + Stability)
                    if entropy < 0.6 and lyapunov < 0.01:
                        # Market is Smooth. Lower the Shields.
                        self.alpha_threshold = 45.0
                        laminar_mode = True
                        self.short_term_memory['market_state']['regime'] = 'LAMINAR'
                        logger.info(f"🌊 LAMINAR FLOW DETECTED (Ent={entropy:.2f}): Shields Lowered to {self.alpha_threshold}% for Rapid Fire.")
                    
                    # Turbulent Condition
                    elif entropy > 0.85:
                        self.alpha_threshold = 75.0
                        logger.info(f"🌪️ TURBULENCE DETECTED (Ent={entropy:.2f}): Shields Raised to {self.alpha_threshold}%.")
        
        if signal_summary:
            logger.info(f"SWARM VOTES: {', '.join(signal_summary)}")
        
        decision = await self.synthesize_thoughts()
        
        # Store Signals in Short Term Memory 
        self.short_term_memory['last_signals'] = current_signals
        
        # Phase 32: Singularity Feedback
        curr_price = tick.get('bid', 0)
        curr_time = tick.get('time', 0)
        await self._process_synaptic_feedback(curr_price, curr_time)
        
        # Return the original decision (calculated at line 206)
        # Previously we called synthesize_thoughts() again here, which cleared the bus and returned WAIT!
        if decision:
            return decision 
        return ("WAIT", 0.0, {})

    async def _process_synaptic_feedback(self, current_price: float, current_time: int):
        """
        Phase 32: The Singularity.
        Back-propagates market reality to the Swarm Agents.
        """
        # 1. Store Current Prediction Snapshot
        # We need the signals generated THIS tick. They are in short_term_memory['last_signals']
        last_signals = self.short_term_memory.get('last_signals', {})
        if not last_signals: return
        
        # Snapshot structure: (Timestamp, PriceAtTime, Signals)
        # We store it in a deque or list. 
        # But we need persistence. For now, in-memory list is fine for session learning.
        if not hasattr(self, 'synaptic_buffer'):
            self.synaptic_buffer = []

        snapshot = {
            'time': current_time,
            'price': current_price,
            'signals': last_signals.copy()
        }
        self.synaptic_buffer.append(snapshot)
        
        # 2. Evaluate Old Snapshots (Delayed Reward)
        # We check snapshots from ~5 minutes ago (300000 ms)
        # If we check too soon, noise. Too late, irrelevant.
        # Let's assess prediction horizon: 5 minutes.
        
        evaluated_indices = []
        horizon_ms = 300000 # 5 mins
        
        for i, snap in enumerate(self.synaptic_buffer):
            age = current_time - snap['time']
            
            if age > horizon_ms:
                # Time to Judge!
                past_price = snap['price']
                delta = current_price - past_price
                
                # Register Outcome
                # If Delta > 0, BUY was correct.
                # If Delta < 0, SELL was correct.
                # We ignore small noise? No, every tick counts in theory, but let's threshold.
                if abs(delta) > 0.5: # Min movement to care (50 points on gold?) No, 0.5 USD.
                    self.neuroplasticity.register_outcome(snap['signals'], delta)
                    evaluated_indices.append(i)
            
            elif age < horizon_ms:
                # Since buffer is chronological, we can break early
                # But assume appending is strictly ordered.
                break
                
        # 3. Prune Buffer
        if evaluated_indices:
            # removing indices from list in reverse to avoid shifting issues?
            # actually we can just slice.
            # Only remove the evaluated ones.
            last_idx = evaluated_indices[-1]
            self.synaptic_buffer = self.synaptic_buffer[last_idx+1:]
        
        # Limit buffer size just in case
        if len(self.synaptic_buffer) > 1000:
             self.synaptic_buffer.pop(0)


    async def synthesize_thoughts(self) -> str:
        """
        The 'Free Will' Simulation.
        Aggregates thousands of micro-signals into a coherent Action.
        """
        thoughts = self.bus.get_recent_thoughts()
        if not thoughts: return None
        
        # --- PHASE 114: THE BALANCE OF POWER (SOFT ARBITRATION) ---
        allowed_actions = ["BUY", "SELL", "WAIT", "EXIT_ALL", "EXIT_LONG", "EXIT_SHORT", "VETO"]
        macro_bias_reason = ""
        strong_macro = False
        macro_score = 0.0
        
        for t in thoughts:
            if t.source == "News_Swarm" and hasattr(t, 'meta_data'):
                macro_score = t.meta_data.get('bias', 0.0)
                # Phase 114: Adaptive Thresholds
                # We only "Hard Block" if News is EXTREME (> 0.60)
                if macro_score > 0.60: 
                    allowed_actions = ["BUY", "WAIT", "EXIT_SHORT", "EXIT_ALL"]
                    macro_bias_reason = f"Extreme Bullish News ({macro_score:.2f})"
                    strong_macro = True
                elif macro_score < -0.60:
                    allowed_actions = ["SELL", "WAIT", "EXIT_LONG", "EXIT_ALL"]
                    macro_bias_reason = f"Extreme Bearish News ({macro_score:.2f})"
                    strong_macro = True

        # 1. Physics Reality Check (Phase 65: Clash of Gods)
        physics_decision = self._resolve_physics_conflict(thoughts)
        if physics_decision:
             action = physics_decision[0]
             conf = physics_decision[1]
             
             # PHASE 112: THE PROPHET CLAUSE (Infinite Certainty)
             if conf > 98.0:
                 logger.info(f"PHYSICS SINGULARITY ({conf}%): Overriding everything.")
                 return (action, conf, physics_decision[2])

             # PHASE 114: HIGH CONFIDENCE OVERRIDE
             # If Physics is Strong (> 85%) and News is not Extreme (< 0.6), Physics Wins.
             if conf > 85.0 and not strong_macro:
                   logger.info(f"PHYSICS DOMINANCE: {action} ({conf}%) > Medium News ({macro_score}). Executing.")
                   return (action, conf, physics_decision[2])
                   
             if action not in allowed_actions:
                  logger.warning(f"PHYSICS BLOCKED BY MACRO: {action} denied due to {macro_bias_reason}.")
                  if action == "SELL" and "EXIT_SHORT" in allowed_actions:
                      return ("EXIT_SHORT", 99.0, {'reason': f"Macro Override ({macro_bias_reason})"})
                  if action == "BUY" and "EXIT_LONG" in allowed_actions:
                      return ("EXIT_LONG", 99.0, {'reason': f"Macro Override ({macro_bias_reason})"})
                  return ("WAIT", 0.0, {})
             
             logger.info(f"PHYSICS OVERRIDE: {action} (God Mode Active)")
             return (action, conf, physics_decision[2])

        # 2. Attention Mechanism (The Transformer)
        attention_signal = self.attention.synthesize(thoughts)
        
        final_decision = None
        final_metadata = {}
        
        # If Transformer is extremely confident (above Alpha Threshold), it leads
        if attention_signal and attention_signal.confidence > self.alpha_threshold:
             logger.info(f"TRANSFORMER OVERRIDE: {attention_signal.signal_type} (Conf: {attention_signal.confidence:.1f} > {self.alpha_threshold})")
             # Prioritize Attention, but allow Veto checks
             final_decision = attention_signal.signal_type
             final_score = attention_signal.confidence # Use its confidence
             final_metadata = attention_signal.meta_data
        else:
            # Fallback to Weighted Voting (Neuroplasticity)
            score_buy = 0
            score_sell = 0
            total_weight = 0
            
            weights = self.neuroplasticity.get_dynamic_weights()
            
            # active_signals_list = []

            for t in thoughts:
                w = weights.get(t.source, 1.0)
                if t.signal_type == "BUY": 
                    score_buy += t.confidence * w
                    # active_signals_list.append(t)
                elif t.signal_type == "SELL": 
                    score_sell += t.confidence * w
                    # active_signals_list.append(t)
                total_weight += w
            
            # Simple Synthesis
            final_decision = "WAIT"
            final_score = 0
            
            if total_weight == 0: total_weight = 1.0

            if score_buy > score_sell:
                raw_score = score_buy # Normalize? Just check threshold
                if raw_score > (total_weight * 25): # Rough threshold heuristic
                     final_decision = "BUY"
                     final_score = 80 # Conceptual
            elif score_sell > score_buy:
                raw_score = score_sell
                if raw_score > (total_weight * 25):
                     final_decision = "SELL"
                     final_score = 80

        # MCTS Verification (The Grandmaster Check)
        # --- NEW HANDLING FOR VETO/EXIT_ALL FROM SUB-AGENTS (Harvester) ---
        # Harvester issues EXIT_ALL. We must catch it.
        # Check thoughts for URGENT Signals
        # --- FILTER ENFORCEMENT ---
        if final_decision not in allowed_actions:
             if "EXIT" not in final_decision:
                  logger.warning(f"CONSENSUS BLOCKED BY MACRO: {final_decision} -> Denied.")
                  if final_decision == "SELL" and "EXIT_SHORT" in allowed_actions:
                      final_decision = "EXIT_SHORT"
                      final_metadata['reason'] = f"Macro Override: {macro_bias_reason}"
                      final_score = 99.0
                  elif final_decision == "BUY" and "EXIT_LONG" in allowed_actions:
                      final_decision = "EXIT_LONG"
                      final_metadata['reason'] = f"Macro Override: {macro_bias_reason}"
                      final_score = 99.0
                  else:
                      final_decision = "WAIT"

        for t in thoughts:
             if t.signal_type == "EXIT_ALL": 
                 logger.warning(f"CRITICAL EXIT TRIGGERED BY: {t.source} (Reason: {t.meta_data.get('reason', 'None')})")
                 return ("EXIT_ALL", t.confidence, t.meta_data)
             if t.signal_type == "VETO":
                 if final_decision != "WAIT" and "EXIT" not in final_decision:
                     logger.info(f"VETO enforced by {t.source} ({t.meta_data.get('reason','')})")
                     final_decision = "WAIT"

        return (final_decision, final_score, final_metadata)

        # 2. VETO CHECK (Absolute Safety)
        # We always check VetoSwarm last
        for t in thoughts:
             if t.signal_type == 'VETO':
                 logger.warning(f"VETO TRIGGERED: {t.meta_data.get('reason')}")
                 return ("WAIT", 0.0, {})


        # 3. Holographic Recall
        if final_decision != "WAIT":
             # Placeholder for vector construction
             current_vector = [final_score] 
             mem_outcome, mem_conf = self.holographic_memory.recall(current_vector)
             if mem_conf > 0.8 and mem_outcome < 0:
                 logger.warning("DEJA VU: Negative Outcome Memory. Aborting.")
                 return ("WAIT", 0.0, {})

        # 4. System 2 Check (The Grandmaster)
        # MCTS Verification
        if final_decision in ["BUY", "SELL"]:
             # Construct minimal state for simulation
             # We use relative pricing (0 base) for simplicity
             sim_state = {
                 'price': 0, 
                 'entry': 0,
                 'side': final_decision,
                 'pnl': 0,
                 'volatility': 1.0
             }
             
             # Trend Bias Injection
             # If Score is 80 (High Conf), Bias is 0.4 (Strong Drift)
             # If score is 0, bias is 0.
             bias = 0.0
             if final_decision == "BUY":
                 bias = (final_score / 100.0) * 0.5
             elif final_decision == "SELL":
                 bias = -(final_score / 100.0) * 0.5
                 
             # Ask Grandmaster
             plan = self.grandmaster.search(sim_state, trend_bias=bias)
             
             if plan == "CLOSE": 
                 logger.warning(f"GRANDMASTER VETO: MCTS predicts negative EV for {final_decision}. Signaling Directional EXIT.")
                 # Don't force exit all, just don't enter *new* trade.
                 # Actually, if we are in a trade, this suggests getting out.
                 # But sticking to "Vetoing the Entry" for now.
                 return ("WAIT", 0.0, {})

        return (final_decision, final_score, final_metadata)

    def _resolve_physics_conflict(self, thoughts):
        """
        Phase 65: The Clash of Gods.
        Physics Agents have Dominion over Statistical Agents.
        Hierarchy: Singularity (2.5) > Hyperdimensional (2.2) > Vortex (2.1) > Kinematic (2.0)
        """
        physics_agents = ['SingularitySwarm', 'HyperdimensionalSwarm', 'KinematicSwarm', 'VortexSwarm', 'Schrodinger_Newton_Swarm']
        # Tightened Threshold: Physics must be nearly certain (> 95%) to override Consensus.
        high_command = [t for t in thoughts if t.source in physics_agents and t.confidence > 95.0]
        
        if not high_command: return None
        
        # Sort by Weight/Authority
        # We need a map or just hardcode hierarchy
        hierarchy = {
            'SingularitySwarm': 3, 
            'HyperdimensionalSwarm': 2.5,
            'Schrodinger_Newton_Swarm': 2.4, # High Authority due to Gravity Model
            'VortexSwarm': 2.2,
            'KinematicSwarm': 1
        }
        
        # Find highest rank signal
        best_signal = None
        best_rank = -1
        
        for t in high_command:
            rank = hierarchy.get(t.source, 0)
            if rank > best_rank:
                best_rank = rank
                best_signal = t
            elif rank == best_rank:
                 # Tie-break by confidence
                 if t.confidence > best_signal.confidence:
                     best_signal = t
                     
        if best_signal:
             return (best_signal.signal_type, best_signal.confidence, best_signal.meta_data)
             
        return None

    async def run(self):
        """Main Life Loop"""
        await self.initialize_swarm()
        while True:
            await asyncio.sleep(0.001) # Keep Event Loop Alive
