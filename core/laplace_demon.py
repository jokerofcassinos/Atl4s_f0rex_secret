import asyncio
import pandas as pd
import numpy as np
import ta 
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- MODULE IMPORTS (RESTORING THE BUGATTI ENGINE) ---

# 1. Signals (New Architecture)
from signals.timing import QuarterlyTheory, M8FibonacciSystem
from signals.structure import SMCAnalyzer
from analysis.smart_money import Liquidator # New Strategy Integration
from analysis.nano_structure import NanoBlockAnalyzer # Nano Algo Protocol
from signals.momentum import MomentumAnalyzer, ToxicFlowDetector
from analysis.vortex_math import VortexMath # Tesla 3-6-9
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm # Seek & Destroy
# from signals.volatility import VolatilityAnalyzer # Conflict: Use Legacy Guard or New?
# Let's use the New Volatility Analyzer if it's cleaner, but Legacy Logic relies on specific methods.
# For safety, we will import legacy classes from analysis for logic consistency.

# 2. Legacy Analysis Modules (The V1 Core)
from analysis.trend_architect import TrendArchitect
from analysis.sniper import Sniper
from analysis.quant import Quant
from analysis.volatility import VolatilityGuard
from analysis.patterns import PatternRecon
from analysis.market_cycle import MarketCycle
from analysis.supply_demand import SupplyDemand
from analysis.divergence import DivergenceHunter
from analysis.kinematics import Kinematics
from analysis.microstructure import MicroStructure
from analysis.fractal_vision import FractalVision
from analysis.math_core import MathCore
from analysis.quantum_core import QuantumCore
from analysis.cortex_memory import CortexMemory
from analysis.prediction_engine import PredictionEngine
from analysis.wavelet_core import WaveletCore
from analysis.topology_engine import TopologyEngine
from analysis.game_theory import GameTheoryCore
from analysis.chaos_engine import ChaosEngine
from analysis.supply_chain import SupplyChainGraph
# Omitting Eye modules here to avoid circular dependencies (Laplace replaces Consensus)
# We will integrate their logic directly or use simplified versions if needed.
# For now, we focus on the ANALYTICAL engines, not the decision agents (which are swarms now).

# 3. LEGION ELITE (The Swarm)
from analysis.swarm.vortex_swarm import VortexSwarm
from analysis.swarm.time_knife_swarm import TimeKnifeSwarm
from analysis.swarm.physarum_swarm import PhysarumSwarm
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm
from analysis.swarm.overlord_swarm import OverlordSwarm

# 4. AGI Helpers
from core.agi.big_beluga.snr_matrix import SNRMatrix
from core.agi.microstructure.flux_heatmap import FluxHeatmap
from core.agi.active_inference.free_energy import FreeEnergyMinimizer
from core.agi.neural_oracle import NeuralOracle
from core.ml.deep_models import ModelHub

from core.swarm_orchestrator import SwarmOrchestrator
from analysis.consensus import ConsensusEngine
from analysis.deep_cognition import DeepCognition

# Swarm Dependencies
from core.consciousness_bus import ConsciousnessBus
from core.genetics import EvolutionEngine
from core.neuroplasticity import NeuroPlasticityEngine
from core.transformer_lite import TransformerLite
from core.mcts_planner import MCTSPlanner

logger = logging.getLogger("LaplaceDemon")

class SignalStrength(Enum):
    VETO = -999
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4
    DIVINE = 5

@dataclass
class LaplacePrediction:
    execute: bool
    direction: str
    confidence: float
    strength: SignalStrength
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    risk_pct: float = 2.0
    lot_multiplier: float = 1.0 # Added for Omega Sniper
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    primary_signal: str = ""
    logic_vector: float = 0.0
    details: Dict = field(default_factory=dict)

class LaplaceDemonCore:
    """
    The True Succesor to ConsensusEngine (V1).
    Merges the 'Bugatti' analytical engine of V1 with the Async Swarm architecture of V2.
    """
    ALLOWED_HOURS_SET = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17} 

    def __init__(self, symbol: str = "GBPUSD", contrarian_mode: bool = False):
        self.symbol = symbol
        self.daily_trades = {'date': None, 'count': 0}
        
        # --- 1. THE BUGATTI ENGINE (Legacy V1 Core) ---
        # Refactored: We now wrap the dedicated ConsensusEngine instead of duplication
        logger.info("Initializing Consensus Engine (V1 Logic)...")
        self.consensus = ConsensusEngine()
        
        # --- 2. DEEP COGNITION (Subconscious) ---
        logger.info("Initializing Deep Cognition...")
        self.deep_brain = DeepCognition()
        
        # --- 3. SWARM INTELLIGENCE (V2 Swarm) ---
        logger.info("Initializing AGI Components & Swarm...")
        # Dependency Injection for Swarm
        self.bus = ConsciousnessBus()
        self.evo = EvolutionEngine()
        self.neuro = NeuroPlasticityEngine()
        self.attention = TransformerLite(embed_dim=64, head_dim=8)
        self.planner = MCTSPlanner()
        
        self.swarm = SwarmOrchestrator(
            bus=self.bus, 
            evolution=self.evo, 
            neuroplasticity=self.neuro, 
            attention=self.attention, 
            grandmaster=self.planner
        )

        # --- 4. GAME THEORY (Nash Equilibrium) ---
        self.game = GameTheoryCore()

        # --- 5. NEURAL ORACLE (Tier 4) ---
        self.neural_oracle = NeuralOracle()
        if self.neural_oracle._loaded:
             logger.info("Neural Oracle Model: LOADED (Ready to Filter)")
        else:
             logger.warning("Neural Oracle Model: NOT LOADED (Bypass Mode)")
        
        # --- 5. TIER 2 SIGNALS (V2 Enhancements) ---
        self.smc_signal = SMCAnalyzer()
        self.momentum = MomentumAnalyzer()
        self.quarterly = QuarterlyTheory()
        self.m8_system = M8FibonacciSystem()
        self.toxic_flow = ToxicFlowDetector()
        
        # 1.5 The Liquidator (Session Sweeps)
        self.liquidator = Liquidator()

        # 1.6 Nano Algo Protocol (Nano Blocks & Icebergs)
        self.nano_analyzer = NanoBlockAnalyzer()
        
        # 1.7 Vortex & Event Horizon (Esoteric Layer)
        self.vortex = VortexMath()
        self.event_horizon = EventHorizonSwarm()
        
        # 2. Analysis Engines6. TIER 3 AGI CORE ---
        self.snr_matrix = SNRMatrix()
        self.flux_heatmap = FluxHeatmap()

        # --- 7. TIER 4 DEEP LEARNING (Neural Oracle) ---
        self.neural_hub = ModelHub()
        
        # --- 8. CONFIGURATION ---
        self.params = {
            'threshold': 20.0  # Base threshold for trade execution
        }

    def record_trade(self, ticket: int, pnl: float, details: Dict):
        """
        Phase 4: Feedback Loop Interface.
        Records closed trade outcome to the training dataset.
        """
        try:
            # Save to CSV for offline training
            import csv
            import os
            
            file_path = "data/training/live_trades.csv"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Prepare row
            row = {
                'timestamp': datetime.now().isoformat(),
                'ticket': ticket,
                'pnl': pnl,
                'direction': details.get('direction', 'UNKNOWN'),
                'entry_price': details.get('entry_price', 0.0),
                'exit_price': details.get('exit_price', 0.0),
                'setup': details.get('source', 'UNKNOWN'),
                'confidence': details.get('confidence', 0.0)
            }
            
            # Write header if new
            file_exists = os.path.exists(file_path)
            
            with open(file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                
            logger.info(f"🧠 LEARNING: Recorded Trade {ticket} (PnL: ${pnl:.2f}) to training set.")
            
            logger.info(f"🧠 LEARNING: Recorded Trade {ticket} (PnL: ${pnl:.2f}) to training set.")
            
        except Exception as e:
            logger.error(f"Failed to record trade for learning: {e}")

    def record_experience(self, ticket: int, outcome: float, features: List[float]):
        """
        Phase 2 Integration: Cortex Memory & Akashic Records Interaction.
        Called by ExecutionEngine upon trade closure.
        outcome: 1.0 (Win), -1.0 (Loss)
        """
        try:
            # 1. Update Short Term Memory (Cortex)
            if self.consensus and self.consensus.cortex:
                 self.consensus.cortex.store_experience(features, outcome)
                 logger.info(f"🧠 CORTEX: Stored new memory (Feature Vector Len: {len(features)})")
                 
            # 2. Update Long Term Memory (Akashic) via Swarm (if available)
            # (Akashic usually trains offline, but we can log unique events here)
            pass 
        except Exception as e:
            logger.error(f"Failed to record experience: {e}")

    async def initialize(self):
        """
        Awakens the Swarm. Must be called after instantiation and before loop.
        """
        logger.info("[INIT] AWAKENING SWARM AGENTS...")
        if self.swarm:
            await self.swarm.initialize_swarm()
            count = len(self.swarm.active_agents)
            logger.info(f"[SWARM] ONLINE: Discovered {count} active agents.")
            
            # Re-inject bridge now that agents exist
            if hasattr(self, 'bridge') and self.bridge:
                self.swarm.inject_bridge(self.bridge)
                logger.info(f"[BRIDGE] PROPAGATED: {count} Agents armed.")

    def set_bridge(self, bridge):
        """
        Phase 1 Integration: Connects the Brain (Laplace) to the Hand (ZmqBridge).
        Allows Swarm Agents to execute advanced orders (Pruning, Harvesting).
        """
        self.bridge = bridge
        if self.swarm:
            self.swarm.inject_bridge(bridge)
        logger.info("[BRIDGE] CONNECTED: Laplace Demon now has physical execution authority.")

    async def analyze(self,
                df_m1: pd.DataFrame,
                df_m5: pd.DataFrame,
                df_h1: pd.DataFrame,
                df_h4: pd.DataFrame,
                df_d1: pd.DataFrame = None, # Added D1
                current_time: datetime = None,
                current_price: float = None,
                **kwargs) -> LaplacePrediction:
        
        if current_time is None: current_time = datetime.now()
        if current_price is None and df_m5 is not None: current_price = df_m5['close'].iloc[-1]
        
        # 1. PERCEPTION (FAST) -> Microstructure (Tier 3 Flux)
        flux_metrics = {}
        details = {} # Initialize details container
        if 'tick' in kwargs:
             flux_metrics = self.flux_heatmap.update(kwargs['tick'])
             self.nano_analyzer.on_tick(kwargs['tick']) 
             self.event_horizon.nano.on_tick(kwargs['tick']) # Sync Swarm Nano
             
        # ... (Legacy Map Update) ...
        
        # 0.A Vortex 3-6-9 Analysis
        vortex_res = self.vortex.analyze(df_m5, current_time) # M8 simulation on M5 for now or mock
        details['Vortex'] = vortex_res
        
        # 0.B Event Horizon Swarm (Seek & Destroy)
        swarm_context = {'df_m5': df_m5, 'tick': kwargs.get('tick')}
        eh_signal = await self.event_horizon.process(swarm_context)
        details['EventHorizon'] = eh_signal.meta_data if eh_signal else {}
        
        # Data Map for V1 Engines
        data_map = {
            'M5': df_m5,
            'H1': df_h1,
            'H4': df_h4,
            'D1': df_d1
        }
        
        # 1. SWARM ORCHESTRATOR (Async)
        # We start the swarm first as it might need time (or runs parallel)
        
        # Construct Inputs
        tick_dict = {'bid': current_price, 'ask': current_price, 'time': current_time, 'symbol': self.symbol}
        swarm_data_map = {'M1': df_m1, 'M5': df_m5, 'H1': df_h1, 'H4': df_h4}
        
        # Call Swarm
        swarm_decision, swarm_score, swarm_meta = await self.swarm.process_tick(
            tick=tick_dict, 
            data_map=swarm_data_map
        )
        
        # Pack into Legion Intel dict for synthesis transparency
        legion_intel = swarm_meta if swarm_meta else {}
        legion_intel['swarm_decision'] = swarm_decision
        legion_intel['swarm_score'] = swarm_score
        
        # 3. LEGACY CONSENSUS (V1 Logic) - The "Bugatti Engine"
        # We run this to get the base "Template" for the trade
        legacy_result, legacy_details = self._run_legacy_consensus(data_map)
        details.update(legacy_details) # Merge legacy details into master details
        
        # --- TIER 2 SIGNALS (SMC / M8 / Toxic) ---
        details['SMC'] = self.smc_signal.analyze(df_m5, current_price)
        details['ToxicFlow'] = self.toxic_flow.detect_compression(df_m5)
        details['Flux'] = flux_metrics
        
        # Liquidator Analysis
        if df_h1 is not None:
             self.liquidator.update_session_levels(df_h1)
        
        liq_signal = self.liquidator.check_sweep(df_m5, current_price)
        details['Liquidator'] = liq_signal

        # 6.4 Nano Algo Protocol
        # Analyze Order Flow Blocks (Red/Green)
        nano_res = self.nano_analyzer.analyze(current_price)
        details['nano_blocks'] = nano_res
        
        # Neural Oracle (Tier 4) - Hypothetical Check
        try:
             # We query the Oracle for both directions to gauge market bias
             # Using 50% confidence as neutral baseline
             prob_buy = self.neural_oracle.predict_win_probability(df_m5, "BUY", 50.0)
             prob_sell = self.neural_oracle.predict_win_probability(df_m5, "SELL", 50.0)
             details['Neural_Oracle_Hypothesis'] = {"BUY": f"{prob_buy:.2f}", "SELL": f"{prob_sell:.2f}"}
        except Exception as e:
             # Soft fail to not crash analysis
             details['Neural_Oracle_Hypothesis'] = "N/A" # str(e)
        
        # SNR Matrix (Tier 3)
        if df_h1 is not None and len(df_h1) > 50:
            self.snr_matrix.scan_structure(df_h1)
            details['SNR'] = self.snr_matrix.get_nearest_levels(current_price)
        
        # M8 Fibonacci (Resample M1)
        if df_m1 is not None and not df_m1.empty:
             try:
                 df_m8 = df_m1.resample('8min').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
                 if not df_m8.empty:
                     details['M8'] = self.m8_system.analyze(df_m8, current_time)
             except Exception as e:
                 logger.warning(f"M8 Analysis Failed: {e}")

        # 3. SYNTHESIS
        prediction = self._synthesize_master_decision(legacy_result, legion_intel, details, current_price, df_m5, df_m1=df_m1)
        
        # Log the internal state so the user sees the "Mind" at work
        swarm_dec = legion_intel.get('swarm_decision', 'WAIT')
        swarm_scr = legion_intel.get('swarm_score', 0.0)
        
        # Determine status flag
        status_flag = ""
        if not prediction.execute and (prediction.primary_signal and prediction.primary_signal != "Monitoring"):
             status_flag = " [VETOED]"
        elif prediction.execute:
             status_flag = " [ACTIVE]"

        log_msg = (f"[THINKING] | Price: {current_price} | "
                   f"Legacy: {legacy_result['score']:.1f} | "
                   f"Swarm: {swarm_dec} ({swarm_scr:.1f}%) | "
                   f"Setup: {prediction.primary_signal if prediction.primary_signal else 'Monitoring'}{status_flag}")
        
        if prediction.primary_signal and prediction.primary_signal != "Monitoring":
             # If vetoed, show the original logic score to show potential
             if not prediction.execute:
                  log_msg += f" (Logic: {prediction.logic_vector:.1f})"
             else:
                  log_msg += f" (Conf: {prediction.confidence:.1f}%)"
                    
        # Log at INFO level so it shows up in console
        logger.info(log_msg)

        if prediction.execute:
            logger.info(f"[SIGNAL GENERATED] {prediction.direction} | Conf: {prediction.confidence:.1f}% | Reasons: {prediction.reasons}")
            
        return prediction

    def _run_legacy_consensus(self, data_map):
        """
        Executes the heavy V1 math engines via the dedicated ConsensusEngine.
        Returns the 'Preliminary Vector' and detailed analysis.
        """
        # Delegate to the Single Source of Truth
        decision, legacy_vector, details = self.consensus.deliberate(data_map)
        
        # Adapt output format to match what _synthesize expects
        # ConsensusEngine returns: (decision_str, vector_float, details_dict)
        # We need: ({'score': vector, 'setup': setup_type}, details)
        
        setup_type = None
        if isinstance(details, dict):
            # Try to find reason in Vectors or general details
            vecs = details.get('Vectors', {})
            setup_type = vecs.get('Reason') if isinstance(vecs, dict) else None
            
            if not setup_type:
                 setup_type = details.get('setup_name') # Just in case I patch consensus later
        
        return {'score': legacy_vector, 'setup': setup_type}, details

    def _synthesize_master_decision(self, legacy_result, legion_intel, details, price, df_m5, df_m1=None):
        """
        Merges the Legacy Vector with Swarm Intelligence.
        THE HYBRID ENGINE: Bugatti V1 + AGI Swarm V2
        """
        score = legacy_result['score']
        setup = legacy_result['setup']
        
        reasons = []
        vetoes = []
        
        # Extract Chaos variables for Vetos
        chaos_data = details.get('Chaos', {})
        if not isinstance(chaos_data, dict): chaos_data = {}
        lyapunov = chaos_data.get('lyapunov', 0.0)
        
        math_data = details.get('Math', {})
        if not isinstance(math_data, dict): math_data = {}
        entropy = math_data.get('entropy', 0.0)
        
        if setup:
            reasons.append(f"Legacy V1 Setup: {setup} (Base Score: {score:.1f})")
            
        # --- SWARM INTELLIGENCE INTEGRATION ---
        # 1. Specific Agent Confirmations (Micro-Boosts)
        knife = legion_intel.get('knife') if isinstance(legion_intel, dict) else None
        horizon = legion_intel.get('horizon') if isinstance(legion_intel, dict) else None
        
        # (REMOVED: LEGION_KNIFE_SCALP - Deprecated)

        # EventHorizon (Gravity/Reversion) - Original Logic
        if horizon and horizon.signal_type in ["BUY", "SELL"]:
             if "SINGULARITY" in horizon.meta_data.get('reason', ''):
                 reasons.append(f"EventHorizon Singularity: {horizon.signal_type}")
                 score += 100 if horizon.signal_type == "BUY" else -100
                 setup = "SINGULARITY_REVERSION"

        # (REMOVED: SWARM_369 - Deprecated)

        # 2. Global Swarm Consensus (The "Legion" Vote)
        # We now respect the 88-agent SwarmOrchestrator
        swarm_decision = legion_intel.get('swarm_decision', 'WAIT')
        swarm_score = legion_intel.get('swarm_score', 0.0)
        swarm_meta = legion_intel.get('swarm_meta', {})
        
        if swarm_decision in ["BUY", "SELL"]:
            # Fusion Logic: Combine Legacy and Swarm Scores
            swarm_vector = swarm_score if swarm_decision == "BUY" else -swarm_score
            
            # Weighted Average? Or Boost?
            # We want Legacy to lead, but Swarm to support.
            
            # Case A: Agreement (Symbiosis)
            if (score > 0 and swarm_vector > 0) or (score < 0 and swarm_vector < 0):
                score += (swarm_vector * 0.5) # Add 50% of Swarm strength
                reasons.append(f"Global Swarm Agreement: {swarm_decision} (Conf {swarm_score:.1f}%)")
                
            # Case B: Significant Disagreement (Civil War)
            elif (score > 20 and swarm_vector < -20) or (score < -20 and swarm_vector > 20):
                # Legacy thinks Buy, Swarm thinks Sell (or vice versa)
                # We punish the confidence but DO NOT VETO immediately (unless Swarm is Elite)
                score *= 0.5 # Halve the confidence
                reasons.append(f"⚠️ Swarm Disagreement ({swarm_decision}). Reducing Confidence.")
                
                # VETO if Swarm is emphatic (Wolf Pack / Elite)
                if swarm_score > 85:
                    vetoes.append(f"Critical Swarm Veto ({swarm_decision} {swarm_score:.1f}%)")
                    
            # Case C: Legacy Indecisive, Swarm Strong (Swarm Lead)
            elif abs(score) < 20 and swarm_score > 75:
                # ✅ PHASE 12 FIX: Chaos Veto on Swarm (Neutral)
                # Swarm is good, but dies in Chaos. Block if Chaos > 0.8.
                chaos_veto_swarm = False
                if lyapunov > 0.8 or entropy > 2.5:
                     chaos_veto_swarm = True
                     reasons.append(f"Swarm Initiative VETOED by Chaos")

                if not chaos_veto_swarm:
                    # If Legacy is sleeping but Swarm sees something clear
                    score += (swarm_vector * 0.8) # Adopt Swarm view
                    reasons.append(f"Swarm Initiative: {swarm_decision} (Conf {swarm_score:.1f}%)")
                    # ✅ BUG FIX: 'setup' is rarely None/Empty string, usually "Neutral". Check explicit list.
                    if not setup or setup in ["Neutral", "WAIT", "None"]:
                         setup = "SWARM_INITIATIVE"

                # ✅ PHASE 15 FIX: Smart Money Veto for Neutral/Swarm Trades
                # Neutral trades are weak. We MUST NOT trade them against Smart Money or Trend.
                if not chaos_veto_swarm: # Only check if not already vetoed
                    cycle_res = details.get('Cycle', ('NEUTRAL', 0))
                    cycle_phase = cycle_res[0] if isinstance(cycle_res, tuple) else str(cycle_res)
                    
                    # Check direction of Swarm Score
                    swarm_dir = 1 if swarm_vector > 0 else -1 if swarm_vector < 0 else 0
                    
                    if swarm_dir == -1: # SELL
                        if "MANIPULATION_BUY" in str(cycle_phase) or "EXPANSION_BUY" in str(cycle_phase):
                            chaos_veto_swarm = True # Reuse veto flag to silence it
                            reasons.append(f"Swarm VETOED by Smart Money (Bullish Phase: {cycle_phase})")
                            score = 0 # Kill the score
                    elif swarm_dir == 1: # BUY
                        if "MANIPULATION_SELL" in str(cycle_phase) or "EXPANSION_SELL" in str(cycle_phase):
                            chaos_veto_swarm = True
                            reasons.append(f"Swarm VETOED by Smart Money (Bearish Phase: {cycle_phase})")
                            score = 0

        # --- TIER 2 ENHANCEMENTS (SMC / M8) ---
        smc_res = details.get('SMC', {})
        if not isinstance(smc_res, dict): smc_res = {}
        
        m8_res = details.get('M8', {})
        if not isinstance(m8_res, dict): m8_res = {}
        
        toxic_flow = details.get('ToxicFlow', False)
        liq_signal = details.get('Liquidator')
        
        nano_res = details.get('nano_blocks', {})
        if not isinstance(nano_res, dict): nano_res = {}
        
        eh_meta = details.get('EventHorizon', {})
        if not isinstance(eh_meta, dict): eh_meta = {}

        # --- STRUCTURE FLIP (Breaker Strategy) ---
        # "Aggressive system to capture the bullish/bearish" - User
        # Logic: We now read the 'breaker_signal' directly from the SMC module.
        # This guarantees we trade EXACTLY when the Veto fires.
        breaker_sig = smc_res.get('breaker_signal')
        
        if breaker_sig and isinstance(breaker_sig, dict):
             # The Veto fired in structure.py and generated this signal
             b_dir = breaker_sig.get('direction')
             b_conf = breaker_sig.get('confidence', 0)
             
             if b_dir == "BUY":
                  # Check for BULL TRAP (Breaking Resistance but Momentum is BEARISH)
                  # If Momentum is effectively SELLING (score < -50) while we break UP, it's likely a liquidity grab.
                  if score < -50:
                       reasons.append(f"BULL TRAP DETECTED: Fake Breakout vs Momentum ({score:.1f})")
                       score = -100 # Flip to SELL
                       setup = "BULL_TRAP_SNIPER"
                  else:
                       reasons.append(f"STRUCTURE RIDER: Breaking Bearish OB in Bullish Trend! (Aggressive Buy)")
                       score = 100 # Override
                       setup = "STRUCTURE_TREND_RIDER"

             elif b_dir == "SELL":
                  # Check for BEAR TRAP (Breaking Support but Momentum is BULLISH)
                  # If Momentum is effectively BUYING (score > 50) while we break DOWN, it's likely a Spring.
                  if score > 50:
                       reasons.append(f"BEAR TRAP DETECTED: Fake Breakdown vs Momentum ({score:.1f})")
                       score = 100 # Flip to BUY
                       setup = "BEAR_TRAP_SNIPER"
                  else:
                       reasons.append(f"STRUCTURE RIDER: Breaking Bullish OB in Bearish Trend! (Aggressive Sell)")
                       score = -100 # Override
                       setup = "STRUCTURE_TREND_RIDER"

        # 0. The Liquidator (New 90% Setup) - WITH KINEMATICS CHECK
        # Get Kinematics direction for counter-trend veto
        k_data = details.get('Kinematics', {})
        if not isinstance(k_data, dict): k_data = {}
        if not isinstance(k_data, dict): k_data = {} # Safety wrap
        
        k_dir = k_data.get('direction', 0)  # +1=UP, -1=DOWN
        k_angle = abs(k_data.get('angle', 0))
        
        if liq_signal:
             # KINEMATICS VETO: Don't fight strong acceleration
             liq_vetoed = False
             if "BUY" in liq_signal:
                  # If Kinematics is strongly ACCELERATING DOWN, don't buy
                  if k_dir == -1 and k_angle > 45:
                       reasons.append(f"LIQUIDATOR VETOED: Kinematics DOWN ({k_angle}°)")
                       liq_vetoed = True
                  if not liq_vetoed:
                       reasons.append(f"LIQUIDATOR: Sweep of {liq_signal} + Kinematics Match")
                       score = -100 if "BUY" in liq_signal else 100 # Invert signal (Sweep Buy = Sell)
                       setup = "LIQUIDATOR_SWEEP"

             elif "SELL" in liq_signal:
                  # If Kinematics is strongly ACCELERATING UP, don't sell
                  if k_dir == 1 and k_angle > 45:
                       reasons.append(f"LIQUIDATOR VETOED: Kinematics UP ({k_angle}°)")
                       liq_vetoed = True
                  if not liq_vetoed:
                       reasons.append(f"THE LIQUIDATOR: {liq_signal}")
                       if score > 0: score = -50
                       else: score -= 50
                       if not setup: setup = "LIQUIDATOR_SWEEP"

        # --- NEW SETUPS (AGI BRAINSTORM 1.1) ---
        
        # 1. THE GOLDEN COIL (M8 Fibonacci Retracement)
        # Logic: 
        #   - Resample M1 -> M8
        #   - Identify Trend (High > PrevHigh)
        #   - Check Pullback to 50-61.8% Zone of Last M8 Impulse
        if df_m1 is not None and len(df_m1) >= 16: # Need at least 2x M8 candles
             try:
                  # Resample M1 to M8
                  m8_res = df_m1.resample('8T').agg({
                       'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                  }).dropna()
                  
                  if len(m8_res) >= 3:
                       last_m8 = m8_res.iloc[-2] # Completed candle
                       prev_m8 = m8_res.iloc[-3] # Previous reference
                       curr_price = df_m1['close'].iloc[-1]
                       
                       # Bullish Coil
                       # 1. Strong Bullish Impulse (Body > 50% of Range)
                       m8_range = last_m8['high'] - last_m8['low']
                       m8_body = abs(last_m8['close'] - last_m8['open'])
                       is_impulse = (m8_body / m8_range) > 0.5 if m8_range > 0 else False
                       
                       if last_m8['close'] > last_m8['open'] and is_impulse:
                            # 2. Fibonacci Zone (50% - 61.8%)
                            fib_50 = last_m8['low'] + (0.50 * m8_range)
                            fib_618 = last_m8['low'] + (0.382 * m8_range) # Deep pullback (inverted fib logic relative to low)
                            # Wait, standard pullback: High - (Range * 0.5) or Low + (Range * 0.5)?
                            # Retracement from High down to Low.
                            # Zone is between 61.8% Retracement Price and 50% Retracement Price.
                            # Price at 50% = Low + 0.5 * Range
                            # Price at 61.8% Retracement = High - 0.618 * Range = Low + 0.382 * Range
                            
                            zone_top = last_m8['low'] + (0.5 * m8_range)
                            zone_bottom = last_m8['low'] + (0.382 * m8_range)
                            
                            if zone_bottom <= curr_price <= zone_top:
                                 # 3. Check for Rejection (Micro-Wick in M1?)
                                 # For now, just level tap + Trend Confluence
                                 if (score > 0 or details.get('Trend', {}).get('H1_hurst', 0) > 0.6) and 150 > abs(score): 
                                      reasons.append(f"THE GOLDEN COIL: M8 Pullback to Golden Zone ({curr_price:.5f})")
                                      setup = "GOLDEN_COIL_M8"
                                      score = 150 # High Priority
                                      
                       # Bearish Coil
                       elif last_m8['close'] < last_m8['open'] and is_impulse:
                            # Retracement UP from Low.
                            # Price at 50% = High - 0.5 * Range
                            # Price at 61.8% Retracement = Low + 0.618 * Range = High - 0.382 * Range
                            
                            zone_bottom = last_m8['high'] - (0.5 * m8_range)
                            zone_top = last_m8['high'] - (0.382 * m8_range)
                            
                            if zone_bottom <= curr_price <= zone_top:
                                 if (score < 0 or details.get('Trend', {}).get('H1_hurst', 0) > 0.6) and 150 > abs(score):
                                      reasons.append(f"THE GOLDEN COIL: M8 Pullback to Golden Zone ({curr_price:.5f})")
                                      setup = "GOLDEN_COIL_M8"
                                      score = -150 # High Priority
                                      
             except Exception as e:
                  logger.warning(f"Golden Coil Error: {e}")
        
        # 2. THE VOID FILLER (FVG Reversion)
        # Logic: Touch and Reject FVG Zone
        active_fvgs = details.get('SMC', {}).get('active_fvgs', [])
        if active_fvgs and len(active_fvgs) > 0:
             curr_candle = df_m5.iloc[-1]
             c_high = curr_candle['high']
             c_low = curr_candle['low']
             c_close = curr_candle['close']
             c_open = curr_candle['open']
             
             for fvg in active_fvgs:
                  # ... (Access logic same) ...
                  
                  try:
                       fvg_top = getattr(fvg, 'top', 0)
                       fvg_bot = getattr(fvg, 'bottom', 0)
                       fvg_type = getattr(fvg, 'type', 'NEUTRAL')
                       
                       # Bullish FVG
                       if fvg_type == "BULLISH":
                            if c_low <= fvg_top and c_close > fvg_bot: 
                                 if c_close < fvg_bot: continue
                                 
                                 # SANITY CHECK: Don't fade a Strong Consensus TREND
                                 # If Consensus says SELL (>50) and we try to BUY, BLOCK IT.
                                 if score < -50:
                                      continue

                                 # PRIORITY CHECK: Only apply if Score 130 > Current Score
                                 if 130 > abs(score):
                                     reasons.append(f"THE VOID FILLER: Bullish FVG Rejection @ {fvg_top:.5f}")
                                     setup = "VOID_FILLER_FVG"
                                     score = 130
                                     break
                                 
                       # Bearish FVG
                       elif fvg_type == "BEARISH":
                            if c_high >= fvg_bot and c_close < fvg_top:
                                 if c_close > fvg_top: continue
                                 
                                 # SANITY CHECK: Don't fade a Strong Consensus TREND
                                 # If Consensus says BUY (>50) and we try to SELL, BLOCK IT.
                                 if score > 50:
                                      continue

                                 # PRIORITY CHECK
                                 if 130 > abs(score):
                                     reasons.append(f"THE VOID FILLER: Bearish FVG Rejection @ {fvg_bot:.5f}")
                                     setup = "VOID_FILLER_FVG"
                                     score = -130
                                     break
                  except Exception as e:
                       pass 
                       
        # 3. VOLATILITY SQUEEZE HUNTER (TTM Squeeze Logic)
        if df_m5 is not None and len(df_m5) >= 20: 
             try:
                  # 1. Calculate Bollinger Bands (20, 2.0)
                  df_sqz = df_m5.iloc[-25:].copy() 
                  df_sqz['ma20'] = df_sqz['close'].rolling(window=20).mean()
                  df_sqz['std'] = df_sqz['close'].rolling(window=20).std()
                  df_sqz['bb_upper'] = df_sqz['ma20'] + (2.0 * df_sqz['std'])
                  df_sqz['bb_lower'] = df_sqz['ma20'] - (2.0 * df_sqz['std'])
                  
                  # 2. Calculate Keltner Channels (20, 1.5 ATR)
                  df_sqz['tr0'] = abs(df_sqz['high'] - df_sqz['low'])
                  df_sqz['tr1'] = abs(df_sqz['high'] - df_sqz['close'].shift())
                  df_sqz['tr2'] = abs(df_sqz['low'] - df_sqz['close'].shift())
                  df_sqz['tr'] = df_sqz[['tr0', 'tr1', 'tr2']].max(axis=1)
                  df_sqz['atr'] = df_sqz['tr'].rolling(window=20).mean()
                  
                  df_sqz['kc_upper'] = df_sqz['ma20'] + (1.5 * df_sqz['atr'])
                  df_sqz['kc_lower'] = df_sqz['ma20'] - (1.5 * df_sqz['atr'])
                  
                  # 3. Check Squeeze Condition (Previous Candle)
                  prev_candle = df_sqz.iloc[-2]
                  squeeze_on = (prev_candle['bb_upper'] < prev_candle['kc_upper']) and \
                               (prev_candle['bb_lower'] > prev_candle['kc_lower'])
                               
                  # 4. Check Breakout (Current Candle)
                  curr_candle = df_sqz.iloc[-1]
                  
                  if squeeze_on:
                       # Breakout UP
                       if curr_candle['close'] > curr_candle['bb_upper']:
                            if 140 > abs(score): # PRIORITY CHECK
                                if score > -50: 
                                    reasons.append(f"SQUEEZE HUNTER: Volatility Expansion UP (Breakout)")
                                    setup = "VOLATILITY_SQUEEZE"
                                    score = 140
                                
                       # Breakout DOWN
                       elif curr_candle['close'] < curr_candle['bb_lower']:
                            if 140 > abs(score): # PRIORITY CHECK
                                if score < 50:
                                    reasons.append(f"SQUEEZE HUNTER: Volatility Expansion DOWN (Breakout)")
                                    setup = "VOLATILITY_SQUEEZE"
                                    score = -140
                                
             except Exception as e:
                  pass # Math errors


        
        # ----------------------------------------

        
        # A. Sell at Red Block (Resistance)
        nano_sell = nano_res.get('algo_sell_detected')
        nano_buy = nano_res.get('algo_buy_detected')
        
        # --- NANO FALLBACK (SNR Bounce) - WITH FULL CONFLUENCE CHECK ---
        snr_data = details.get('SNR', {})
        if isinstance(snr_data, tuple): snr_data = {} # Safety fallback
        
        div_data_check = details.get('Divergence', {})
        if not isinstance(div_data_check, dict): div_data_check = {}
        div_type_check = div_data_check.get('type', '')
        
        pat_data_check = details.get('Patterns', {})
        if not isinstance(pat_data_check, dict): pat_data_check = {}
        pat_name_check = pat_data_check.get('pattern', '')

        # Fix for SupplyDemand Tuple Crash (LION)
        sd_data = details.get('SupplyDemand', {})
        if not isinstance(sd_data, dict): sd_data = {}
        
        # Fix for Quant Tuple Crash (LION)
        quant_res = details.get('Quant', {})
        if not isinstance(quant_res, dict): quant_res = {}
        quant_z = quant_res.get('z_score', 0)

        if setup == "LIQUIDATOR_SWEEP":
            # Safety: Check for Wall collision
            dist_res = snr_data.get('res_dist', 999) if isinstance(snr_data, dict) else 999
            dist_sup = snr_data.get('sup_dist', 999) if isinstance(snr_data, dict) else 999
            
            # If we are BUYING, ensure no Resistance immediately above (< 2 pips)
            if score > 0 and dist_res < 0.0002:
                 reasons.append(f"LIQUIDATOR VETO: Resistance Wall ahead ({dist_res:.5f})")
                 vetoes.append("Liquidity Trap: Resistance Wall")
                 score = 0
            # If we are SELLING, ensure no Support immediately below (< 2 pips)
            elif score < 0 and dist_sup < 0.0002:
                 reasons.append(f"LIQUIDATOR VETO: Support Wall ahead ({dist_sup:.5f})")
                 vetoes.append("Liquidity Trap: Support Wall")
                 score = 0
        
        if not nano_sell and not nano_buy and snr_data:
            dist_res = snr_data.get('res_dist', 999)
            dist_sup = snr_data.get('sup_dist', 999)
            
            # Near Resistance + Bearish → SELL, BUT check all bullish signals
            if dist_res < 0.0010 and score < 0:
                if 'bullish' in str(div_type_check).lower():
                    reasons.append("NANO FALLBACK VETOED: Bullish Divergence vs Resistance")
                elif pat_name_check in ["Hammer", "Bullish Engulfing", "Morning Star"]:
                    reasons.append(f"NANO FALLBACK VETOED: Bullish Pattern ({pat_name_check})")
                elif k_dir == 1 and k_angle > 45:
                    reasons.append(f"NANO FALLBACK VETOED: Kinematics UP ({k_angle}°)")
                else:
                    nano_sell = True
                    reasons.append("NANO FALLBACK: SNR Resistance Bounce")
            # Near Support + Bullish → BUY, BUT check all bearish signals
            elif dist_sup < 0.0010 and score > 0:
                if 'bearish' in str(div_type_check).lower():
                    reasons.append("NANO FALLBACK VETOED: Bearish Divergence vs Support")
                elif pat_name_check in ["Shooting Star", "Bearish Engulfing", "Evening Star"]:
                    reasons.append(f"NANO FALLBACK VETOED: Bearish Pattern ({pat_name_check})")
                elif k_dir == -1 and k_angle > 45:
                    reasons.append(f"NANO FALLBACK VETOED: Kinematics DOWN ({k_angle}°)")
                else:
                    nano_buy = True
                    reasons.append("NANO FALLBACK: SNR Support Bounce")

        # --- UNIVERSAL VETO CHECKS (Sniper, Div, Pattern) ---
        sniper_data = details.get('Sniper', {})
        if not isinstance(sniper_data, dict): sniper_data = {}
        
        div_data = details.get('Divergence', {})
        if not isinstance(div_data, dict): div_data = {}
        if not isinstance(div_data, dict): div_data = {}

        pat_data = details.get('Patterns', {})
        if not isinstance(pat_data, dict): pat_data = {}
        if not isinstance(pat_data, dict): pat_data = {}

        s_dir = sniper_data.get('dir', 0)
        s_score_v = sniper_data.get('score', 0)
        d_type = div_data.get('type', '')
        p_name = pat_data.get('pattern', '')

        # Trigger NANO SELL - WITH KINEMATICS CHECK
        if nano_sell:
             vetoed = False
             if s_dir == 1 and s_score_v > 50: vetoed = True
             if 'bullish' in str(d_type).lower(): vetoed = True
             if p_name in ["Hammer", "Bullish Engulfing"]: vetoed = True
             
             # KINEMATICS VETO: Don't sell into accelerating UP move
             if k_dir == 1 and k_angle > 45:
                 vetoed = True
                 reasons.append(f"NANO SELL VETOED: Kinematics UP ({k_angle}°)")
             
             if not vetoed:
                 reasons.append("NANO ALGO: Sell Block/Resistance Detected")
                 if score > 0: score = -50
                 else: score -= 30
                 if not setup: setup = "NANO_SCALE_IN"
             else:
                 if "VETOED" not in ' '.join(reasons): reasons.append("NANO SELL VETOED by Conflict")

        # Trigger NANO BUY
        if nano_buy:
             vetoed = False
             if s_dir == -1 and s_score_v > 50: vetoed = True
             if 'bearish' in str(d_type).lower(): vetoed = True
             if p_name in ["Shooting Star", "Bearish Engulfing"]: vetoed = True
             
             # ✅ PHASE 12 FIX: Nano Safeguards (Chaos + Wick)
             veto_nano = False
             if lyapunov > 0.8 or entropy > 2.5:
                  veto_nano = True
                  reasons.append("NANO VETOED by Chaos")
             
             # KINEMATICS VETO: Don't buy into accelerating DOWN move
             if not veto_nano and k_dir == -1 and k_angle > 45:
                 veto_nano = True
                 reasons.append(f"NANO BUY VETOED: Kinematics DOWN ({k_angle}°)")
             
             # Wick Check for Nano Buy (Catching Knife Protection)
             if not veto_nano:
                  if setup == "MOMENTUM_BREAKOUT":
                       micro = details.get('Micro', {})
                       if not isinstance(micro, dict): micro = {}
                       rejection = micro.get('rejection')
                       
                       if rejection != "BULLISH_REJECTION" and lyapunov > 0.5:
                            reasons.append(f"Chaos Veto: Momentum needs Micro Confirmation (Got {rejection})")
                            score = 0
                  else: # Original Nano Wick Check
                       micro = details.get('Micro', {})
                       if not isinstance(micro, dict): micro = {}
                       if micro.get('rejection') != "BULLISH_REJECTION" and lyapunov > 0.5:
                            veto_nano = True
                            reasons.append("NANO VETOED by No Wick (Knife)")

             if not vetoed and not veto_nano:
                 reasons.append("NANO ALGO: Buy Block/Support Detected")
                 if score < 0: score = 50
                 else: score += 30
                 if not setup or setup in ["Neutral", "WAIT", "None"]: setup = "NANO_SCALE_IN"
             else:
                 reasons.append("NANO BUY VETOED by Conflict/Chaos")

        # B. Event Horizon Swarm (Seek & Destroy)
        # ✅ PHASE 11 FIX: Chaos Veto on Event Horizon
        # Safety check for eh_meta
        if not isinstance(eh_meta, dict): eh_meta = {}

        if eh_meta.get('mode') == "SWARM_369":
             if "Buy" in eh_meta.get('reason', ''):
                  swarm_dir = 1
             elif "Sell" in eh_meta.get('reason', ''):
                  swarm_dir = -1
             else:
                  swarm_dir = 0
             
             # Chaos Veto - RELAXED (Agile Swarm)
             vetoed = False
             if lyapunov > 0.8 or entropy > 2.5:
                  reasons.append(f"EVENT HORIZON SILENCED by Chaos (Lyapunov {lyapunov:.2f})")
                  vetoed = True

             # Conflict checks - RELAXED (Phase 15)
             if not vetoed and swarm_dir == 1:
                 if s_dir == -1 and s_score_v > 90: vetoed = True  # Relaxed from 50
             elif not vetoed and swarm_dir == -1:
                 if s_dir == 1 and s_score_v > 90: vetoed = True  # Relaxed from 50
                 
             if not vetoed:
                 reasons.append(f"EVENT HORIZON: {eh_meta.get('reason', 'Swarm Attack')}")
                 setup = "SWARM_369"
                 if "Buy Detected" in eh_meta.get('reason', ''):
                      boost = 50
                      if toxic_flow: boost = 10
                      if score < 0: score = 60
                      else: score += boost
                 elif "Sell Detected" in eh_meta.get('reason', ''):
                      boost = 50
                      if toxic_flow: boost = 10
                      if score > 0: score = -60
                      else: score -= boost

        # --- DIVERGENCE CONFIRMATION (Trade #92 Fix) ---
        # If Setup is Counter-Trend (LION/REVERSION), we WANT Divergence.
        # But if Setup is MOMENTUM, Divergence AGAINST us is fatal.
        divergence_data = details.get('Divergence', {})
        if not isinstance(divergence_data, dict): divergence_data = {}
        
        div_type = divergence_data.get('type', '')
        
        if setup == "MOMENTUM_BREAKOUT":
             if score > 0 and "bearish" in str(div_type).lower():
                  # Buying into Bearish Div -> Bad idea
                  score *= 0.5
                  reasons.append(f"Caution: Momentum vs Bearish Div ({div_type})")
             elif score < 0 and "bullish" in str(div_type).lower():
                  score *= 0.5
                  reasons.append(f"Caution: Momentum vs Bullish Div ({div_type})")
        
        # --- PATTERN BOOST ---
        pattern_data = details.get('Patterns', {}) # Was 'Pattern' - likely typo if dict keys inconsistent
        if not isinstance(pattern_data, dict): pattern_data = {}
        
        pat_name = pattern_data.get('pattern')
        if pat_name:
             reasons.append(f"Pattern Detected: {pat_name}")
             score += 10 # Small boost

        # 7. Final Confidence Adjustment
        smc_zone = smc_res.get('zone', 'NEUTRAL')
        if smc_zone:
             if "BUY" in smc_zone and score > -10:
                  score += 25
                  reasons.append(f"SMC: Reacting off Bullish Order Block")
             elif "SELL" in smc_zone and score < 10:
                  score -= 25
                  reasons.append(f"SMC: Reacting off Bearish Order Block")
        
        # 2. M8 Fibonacci
        if isinstance(m8_res, dict) and m8_res.get('gate') == "GOLDEN":
             direction = m8_res.get('signal', 'NEUTRAL')
             m8_boost = 15 if direction == "BUY" else -15 if direction == "SELL" else 0
             score += m8_boost
             if m8_boost != 0:
                 reasons.append(f"M8: Golden Gate ({direction})")
             
        # 3. Toxic Flow Veto (DISABLED - Was killing profits)
        # Commented out to restore high-volume trades.
        
        is_neutral_setup = any("Setup: Neutral" in r for r in reasons)
        
        hard_veto = False # Persistent veto flag
        
        # DISABLED: Toxic Flow Neutral Veto
        # if toxic_flow and is_neutral_setup:
        #      reasons.append(f"Toxic Flow: VETOING Neutral Setup in Chop")
        #      vetoes.append("Toxic Flow: Neutral Setup Unsafe in Compression")
        #      execute = False
        #      hard_veto = True
              
        # Keep only score capping (mild, not veto)
        if toxic_flow:
             # For Snipers, we caution but DO NOT VETO immediately.
             reasons.append(f"Toxic Flow: Compression Detected (Capping Score to 80)")
             if abs(score) > 80:
                  score = 80.0 if score > 0 else -80.0
              
             # DISABLED: Regime Lock
             # momentum_strategies = ["MOMENTUM_BREAKOUT", "CONSENSUS_VOTE", "KINETIC_BOOM", "LION_PROTOCOL", "QUANTUM_HARMONY"]
             # if setup in momentum_strategies:
             #     reasons.append(f"REGIME LOCK: Vetoing {setup} in Toxic Flow (Momentum fails in Chop)")
             #     vetoes.append(f"Regime Lock: {setup} invalid in Chop")
             #     execute = False
             #     hard_veto = True
                 
             # DISABLED: Wick Filter
             # if setup == "REVERSION_SNIPER" and toxic_flow:
             #     micro = details.get('Micro', {})
             #     rejection = micro.get('rejection', 'NEUTRAL')
             #     if score > 0 and rejection == "BEARISH_REJECTION":
             #          reasons.append("Wick Filter: Vetoing Buy (Bearish Rejection Detected)")
             #          vetoes.append("Wick Filter: Knife Catching Prevented")
             #          execute = False
             #          hard_veto = True
             #     elif score < 0 and rejection == "BULLISH_REJECTION":
             #          reasons.append("Wick Filter: Vetoing Sell (Bullish Rejection Detected)")
             #          vetoes.append("Wick Filter: Rocket Catching Prevented")
             #          execute = False
             #          hard_veto = True
               
             # MATH RESTORE: Reverting to 80.0 Cap (99% Conf) per user request.
             if abs(score) > 80:
                  score = 80.0 if score > 0 else -80.0
              
             # DISABLED: Weak Signal in Chop Veto
             # if abs(score) < 40: 
             #     vetoes.append("Toxic Flow: Weak Signal in Chop")
             #     execute = False 
             #     hard_veto = True 

        # --- 4. DEEP FORENSIC ANALYSIS (DISABLED - Was killing profits) ---
        # COMMENTED OUT TO RESTORE PROFITABILITY. 
        
        # micro = details.get('Micro', {})
        # entropy = micro.get('entropy', 1.0)
        # ofi = micro.get('ofi', 0)
        
        # A. ENTROPY SHIELD (DISABLED)
        # if entropy < 0.6 and setup == "REVERSION_SNIPER":
        #      reasons.append(f"ENTROPY SHIELD: Vetoing Reversion in Structured Drift (Entropy {entropy:.2f})")
        #      vetoes.append("Entropy Shield: Iceberg Detected")
        #      execute = False
        #      hard_veto = True
              
        # B. OFI GUARD (DISABLED)
        # if score > 0 and ofi < -50:
        #      reasons.append(f"OFI GUARD: Vetoing BUY against Sell Pressure (OFI {ofi})")
        #      vetoes.append("OFI Guard: Tape Reading Sell Pressure")
        #      execute = False
        #      hard_veto = True
        # elif score < 0 and ofi > 50:
        #      reasons.append(f"OFI GUARD: Vetoing SELL against Buy Pressure (OFI {ofi})")
        #      vetoes.append("OFI Guard: Tape Reading Buy Pressure")
        #      execute = False
        #      hard_veto = True

        # 5. SNR Walls (Tier 3)
        snr_data = details.get('SNR', {})
        if not isinstance(snr_data, dict): snr_data = {}
        
        res_dist = snr_data.get('res_dist', 999.0)
        sup_dist = snr_data.get('sup_dist', 999.0)
             
        # VETO BUY into Resistance (< 5 Pips)
        if score > 0 and res_dist < 0.0005: 
            score *= 0.1 # Kill confidence
            vetoes.append(f"SNR Wall Veto: Resistance Ahead ({res_dist:.5f})")
            if not setup: setup = "BLOCKED_BY_WALL"
            
            score *= 0.1
            vetoes.append(f"SNR Wall Veto: Support Ahead ({sup_dist:.5f})")
            if not setup: setup = "BLOCKED_BY_WALL"

        # --- GLOBAL VETOES & COUNTER-STRATEGIES ---

        # 1. Chaos Veto for Neutral/Consensus (Trade #77 Fix)
        # Neutral trades (Consensus Vote) must NOT trade in Extreme Chaos.
        chaos_details = details.get('Chaos', {})
        chaos_val = 0
        if isinstance(chaos_details, dict):
            chaos_val = chaos_details.get('lyapunov', 0)
        elif isinstance(chaos_details, tuple) and len(chaos_details) > 0: chaos_val = chaos_details[0] # Assume Lyapunov is first
        
        if (setup == "Neutral" or "Consensus" in str(setup) or "WAIT" in str(setup)) and chaos_val > 0.9:
             reasons.append(f"GLOBAL VETO: Extreme Chaos ({chaos_val:.2f}) kills Neutral setup.")
             vetoes.append("Chaos Veto: Neutral in Storm")
             score = 0

        # 2. Cycle & Divergence Logic & SMART MONEY REVERSAL
        cycle_res = details.get('Cycle', ('NEUTRAL', 0))
        cycle_phase = cycle_res[0] if isinstance(cycle_res, tuple) else str(cycle_res)
        
        # Robust Divergence Extraction (Fixes Tuple crash)
        div_res = details.get('Divergence', {})
        div_type = ""
        if isinstance(div_res, dict): div_type = div_res.get('type', '')
        elif isinstance(div_res, tuple) and len(div_res) > 0: div_type = div_res[0]
        else: div_type = str(div_res)

        # Define atr_src early for SMR checks
        atr_src = details.get('Trend', {})
        if not isinstance(atr_src, dict): atr_src = {}

        if score > 0: # We want to BUY
            # --- BYPASS: STRUCTURE RIDER & SMR --- 
            if setup == "STRUCTURE_TREND_RIDER":
                 reasons.append(f"GLOBAL BYPASS: Structure Rider overrides Cycle Veto.")
            elif "MANIPULATION_SELL" in str(cycle_phase):
                # --- STRATEGY: SMART MONEY REVERSAL (Sell into the Trap) ---
                # Require Bearish Divergence (Confluence) to flip
                # TOXIC FILTER: Require ADX > 20 (Don't reverse in dead market)
                adx = atr_src.get('adx', 25.0) if isinstance(atr_src, dict) else 25.0
                
                if ("LION" in str(setup) or "BREAKOUT" in str(setup)) and "bearish" in str(div_type).lower():
                    if adx < 20: 
                         score = 0
                         vetoes.append("SMR VETO: ADX too low (<20) for Reversal")
                    else:
                         score = -100 # Maximum Conviction SELL
                         setup = "SMART_MONEY_REVERSAL"
                         reasons.append(f"GLOBAL FLIP: Smart Money Selling + Bearish Div! REVERSING TO SELL.")
                         if "Smart Money Veto" in vetoes: vetoes.remove("Smart Money Veto") 
                else:
                    if setup == "LION_BREAKOUT": # STRICT: LION dies instantly in Manipulation
                         vetoes.append(f"LION KILLER: Trading into Bearish Manipulation")
                    
                    reasons.append(f"GLOBAL VETO: Smart Money is Selling (Bearish Manipulation).")
                    vetoes.append("Smart Money Veto: Bearish Manipulation")
                    score = 0
            elif "EXPANSION_SELL" in str(cycle_phase):
                if setup != "STRUCTURE_TREND_RIDER": # Protect Rider
                    if setup == "LION_BREAKOUT": # STRICT: LION dies trading AGAINST Expansion
                         vetoes.append(f"LION KILLER: Buying against Expansion Sell")
                    
                    reasons.append(f"GLOBAL VETO: Market is Trending Down (Expansion Sell).")
                    vetoes.append("Trend Veto: Expansion Sell")
                    score = 0
            elif "bearish" in str(div_type).lower():
                if setup != "STRUCTURE_TREND_RIDER" and setup != "GOLDEN_COIL_M8" and setup != "VOID_FILLER_FVG" and setup != "VOLATILITY_SQUEEZE": # Protect Rider, Coil, Filler, Squeeze
                    reasons.append(f"GLOBAL VETO: Bearish Divergence ({div_type}) opposes BUY.")
                    vetoes.append("Divergence Veto: Bearish Conflict")
                    score = 0
                
        elif score < 0: # We want to SELL
            # --- BYPASS: STRUCTURE RIDER & SMR ---
            if setup == "STRUCTURE_TREND_RIDER":
                 reasons.append(f"GLOBAL BYPASS: Structure Rider overrides Cycle Veto.")
            elif "MANIPULATION_BUY" in str(cycle_phase):
                # --- STRATEGY: SMART MONEY REVERSAL (Buy into the Trap) ---
                # Require Bullish Divergence (Confluence) to flip
                # TOXIC FILTER: Require ADX > 20
                adx = atr_src.get('adx', 25.0) if isinstance(atr_src, dict) else 25.0
                
                if ("LION" in str(setup) or "BREAKOUT" in str(setup)) and "bullish" in str(div_type).lower():
                    if adx < 20:
                         score = 0
                         vetoes.append("SMR VETO: ADX too low (<20) for Reversal")
                    else:
                         score = 100 # Maximum Conviction BUY
                         setup = "SMART_MONEY_REVERSAL"
                         reasons.append(f"GLOBAL FLIP: Smart Money Buying + Bullish Div! REVERSING TO BUY.")
                         if "Smart Money Veto" in vetoes: vetoes.remove("Smart Money Veto")
                else:
                    if setup == "LION_BREAKOUT": # STRICT
                         vetoes.append(f"LION KILLER: Trading into Bullish Manipulation")

                    reasons.append(f"GLOBAL VETO: Smart Money is Buying (Bullish Manipulation).")
                    vetoes.append("Smart Money Veto: Bullish Manipulation")
                    score = 0
            elif "EXPANSION_BUY" in str(cycle_phase):
                if setup != "STRUCTURE_TREND_RIDER":
                    if setup == "LION_BREAKOUT": # STRICT
                         vetoes.append(f"LION KILLER: Selling against Expansion Buy")

                    reasons.append(f"GLOBAL VETO: Market is Trending Up (Expansion Buy).")
                    vetoes.append("Trend Veto: Expansion Buy")
                    score = 0
            
            
            # --- STRUCTURE VETO (Trade #46 Fix) ---
            # If SMC Structure says BULLISH, we should NOT sell unless it's a Smart Money Reversal.
            smc_struc = smc_res.get('structure', 'NEUTRAL')
            if setup != "SMART_MONEY_REVERSAL":
                 if smc_struc == "BULLISH" and setup != "STRUCTURE_TREND_RIDER" and setup != "GOLDEN_COIL_M8" and setup != "VOID_FILLER_FVG" and setup != "VOLATILITY_SQUEEZE": # Allow Rider, Coil, Filler, Squeeze
                      reasons.append(f"STRUCTURE VETO: Blocking SELL (Structure is {smc_struc})")
                      vetoes.append(f"Structure Veto: Cannot Sell in Bullish Structure")
                      score = 0
            
            # --- FINAL LION SANITY CHECK (SMC Alignment) ---
            if setup == "LION_BREAKOUT" and not score == 0:
                 smc_trend = smc_res.get('trend', 'NEUTRAL')
                 if (score > 0 and smc_trend == "BEARISH") or (score < 0 and smc_trend == "BULLISH"):
                      score = 0
                      vetoes.append(f"LION KILLER: SMC Trend Conflict ({smc_trend})")
            elif "bullish" in str(div_type).lower():
                 if setup != "STRUCTURE_TREND_RIDER" and setup != "GOLDEN_COIL_M8" and setup != "VOID_FILLER_FVG" and setup != "VOLATILITY_SQUEEZE":
                    reasons.append(f"GLOBAL VETO: Bullish Divergence ({div_type}) opposes SELL.")
                    vetoes.append("Divergence Veto: Bullish Conflict")
                    score = 0
                    
        # --- CHAOS VETO (Trade #75 & #78-105 Fix) ---
        # 1. Extreme Chaos Hard Veto (Safety First)
        if lyapunov > 0.9:
             reasons.append(f"EXTREME CHAOS VETO: Blocking ALL trades (Lyapunov {lyapunov:.2f} > 0.9)")
             vetoes.append(f"Chaos Veto: Market is too unpredictable (Lyapunov {lyapunov:.2f})")
             score = 0
             
        # 2. Reversion/Structure Block in High Chaos
        elif "REVERSION" in str(setup) or "SNIPER" in str(setup) or "RIDER" in str(setup):
             if lyapunov > 0.5:
                  reasons.append(f"CHAOS VETO: Blocking {setup} (Lyapunov {lyapunov:.2f} > 0.5)")
                  vetoes.append(f"Chaos Veto: Market too chaotic for {setup}")
                  score = 0  
                  
        # --- GLOBAL SNIPER CONFLICT VETO (Trade #109 Fix) ---
        # If Sniper says one direction (>50) but we're going opposite, BLOCK.
        # THRESHOLD LOWERED from 70 to 50 (Trade #106 had Sniper 57.2)
        sniper_data = details.get('Sniper', {})
        if not isinstance(sniper_data, dict): sniper_data = {}

        sniper_dir = sniper_data.get('dir', 0)
        sniper_score = sniper_data.get('score', 0)
        
        decision_dir = "BUY" if score > 0 else "SELL" if score < 0 else "WAIT"
        
        # Sniper dir: +1 = BUY, -1 = SELL (THRESHOLD: 50)
        if decision_dir == "BUY" and sniper_dir == -1 and sniper_score > 50:
            # RELAXED VETO: If we have > 90% Confidence (e.g. Swarm + Legacy), we ignore Sniper.
            if abs(score) >= 90 or setup == "GOLDEN_COIL_M8" or setup == "VOID_FILLER_FVG" or setup == "VOLATILITY_SQUEEZE":
                 reasons.append(f"SNIPER OVERRIDE: High Confidence ({score:.1f}) bypasses Sniper Veto.")
            else:
                reasons.append(f"SNIPER CONFLICT VETO: Blocking BUY (Sniper says SELL {sniper_score:.1f})")
                vetoes.append(f"Sniper Conflict: Sniper SELL {sniper_score:.0f}% opposes BUY")
                execute = False
                hard_veto = True
                logger.warning(f"SNIPER CONFLICT VETO: Blocking BUY (Sniper SELL {sniper_score:.1f})")
            
        elif decision_dir == "SELL" and sniper_dir == 1 and sniper_score > 50:
            if abs(score) >= 90 or setup == "GOLDEN_COIL_M8" or setup == "VOID_FILLER_FVG" or setup == "VOLATILITY_SQUEEZE":
                 reasons.append(f"SNIPER OVERRIDE: High Confidence ({score:.1f}) bypasses Sniper Veto.")
            else:
                reasons.append(f"SNIPER CONFLICT VETO: Blocking SELL (Sniper says BUY {sniper_score:.1f})")
                vetoes.append(f"Sniper Conflict: Sniper BUY {sniper_score:.0f}% opposes SELL")
                execute = False
                hard_veto = True
                logger.warning(f"SNIPER CONFLICT VETO: Blocking SELL (Sniper BUY {sniper_score:.1f})")

        # --- DIVERGENCE CONFLICT VETO (Trade #106 Fix) ---
        # If Bullish Divergence but decision is SELL, BLOCK (and vice versa)
        divergence_data = details.get('Divergence', {})
        if isinstance(divergence_data, dict):
            div_type = divergence_data.get('type', '')
            if decision_dir == "SELL" and 'bullish' in str(div_type).lower():
                if setup not in ["GOLDEN_COIL_M8", "VOID_FILLER_FVG", "VOLATILITY_SQUEEZE"]:
                     reasons.append("DIVERGENCE CONFLICT: Blocking SELL (Bullish Divergence)")
                     vetoes.append("Divergence Conflict: Bullish signal opposes SELL")
                     execute = False
                     hard_veto = True
                     logger.warning("DIVERGENCE CONFLICT VETO: Blocking SELL (Bullish Divergence detected)")
            elif decision_dir == "BUY" and 'bearish' in str(div_type).lower():
                if setup not in ["GOLDEN_COIL_M8", "VOID_FILLER_FVG", "VOLATILITY_SQUEEZE"]:
                     reasons.append("DIVERGENCE CONFLICT: Blocking BUY (Bearish Divergence)")
                     vetoes.append("Divergence Conflict: Bearish signal opposes BUY")
                     execute = False
                     hard_veto = True
                     logger.warning("DIVERGENCE CONFLICT VETO: Blocking BUY (Bearish Divergence detected)")

        # --- PATTERN CONFLICT VETO (Trade #106 Fix) ---
        # If Bullish Pattern (Hammer, Engulfing) but decision is SELL, BLOCK
        pattern_data = details.get('Pattern', '')
        bullish_patterns = ['hammer', 'bullish engulfing', 'morning star', 'doji']
        bearish_patterns = ['shooting star', 'bearish engulfing', 'evening star']
        
        pattern_lower = str(pattern_data).lower() if pattern_data else ''
        
        if decision_dir == "SELL" and any(bp in pattern_lower for bp in bullish_patterns):
            if setup not in ["GOLDEN_COIL_M8", "VOID_FILLER_FVG", "VOLATILITY_SQUEEZE"]:
                reasons.append(f"PATTERN CONFLICT: Blocking SELL (Bullish pattern: {pattern_data})")
                vetoes.append("Pattern Conflict: Bullish pattern opposes SELL")
                execute = False
                hard_veto = True
                logger.warning(f"PATTERN CONFLICT VETO: Blocking SELL (Bullish pattern: {pattern_data})")
        elif decision_dir == "BUY" and any(bp in pattern_lower for bp in bearish_patterns):
             if setup not in ["GOLDEN_COIL_M8", "VOID_FILLER_FVG", "VOLATILITY_SQUEEZE"]:
                reasons.append(f"PATTERN CONFLICT: Blocking BUY (Bearish pattern: {pattern_data})")
                vetoes.append("Pattern Conflict: Bearish pattern opposes BUY")
                execute = False
                hard_veto = True
                logger.warning(f"PATTERN CONFLICT VETO: Blocking BUY (Bearish pattern: {pattern_data})")

        # --- MONDAY CAUTION FILTER (Weekend Gap Fix) ---
        # Chart shows: Monday = 45% Win Rate, -$200 PnL (while other days = 90%+ WR)
        # The WeekendGapPredictor has errors, so Monday trades are unreliable.
        # SOLUTION: Block ALL trades on Monday to avoid weekend gap issues.
        if df_m5 is not None and len(df_m5) > 0:
            try:
                current_time = df_m5.index[-1] if hasattr(df_m5.index[-1], 'weekday') else None
                if current_time is not None and current_time.weekday() == 0:  # 0 = Monday
                    reasons.append("MONDAY CAUTION: Blocking trade (Weekend Gap Risk)")
                    vetoes.append("Monday Caution: Weekend gap instability")
                    execute = False
                    hard_veto = True
                    logger.warning("MONDAY CAUTION: Blocking trade due to weekend gap risk")
            except:
                pass  # If we can't check weekday, proceed normally

        # 4.5 KINETIC SHIELD (DISABLED - Was killing profits)
        # COMMENTED OUT TO RESTORE PROFITABILITY.
        # kin_data = details.get('Kinematics', {})
        # k_angle = kin_data.get('angle', 0)
        
        # if score > 0 and (180 <= k_angle <= 270):
        #      reasons.append(f"KINETIC SHIELD: Vetoing BUY against Downward Acceleration ({k_angle:.0f}°)")
        #      vetoes.append("Kinetic Shield: Buying into a Crash")
        #      execute = False
        #      hard_veto = True
        # elif score < 0 and (0 <= k_angle <= 90):
        #      reasons.append(f"KINETIC SHIELD: Vetoing SELL against Upward Acceleration ({k_angle:.0f}°)")
        #      vetoes.append("Kinetic Shield: Selling into a Rocket")
        #      execute = False
        #      hard_veto = True

        # 5. Neural Oracle (Tier 4 - Experimental)
        neural_pred = details.get('Neural')
        neural_conf = 0.0 # Initialize
        if neural_pred is not None:
             try:
                 # Flatten if nested list
                 import numpy as np # Import numpy for ndarray check
                 val = neural_pred[0] if isinstance(neural_pred, (list, np.ndarray)) else neural_pred
                 if isinstance(val, (list, np.ndarray)): val = val[0] # Double unpack
                 
                 # Final sanity check - if still not float, force 0
                 if isinstance(val, (float, int)):
                      neural_conf = float(val) / 100.0 if float(val) > 1.0 else float(val)
                 else:
                      neural_conf = 0.0
             except:
                 neural_conf = 0.0
        
        # Define threshold for this specific boost, distinct from the later win_prob threshold
        neural_boost_threshold = 0.70 # Example threshold for boosting score
        if neural_conf > neural_boost_threshold:
             reasons.append(f"EXTREME CONFIDENCE: Neural Network agrees ({neural_conf*100:.1f}%)")
             score += 50 # Boost

        # --- DECISION THRESHOLD ---
        decision_dir = "WAIT"
        execute = False
        confidence = 0.0
        
        if score > self.params['threshold']:
            decision_dir = "BUY"
            execute = True
            confidence = min(99.0, 50 + (score - self.params['threshold']))
        elif score < -self.params['threshold']:
            decision_dir = "SELL"
            execute = True
            confidence = min(99.0, 50 + (abs(score) - self.params['threshold']))
            
        # ENFORCE HARD VETO (Prevent Resets)
        if hard_veto:
            execute = False
            
        # --- GAME THEORY VETO (Nash) ---
        # Don't buy if far above Nash, Don't sell if far below
        nash_price = self.game.calculate_nash_equilibrium(df_m5)['equilibrium_price']
        # Dynamic SL based on Volatility
        atr_src = details.get('Trend', {})
        if not isinstance(atr_src, dict): atr_src = {}
        
        atr = atr_src.get('atr', 0.0010)
        
        # Default SL: 15 pips
        # Volatility Adjusted: 1.5 * ATR (approx 15-25 pips)
        sl_raw = max(0.0010, atr * 1.5)
        
        # SCALP CAP: For specific scalp setups, cap SL at 20 pips to preserve R:R
        if setup in ["VOID_FILLER_FVG", "GOLDEN_COIL_M8", "VOLATILITY_SQUEEZE", "LEGION_KNIFE_SCALP"]:
             sl_pips = min(sl_raw, 0.0020) # Cap at 20 pips
        else:
             sl_pips = sl_raw
        
        if execute:
            # Bypass Logic: Aggressive setups or High Confidence override the "Safe" Nash filter
            # "The Lion does not ask for permission."
            bypass_nash = (
                setup in ["KINETIC_BOOM", "LEGION_KNIFE_SCALP", "LION_PROTOCOL", "REVERSION_SNIPER", "MOMENTUM_BREAKOUT"] 
                or confidence >= 90.0
            )

            if not bypass_nash:
                if decision_dir == "BUY" and price > (nash_price + 3*atr):
                    # We are buying at a severe premium
                    execute = False
                    vetoes.append("Price far above Nash Equilibrium (Overextended)")
                elif decision_dir == "SELL" and price < (nash_price - 3*atr):
                    # We are selling at a discount
                    execute = False
                    vetoes.append("Price far below Nash Equilibrium (Oversold)")
            elif bypass_nash and ((decision_dir == "BUY" and price > nash_price + 3*atr) or (decision_dir == "SELL" and price < nash_price - 3*atr)):
                reasons.append(f"Nash Veto BYPASSED (Setup: {setup} | Conf: {confidence:.1f}%)")

        # --- TIER 4 NEURAL ORACLE OPTIMIZATION (Fase 6 CALIBRATION) ---
        if execute:
            win_prob = self.neural_oracle.predict_win_probability(df_m5, decision_dir, confidence)
            details['Neural_Win_Prob'] = float(win_prob)
            
            # CALIBRATION: Phase 6 (Round 3) - Target 70% WR
            # We enforce stricter thresholds: WinProb > 60% and High Bypass Bar (85.0)
            
            threshold = 0.60 # Increased from 0.40
            bypass_score = 85.0 # Restored to 85.0 (Allows Trade #96 Sniper to pass)
            
            # Special Case: If Toxic Flow is active, we demand stricter confirmation.
            # Trade #83 (76%) failed here.
            if toxic_flow:
                threshold = 0.85
                reasons.append(f"Toxic Flow Active: Raising Neural Threshold to {threshold:.2f}")
            
            # Special Case: If Toxic Flow is active, we ALREADY handled strictness above.
            # No need to double penalize here for Snipers.
            
            if win_prob < threshold and abs(score) < bypass_score:
                execute = False
                vetoes.append(f"Neural Oracle: Low Probability ({win_prob:.1%}) | Score: {abs(score):.1f}")
                reasons.append(f"Neural Filter: REJECTED ({win_prob:.1%})")
                
        # --- MINIMUM SCORE FILTER (The TRADE #1 Fix) ---
        # Prevent very weak signals from executing.
        # TRADE #1 executed with Score 16.5 and lost. This shouldn't happen.
        MIN_SCORE_THRESHOLD = 25.0
        if abs(score) < MIN_SCORE_THRESHOLD and execute:
            execute = False
            vetoes.append(f"Weak Signal Veto: Score {abs(score):.1f} < {MIN_SCORE_THRESHOLD}")
            reasons.append(f"Weak Signal Filter: Score too low ({abs(score):.1f})")


        # --- OMEGA MULTIPLIER LOGIC ---
        # PRIORITY: Use lot_multiplier from consensus.py details if available (10x OMEGA)
        lot_multiplier = details.get('lot_multiplier', 1.0) if isinstance(details, dict) else 1.0
        
        # Fallback logic only if consensus didn't set a multiplier
        # Fallback logic only if consensus didn't set a multiplier
        if lot_multiplier <= 1.0 and execute:
            if setup in ["LION_PROTOCOL", "LION_BREAKOUT", "SMART_MONEY_REVERSAL"]:
                lot_multiplier = 10.0  # Upgraded for High Conviction
            elif setup == "STRUCTURE_TREND_RIDER":
                lot_multiplier = 30.0 # User requested aggressive system ("turbinar")
            elif setup == "VOID_FILLER_FVG":
                lot_multiplier = 30.0 # SPLIT FIRE: Aggressive Filling
                reasons.append("SPLIT FIRE ACTIVATED: Aggressive Void Filling")
            elif setup in ["GOLDEN_COIL_M8", "VOLATILITY_SQUEEZE"]:
                lot_multiplier = 20.0 # SPLIT FIRE: Trend/Momentum
                reasons.append(f"SPLIT FIRE ACTIVATED: {setup}")
            elif setup == "NANO_SCALE_IN":
                lot_multiplier = 3.0
            elif setup == "MOMENTUM_BREAKOUT" and confidence >= 95.0:
                lot_multiplier = 5.0  # Upgraded from 2.0
            elif setup == "MOMENTUM_BREAKOUT":
                lot_multiplier = 2.0  # Upgraded from 1.0

        prediction = LaplacePrediction(
            execute=execute,
            direction=decision_dir,
            confidence=confidence,
            strength=SignalStrength.STRONG if abs(score) > 40 else SignalStrength.MODERATE,
            primary_signal=setup if setup else "CONSENSUS_VOTE",
            reasons=reasons,
            vetoes=vetoes,
            logic_vector=score,
            details=details,
            lot_multiplier=lot_multiplier # Pass calculated multiplier
        )
        
        if execute:
            self._apply_risk_management(prediction, df_m5)
            
        return prediction
        
    def _apply_risk_management(self, prediction, df_m5):
        """Standard SL/TP calculation (Sniper Protocol) - Phase 7 Volatility Expansion"""
        # ... (simplified) ...
        # Get ATR
        atr = ta.volatility.AverageTrueRange(df_m5['high'], df_m5['low'], df_m5['close'], window=14).average_true_range().iloc[-1]
        
        # SL = 40.0 pips (Fixed per Balance Optimization)
        sl_pips = 40.0 
        
        # TP = 20.0 pips (Balanced Scalp Profile)
        tp_pips = 20.0 

        
        prediction.sl_pips = sl_pips
        prediction.tp_pips = tp_pips
        
        # Convert to Price
        pip_val = 0.0001
        current_price = df_m5['close'].iloc[-1]
        
        if prediction.direction == "BUY":
            prediction.sl_price = current_price - (sl_pips * pip_val)
            prediction.tp_price = current_price + (tp_pips * pip_val)
        else:
            prediction.sl_price = current_price + (sl_pips * pip_val)
            prediction.tp_price = current_price - (tp_pips * pip_val)

_laplace_instance: Optional[LaplaceDemonCore] = None
def get_laplace_demon(symbol: str = "GBPUSD") -> LaplaceDemonCore:
    global _laplace_instance
    if _laplace_instance is None or _laplace_instance.symbol != symbol:
        _laplace_instance = LaplaceDemonCore(symbol)
    return _laplace_instance
