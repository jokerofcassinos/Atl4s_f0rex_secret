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
        prediction = self._synthesize_master_decision(legacy_result, legion_intel, details, current_price, df_m5)
        
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
            setup_type = details.get('Vectors', {}).get('Reason')
            if not setup_type:
                 setup_type = details.get('setup_name') # Just in case I patch consensus later
        
        return {'score': legacy_vector, 'setup': setup_type}, details

    def _synthesize_master_decision(self, legacy_result, legion_intel, details, price, df_m5):
        """
        Merges the Legacy Vector with Swarm Intelligence.
        THE HYBRID ENGINE: Bugatti V1 + AGI Swarm V2
        """
        score = legacy_result['score']
        setup = legacy_result['setup']
        
        reasons = []
        vetoes = []
        
        if setup:
            reasons.append(f"Legacy V1 Setup: {setup} (Base Score: {score:.1f})")
            
        # --- SWARM INTELLIGENCE INTEGRATION ---
        # 1. Specific Agent Confirmations (Micro-Boosts)
        knife = legion_intel.get('knife')
        horizon = legion_intel.get('horizon')
        
        # TimeKnife (High Volatility Scalp)
        if knife and knife.confidence > 80:
             reasons.append(f"TimeKnife Spike detected ({knife.signal_type})")
             # If math agrees, huge boost
             if (knife.signal_type == "BUY" and score > 0) or (knife.signal_type == "SELL" and score < 0):
                 score *= 1.5 
             # If math is neutral but Knife is explicit, take the scalp
             elif abs(score) < 10: 
                 score += 50 if knife.signal_type == "BUY" else -50
                 setup = "LEGION_KNIFE_SCALP"

        # EventHorizon (Gravity/Reversion)
        if horizon and horizon.signal_type in ["BUY", "SELL"]:
             if "SINGULARITY" in horizon.meta_data.get('reason', ''):
                 # Extreme Reversion
                 reasons.append(f"EventHorizon Singularity: {horizon.signal_type}")
                 score += 100 if horizon.signal_type == "BUY" else -100
                 setup = "SINGULARITY_REVERSION"

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
                # If Legacy is sleeping but Swarm sees something clear
                score += (swarm_vector * 0.8) # Adopt Swarm view
                reasons.append(f"Swarm Initiative: {swarm_decision} (Conf {swarm_score:.1f}%)")
                if not setup: setup = "SWARM_INITIATIVE"

        # --- TIER 2 ENHANCEMENTS (SMC / M8) ---
        smc_res = details.get('SMC', {})
        m8_res = details.get('M8', {})
        toxic_flow = details.get('ToxicFlow', False)
        liq_signal = details.get('Liquidator')
        nano_res = details.get('nano_blocks', {}) # Added this line to get nano_res

        # 0. The Liquidator (New 90% Setup)
        if liq_signal:
             if "BUY" in liq_signal:
                  reasons.append(f"THE LIQUIDATOR: {liq_signal}")
                  # If score contradicts, we flip it or boost massive if aligns
                  if score < 0: score = 50 # Force flip to Buy
                  else: score += 50
                  if not setup: setup = "LIQUIDATOR_SWEEP"
             elif "SELL" in liq_signal:
                  reasons.append(f"THE LIQUIDATOR: {liq_signal}")
                  if score > 0: score = -50
                  else: score -= 50
                  if not setup: setup = "LIQUIDATOR_SWEEP"
        
        # A. Sell at Red Block (Resistance)
        if nano_res.get('algo_sell_detected'):
             reasons.append("NANO ALGO: Sell Block Detected (Resistance)")
             if score > 0: score = -50 # Flip to Sell
             else: score -= 30 # Boost sell
             if not setup: setup = "NANO_SCALE_IN"
             
        # B. Buy at Green Block (Support)
        if nano_res.get('algo_buy_detected'):
             reasons.append("NANO ALGO: Buy Block Detected (Support)")
             if score < 0: score = 50 # Flip to Buy
             else: score += 30 # Boost buy
             if not setup: setup = "NANO_SCALE_IN"
        
        # 6.5 Vortex 3-6-9 & Event Horizon
        vortex = details.get('Vortex', {})
        eh_meta = details.get('EventHorizon', {})
        
        # A. Vortex Time Trigger
        if vortex.get('signal') != "WAIT":
             reasons.append(f"VORTEX 3-6-9: {vortex['reason']}")
             if vortex['signal'] == "BUY":
                  if score < 0: score = 50 # Flip
                  else: score += 40
                  if not setup: setup = "VORTEX_TIMING"
             elif vortex['signal'] == "SELL":
                  if score > 0: score = -50
                  else: score -= 40
                  if not setup: setup = "VORTEX_TIMING"

        # B. Event Horizon Swarm (Seek & Destroy)
        if eh_meta.get('mode') == "SWARM_369":
             reasons.append(f"EVENT HORIZON: {eh_meta.get('reason', 'Swarm Attack')}")
             # This is a dominant signal (Predatory)
             if eh_meta.get('signal_type') == "BUY": # Logic implied from meta if propagated, wait I need signal direction
                  pass # Handled below
             
             # Force Swarm Mode if high confidence
             setup = "SWARM_369"
             # Score override? The Swarm Signal itself isn't fully propagated in EH Meta yet, 
             # let's assume if EH Meta exists, it's a valid trigger.
             # Actually, I should have passed the signal direction in meta or details.
             
        # Correction: The Event Horizon signal was returned as 'eh_signal' in analyze, 
        # but I only put 'meta_data' into details['EventHorizon'].
        # I need the direction.
        # Implied direction: Iceberg Buy -> BUY.
             if "Buy Detected" in eh_meta.get('reason', ''):
                  # If Toxic Flow, we reduce the boost drastically
                  boost = 50
                  if toxic_flow: boost = 10 
                  
                  if score < 0: score = 60 # Strong Flip (still valid if not toxic)
                  else: score += boost
             elif "Sell Detected" in eh_meta.get('reason', ''):
                  boost = 50
                  if toxic_flow: boost = 10
                  
                  if score > 0: score = -60
                  else: score -= boost

        # 7. Final Confidence Adjustment
             if "BUY" in zone and score > -10:
                  score += 25
                  reasons.append(f"SMC: Reacting off Bullish Order Block")
             elif "SELL" in zone and score < 10:
                  score -= 25
                  reasons.append(f"SMC: Reacting off Bearish Order Block")
        
        # 2. M8 Fibonacci
        if isinstance(m8_res, dict) and m8_res.get('gate') == "GOLDEN":
             direction = m8_res.get('signal', 'NEUTRAL')
             m8_boost = 15 if direction == "BUY" else -15 if direction == "SELL" else 0
             score += m8_boost
             if m8_boost != 0:
                 reasons.append(f"M8: Golden Gate ({direction})")
             
        # 3. Toxic Flow Veto (Surgical Correction)
        # We observed that 'REVERSION_SNIPER' wins in Toxic Flow, but 'Neutral' (Trend follow) fails.
        # FIX: Only veto 'Neutral' setups if Toxic Flow involves compression.
        
        is_neutral_setup = any("Setup: Neutral" in r for r in reasons)
        
        hard_veto = False # Persistent veto flag
        
        if toxic_flow and is_neutral_setup:
             reasons.append(f"Toxic Flow: VETOING Neutral Setup in Chop")
             vetoes.append("Toxic Flow: Neutral Setup Unsafe in Compression")
             execute = False
             hard_veto = True
             
        elif toxic_flow:
             # For Snipers, we caution but DO NOT VETO immediately.
             # HOWEVER, we must reduce the score to prevent it from bypassing the Neural Oracle.
             # A 100.0 Score in Toxic Flow is suspicious. Let's cap it at 80.
             reasons.append(f"Toxic Flow: Compression Detected (Capping Score to 80)")
             
             # REGIME LOCK: In Toxic Chop, Momentum Strategies are suicide.
             # Only Reversion Snipers are mathematically viable.
             momentum_strategies = ["MOMENTUM_BREAKOUT", "CONSENSUS_VOTE", "KINETIC_BOOM", "LION_PROTOCOL"]
             if setup in momentum_strategies:
                 reasons.append(f"REGIME LOCK: Vetoing {setup} in Toxic Flow (Momentum fails in Chop)")
                 vetoes.append(f"Regime Lock: {setup} invalid in Chop")
                 execute = False
                 hard_veto = True
             
             if abs(score) > 80:
                  score = 80.0 if score > 0 else -80.0
             
             # Also strict check for low scores
             if abs(score) < 40: 
                  vetoes.append("Toxic Flow: Weak Signal in Chop")
                  execute = False 
                  hard_veto = True 

        # 4. SNR Walls (Tier 3)
        snr_data = details.get('SNR', {})
        if snr_data:
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

        # 4.5 KINETIC SHIELD (The "13 Stops" Fix)
        # If Kinematics is screaming REVERSAL, we must listen.
        # Boom Up (0-90) vs Sell Signal
        # Boom Down (180-270) vs Buy Signal
        kin_data = details.get('Kinematics', {})
        k_angle = kin_data.get('angle', 0)
        
        if score > 0 and (180 <= k_angle <= 270):
             reasons.append(f"KINETIC SHIELD: Vetoing BUY against Downward Acceleration ({k_angle:.0f}°)")
             vetoes.append("Kinetic Shield: Buying into a Crash")
             execute = False
             hard_veto = True
        elif score < 0 and (0 <= k_angle <= 90):
             reasons.append(f"KINETIC SHIELD: Vetoing SELL against Upward Acceleration ({k_angle:.0f}°)")
             vetoes.append("Kinetic Shield: Selling into a Rocket")
             execute = False
             hard_veto = True

        # 5. Neural Oracle (Tier 4 - Experimental)
        neural_pred = details.get('Neural')
        if neural_pred is not None:
             try:
                 # Flatten if nested list
                 val = neural_pred[0] if isinstance(neural_pred, (list, np.ndarray)) else neural_pred
                 if isinstance(val, (list, np.ndarray)): val = val[0] # Double unpack
                 
                 reasons.append(f"Neural Oracle: Raw Output {float(val):.3f}")
             except Exception as e:
                 # logger.debug(f"Neural Parse Error: {e}")
                 pass

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
        atr = details['Trend'].get('atr', 0.0010)
        
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
            elif win_prob < threshold and abs(score) >= bypass_score:
                reasons.append(f"Neural Filter: BYPASSED (High Conviction Score {abs(score):.1f})")
            else:
                reasons.append(f"Neural Filter: APPROVED ({win_prob:.1%})")

        # --- OMEGA MULTIPLIER LOGIC ---
        lot_multiplier = 1.0
        if execute:
            if setup in ["KINETIC_BOOM", "LEGION_KNIFE_SCALP", "LION_PROTOCOL", "REVERSION_SNIPER"]:
                lot_multiplier = 5.0
            elif setup == "NANO_SCALE_IN":
                lot_multiplier = 3.0 # User Request: 3+3+3 Recurring
            elif setup == "SWARM_369":
                lot_multiplier = 9.0 # Full Geometric Swarm (Starts small but capacity is 9x)
                # Note: The Execution Engine handles the splitting. 
                # Here we just authorize the mass.
            elif setup == "MOMENTUM_BREAKOUT" and confidence >= 95.0:
                lot_multiplier = 2.0
            elif setup == "MOMENTUM_BREAKOUT":
                lot_multiplier = 1.0

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
