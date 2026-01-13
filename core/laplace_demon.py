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
from signals.momentum import MomentumAnalyzer, ToxicFlowDetector
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
        
        # --- 6. TIER 3 AGI CORE ---
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
        if 'tick' in kwargs:
             flux_metrics = self.flux_heatmap.update(kwargs['tick'])
        
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
        
        # 2. LEGACY CONSENSUS (CPU Bound - Run in Thread)
        # The heavy math engines
        loop = asyncio.get_event_loop() # Changed from get_running_loop()
        legacy_result, details = await loop.run_in_executor(
            None, # Use default executor (self.executor if set, otherwise default ThreadPoolExecutor)
            self._run_legacy_consensus, 
            data_map
        )
        
        # --- TIER 2 SIGNALS (SMC / M8 / Toxic) ---
        details['SMC'] = self.smc_signal.analyze(df_m5, current_price)
        details['ToxicFlow'] = self.toxic_flow.detect_compression(df_m5)
        details['Flux'] = flux_metrics

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
        
        # 1. SMC Order Blocks
        if isinstance(smc_res, dict) and smc_res.get('zone_type') and smc_res.get('zone_type') != "NONE":
             zone = smc_res['zone_type']
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
             
        # 3. Toxic Flow Veto
        if toxic_flow:
             reasons.append(f"Toxic Flow: Compression Detected (Caution)")
             if abs(score) < 30: 
                  score *= 0.5 

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
                 
             # VETO SELL into Support
             if score < 0 and sup_dist < 0.0005:
                 score *= 0.1
                 vetoes.append(f"SNR Wall Veto: Support Ahead ({sup_dist:.5f})")
                 if not setup: setup = "BLOCKED_BY_WALL"

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
            
            # CALIBRATION: Since training data (83 trades) is small, we use a more relaxed threshold.
            # Base WR was 41%, so anything > 40% is 'acceptable' for now.
            # We also BYPASS veto if Consensus score is extremely high (Momentum Overlap).
            
            threshold = 0.40 # Reduced from 0.60
            bypass_score = 45.0 # Extremely high conviction
            
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
        
        # SL = 200 pips (User Requested: Restore distance)
        sl_pips = max(200, (atr * 3.5) * 10000)
        
        # TP = Optimized Ratio (User Request: Lower TP for faster rotation)
        tp_pips = sl_pips * 0.8 # 1:0.8 RR (Ensures trades close faster)
        
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
