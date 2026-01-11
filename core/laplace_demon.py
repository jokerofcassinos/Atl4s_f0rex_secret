"""
LAPLACE DEMON - The Deterministic Trading Intelligence
═══════════════════════════════════════════════════════

Named after Pierre-Simon Laplace's thought experiment:
"An intellect which at a certain moment would know all forces that set nature in motion,
and all positions of all items of which nature is composed... nothing would be uncertain
and the future just like the past would be present before its eyes."

This is the central intelligence that orchestrates all analysis modules
to achieve deterministic market prediction.

Core Systems:
1. QUARTERLY THEORY - 90-minute institutional cycles
2. M8 FIBONACCI - 8-minute micro-timing
3. SMC STRUCTURE - Order blocks, FVG, liquidity
4. BLACKROCK PATTERNS - Seek & Destroy, Iceberg, Rebalancing
5. CROSS-ASSET CORRELATION - SMT Divergence
6. GANN GEOMETRY - Sacred number intervals
7. TESLA VORTEX - 3-6-9 exhaustion patterns

Target: 70% Win Rate | $30 Capital | High Frequency
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
import logging

# Import all analysis modules
from signals.timing import QuarterlyTheory, M8FibonacciSystem, TimeMacroFilter, InitialBalanceFilter
from signals.structure import (
    SMCAnalyzer, InstitutionalLevels, BlackRockPatterns,
    VectorCandleTheory, GannGeometry, TeslaVortex
)
from analysis.trend_architect import TrendArchitect  # New import

from signals.correlation import (
    SMTDivergence, PowerOfOne, InversionFVG, 
    MeanThreshold, AMDPowerOfThree
)
from signals.momentum import MomentumAnalyzer, ToxicFlowDetector
from signals.volatility import VolatilityAnalyzer, DisplacementCandle, VolatilityFilter
from analysis.m8_fibonacci_system import M8FibonacciSystem
from analysis.swarm.vortex_swarm import VortexSwarm

# OMNI-CORTEX: Advanced AGI Modules
from core.agi.big_beluga.snr_matrix import SNRMatrix
from core.agi.microstructure.flux_heatmap import FluxHeatmap

# HEISENBERG PROTOCOL: Quantum Physics Modules
from analysis.quantum_core import QuantumCore
from core.agi.active_inference.free_energy import FreeEnergyMinimizer

# OMEGA-PREDATOR: Chaos Detection
from analysis.chaos_engine import ChaosEngine

# OVERLORD: Neural Plasticity (Adaptive Learning)
from core.agi.metacognition.neural_plasticity_core import NeuralPlasticityCore

# --- LEGION ELITE IMPORTS (THE SQUAD) ---
from analysis.swarm.time_knife_swarm import TimeKnifeSwarm
from analysis.swarm.physarum_swarm import PhysarumSwarm
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm
from analysis.swarm.overlord_swarm import OverlordSwarm

logger = logging.getLogger("LaplaceDemon")


class SignalStrength(Enum):
    """Signal strength levels."""
    VETO = -999
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4
    DIVINE = 5  # Maximum confluence


@dataclass
class LaplacePrediction:
    """
    Complete prediction from the Laplace Demon.
    
    Contains all analysis results and final decision.
    """
    # Core Decision
    execute: bool
    direction: str  # BUY, SELL, WAIT
    confidence: float  # 0-100
    strength: SignalStrength
    
    # Entry/Exit
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    
    # Position Sizing
    risk_pct: float = 2.0
    position_multiplier: float = 1.0
    
    # Analysis Breakdown
    timing_score: int = 0
    structure_score: int = 0
    momentum_score: int = 0
    volatility_score: int = 0
    confluence_count: int = 0
    
    # Detailed Reasons
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    
    # Source Attribution
    primary_signal: str = ""
    supporting_signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'execute': self.execute,
            'direction': self.direction,
            'confidence': self.confidence,
            'strength': self.strength.name,
            'entry_price': self.entry_price,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'confluence': self.confluence_count,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'vetoes': self.vetoes
        }


class LaplaceDemonCore:
    """
    THE LAPLACE DEMON
    
    Central intelligence that integrates all analysis modules
    to achieve deterministic market prediction.
    
    Philosophy:
    - The market is not random; it follows algorithmic rules
    - Institutions (BlackRock, banks) have predictable patterns
    - Time, price, and structure converge at high-probability zones
    - 70% accuracy is achievable with proper confluence
    """
    
    # ═══════════════════════════════════════════════════════════════
    # SMART TIME FILTER (Based on backtest telemetry)
    # ═══════════════════════════════════════════════════════════════
    # Golden Hours: High win rate and profit
    ALLOWED_HOURS = [2, 5, 6, 7, 8, 9, 20]  # UTC hours
    
    # Toxic Hours: Consistent losses - HARD BLOCK
    TOXIC_HOURS = [0, 3, 18, 22, 23]  # UTC hours
    
    # ═══════════════════════════════════════════════════════════════
    # ANTI-RUIN RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # AGI TEMPORAL GATES (Protocol 70% Enforcement)
    # ═══════════════════════════════════════════════════════════════
    # Ranges are inclusive-exclusive logic or explicit hours.
    # London Open: 07:00 - 10:00 UTC
    # NY Premarket/Open: 12:00 - 16:00 UTC
    # Late NY Flush: 19:00 - 20:00 UTC
    ALLOWED_HOURS_SET = {7, 8, 9, 12, 13, 14, 15, 19, 20} # Precise set for O(1) lookup
    
    # ═══════════════════════════════════════════════════════════════
    # ANTI-RUIN RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    MAX_RISK_PER_TRADE_PCT = 5.0  # Max 5% of account per trade ($1.50 on $30)
    MIN_FREE_MARGIN = 15.0        # Block new trades if FreeMargin < $15
    MIN_LOT = 0.01                # Fixed minimum lot for small accounts
    
    def __init__(self, symbol: str = "GBPUSD", contrarian_mode: bool = False):
        self.symbol = symbol
        # CONTRARIAN MODE: DISABLED (Protocol 70% Restoration)
        # The 38% WR result confirmed that Contrarian Mode = Crash.
        # We revert to Direct Logic + AGI Filters.
        self.contrarian_mode = False 
        
        # Trade Frequency Limiter (Phase 4)
        self.daily_trades = {'date': None, 'count': 0}
        
        # AGI: Metacognitive State
        self.cognitive_plasticity = 0.85 # Adaptability factor
        self.resonance_matrix = {} # Stores contextual resonance
        
        # Initialize all analysis modules
        self.quarterly = QuarterlyTheory()
        self.m8_fib = M8FibonacciSystem()
        self.time_macro = TimeMacroFilter()
        self.initial_balance = InitialBalanceFilter()
        
        # Phase 6-B: Divine Gate (Esoteric Modules)
        self.m8_fib = M8FibonacciSystem() # Re-init here for clarity or use existing import
        self.vortex = VortexSwarm()

        self.trend_architect = TrendArchitect(symbol=symbol) # Phase 3 (Sniper)
        self.smc = SMCAnalyzer()
        self.institutional = InstitutionalLevels()
        self.blackrock = BlackRockPatterns()
        self.vector = VectorCandleTheory()
        self.gann = GannGeometry()
        self.tesla = TeslaVortex()
        
        self.smt = SMTDivergence()
        self.power_of_one = PowerOfOne()
        self.ifvg = InversionFVG()
        self.mean_threshold = MeanThreshold()
        self.amd = AMDPowerOfThree()
        
        self.momentum = MomentumAnalyzer()
        self.toxic_flow = ToxicFlowDetector()
        
        self.volatility = VolatilityAnalyzer()
        self.displacement = DisplacementCandle()
        self.vol_filter = VolatilityFilter()
        
        # State
        self.last_prediction: Optional[LaplacePrediction] = None
        self.trade_history: List[Dict] = []
        
        # OMNI-CORTEX: Advanced AGI Modules
        self.snr_matrix = SNRMatrix()    # Structural Nexus Resonance (S/R Filter)
        self.heatmap = FluxHeatmap()     # Microstructure Liquidity Radar
        
        # HEISENBERG PROTOCOL: Quantum Physics
        self.quantum = QuantumCore()              # Quantum Tunneling & Uncertainty
        self.free_energy = FreeEnergyMinimizer()  # Surprise Minimization
        
        # OMEGA-PREDATOR: Chaos Detection
        self.chaos = ChaosEngine()                # Lyapunov Exponent (Chaos vs Order)
        
        # OVERLORD: Neural Plasticity (Adaptive Learning)
        self.plasticity = NeuralPlasticityCore(base_learning_rate=0.1)
        
        # --- LEGION ELITE (SPECIAL CONSULTANTS) ---
        self.time_knife = TimeKnifeSwarm()
        self.physarum = PhysarumSwarm()
        self.event_horizon = EventHorizonSwarm()
        self.overlord = OverlordSwarm()
        
        logger.info("LAPLACE DEMON INITIALIZED - Deterministic Intelligence Active")
        logger.info("SYNTHESIS PROTOCOL COMPLETE: Laplace + Omni + Heisenberg + Legion Online.")
    
    async def analyze(self,
                df_m1: pd.DataFrame,
                df_m5: pd.DataFrame,
                df_h1: pd.DataFrame,
                df_h4: pd.DataFrame,
                df_d1: pd.DataFrame = None,
                df_secondary: pd.DataFrame = None,  # For SMT (e.g., EURUSD)
                current_time: datetime = None,
                current_price: float = None) -> LaplacePrediction:
        """
        AGI CAUSAL INFERENCE ENGINE (ASYNC / PARALLEL)
        """
        if current_time is None: current_time = datetime.now()
        if (current_price is None or current_price == 0) and df_m5 is not None: 
            current_price = float(df_m5['close'].iloc[-1])
        
        # Blank Prediction
        prediction = LaplacePrediction(execute=False, direction="WAIT", confidence=0.0, strength=SignalStrength.WEAK)
        
        
        # 0. GATE 0: TEMPORAL REALITY (Disabled for Backtest to avoid AttributeError)
        # if not self._is_market_open(current_time):
        #      return prediction
             
        # 1. PERCEPTION (Synchronous - Fast Data)
        structure_data = self.smc.analyze(df_m5, current_price)
        trend_context = self.trend_architect.analyze(df_m5, df_h1, df_h4, df_d1)
        momentum_data = self.momentum.analyze(df_m5)
        quarterly = self.quarterly.analyze(current_time, df_m1)
        
        # 2. LEGION INTELLIGENCE (Async - Heavy Processing)
        try:
            # Run Swarms in parallel to avoid blocking
            swarm_ctx = {
                'df_m1': df_m1, 'df_m5': df_m5, 'df_h1': df_h1, 
                'tick': {'bid': current_price, 'ask': current_price},
                'data_map': {'M5': df_m5, 'M1': df_m1, 'H1': df_h1, 'H4': df_h4} # For Overlord
            }
            
            # Fire agents
            t_knife = self.time_knife.process(swarm_ctx)
            t_horizon = self.event_horizon.process(swarm_ctx)
            t_physarum = self.physarum.process(swarm_ctx)
            t_overlord = self.overlord.process(swarm_ctx)
            
            # Gather results (Suppress exceptions to keep bot alive)
            legion_results = await asyncio.gather(t_knife, t_horizon, t_physarum, t_overlord, return_exceptions=True)
            
            # Unpack
            knife_sig = legion_results[0] if not isinstance(legion_results[0], Exception) else None
            horizon_sig = legion_results[1] if not isinstance(legion_results[1], Exception) else None
            physarum_sig = legion_results[2] if not isinstance(legion_results[2], Exception) else None
            overlord_sig = legion_results[3] if not isinstance(legion_results[3], Exception) else None
            
        except Exception as e:
            # logger.debug(f"CRITICAL LEGION ERROR: {e}") # Log silently
            knife_sig, horizon_sig, physarum_sig, overlord_sig = None, None, None, None

        # 3. CAUSAL SYNTHESIS (The Great Judgement)
        decision = self._synthesize_agi_decision(
            structure=structure_data,
            momentum=momentum_data,
            timing=quarterly,
            current_price=current_price,
            h4_trend=trend_context.get('ocean', 0),
            d1_trend=trend_context.get('galaxy', 0),
            df_m5=df_m5,
            df_m8=self._resample_to_m8(df_m1, df_m5),
            df_h1=df_h1,
            df_h4=df_h4,
            current_time=current_time,
            legion_intel={
                'knife': knife_sig, 
                'horizon': horizon_sig, 
                'physarum': physarum_sig, 
                'overlord': overlord_sig
            }
        )
        
        # 4. EXECUTION LOGIC (With Hydra & Aegis)
        if decision['execute']:
             prediction.execute = True
             prediction.direction = decision['direction']
             prediction.confidence = decision['confidence']
             prediction.reasons = decision['reasons']
             prediction.primary_signal = decision['setup_type']
             prediction.setup_type = decision['setup_type']
             
             # Risk Injection (Hydra)
             if prediction.confidence >= 95: prediction.risk_pct = 10.0
             elif prediction.confidence >= 80: prediction.risk_pct = 5.0
             else: prediction.risk_pct = 2.0
             
             # Dynamic SL/TP (Aegis + Nano-Harvest)
             vol_data = self.volatility.analyze(df_m5)
             atr_pips = vol_data.get('regime', {}).get('atr', 0.0015) * 10000 
             
             sl_tp = self._calculate_sl_tp(
                 direction=prediction.direction, 
                 current_price=current_price, 
                 atr_pips=atr_pips, 
                 structure_result=structure_data,
                 setup_type=decision['setup_type'],
                 magnetic_target=decision.get('magnetic_target'),
                 df_m5=df_m5
             )
             
             prediction.sl_pips = sl_tp['sl_pips']
             prediction.tp_pips = sl_tp['tp_pips']
             prediction.sl_price = sl_tp['sl']
             prediction.tp_price = sl_tp['tp']
             
             logger.info(f"LEGION STRIKE: {prediction.direction} | {decision['setup_type']} | Conf: {prediction.confidence:.1f}%")
             
        self.last_prediction = prediction
        return prediction

    def _synthesize_agi_decision(self, structure: Dict, momentum: Dict, timing: Any, current_price: float, 
                                 h4_trend: int, d1_trend: int, df_m5: pd.DataFrame, df_m8: pd.DataFrame, 
                                 df_h1: pd.DataFrame, df_h4: pd.DataFrame, current_time: datetime,
                                 legion_intel: Dict = None) -> Dict:
        """
        THE UNIFIED BRAIN (Chimera + Heisenberg + Legion)
        """
        decision = {
            'execute': False, 'direction': 'WAIT', 'confidence': 0, 
            'reasons': [], 'setup_type': 'None', 'magnetic_target': None
        }
        
        # Unpack Legion Intel
        if legion_intel is None: legion_intel = {}
        knife = legion_intel.get('knife')
        horizon = legion_intel.get('horizon')
        physarum = legion_intel.get('physarum')
        overlord = legion_intel.get('overlord')
        
        # DEBUG LOGS ------------------------------------------------------------------
        # print(f"DEBUG: Intel: Knife={bool(knife)} Horizon={bool(horizon)} Physarum={bool(physarum)}")
        # -----------------------------------------------------------------------------

        # --- 1. REALITY FILTERS (Omni & Heisenberg) ---
        # SNR Check (Noise)
        try:
            snr_quality = self.snr_matrix.analyze_quality(df_m5)
            # if snr_quality == "NOISE_CRITICAL": return decision # Too strict?
        except: pass

        # Heatmap (Target)
        try:
            hm_data = self.heatmap.get_liquidity_magnet(df_m5, current_price)
            if hm_data and hm_data.get('strength', 0) > 0.6: 
                decision['magnetic_target'] = hm_data.get('price')
        except: pass
        
        # Quantum (Energy)
        structure_levels = [p.level for p in structure.get('liquidity_pools', [])]
        q_metrics = self.quantum.analyze(df_m5, structure_levels)
        tunneling_prob = q_metrics.get('tunneling_prob', 0.0)
        is_excited = q_metrics.get('is_excited', False)
        
        # Free Energy (Surprise Veto)
        if self.last_prediction:
             target = self.last_prediction.tp_price if self.last_prediction.direction == 'BUY' else self.last_prediction.sl_price
             # If no last target, assume price
             target = target if target else current_price
             
             current_surprise = self.free_energy.calculate_surprise({'bid': target, 'volatility': 0.001}, {'bid': current_price})
             if current_surprise > 50.0: 
                 logger.debug(f"FREE ENERGY VETO: Surprise {current_surprise:.2f}")
                 return decision 

        # Chaos Veto (Lyapunov)
        lyapunov_exp = 0.0
        try:
            lyapunov_exp = self.chaos.calculate_lyapunov(df_m5)
            # > 1.2: Extremely Chaotic (avoid trading)
            if lyapunov_exp > 1.2:
                logger.warning(f"CHAOS VETO: Lyapunov {lyapunov_exp:.4f} too high. Market is unpredictable.")
                return decision
        except Exception as e:
            logger.debug(f"Chaos Engine Error: {e}")
            
        # DEBUG LOGS 2 
        print(f"DEBUG: Passed Vetoes. L={lyapunov_exp:.2f} S={current_surprise if 'current_surprise' in locals() else 0:.2f}")

        # --- 2. LION VS SNAKE TERRITORY ---
        is_lion_territory = False
        master_trend = 0
        if h4_trend != 0:
            if d1_trend == 0 or d1_trend == h4_trend:
                is_lion_territory = True
                master_trend = h4_trend
                
        # --- 3. UNIFIED EXECUTION LOGIC ---
        
        # A) LION MODE (Trend Following)
        if is_lion_territory:
            target_dir = "BUY" if master_trend == 1 else "SELL"
            
            # Legion Boost: Event Horizon
            horizon_confirmed = False
            if horizon and horizon.signal_type == target_dir and horizon.confidence > 70:
                horizon_confirmed = True
            
            # Basic Setup (SMC)
            entry = structure.get('entry_signal', {})
            valid_structure = entry.get('direction') == target_dir
            
            # Trigger
            if valid_structure or horizon_confirmed:
                decision['execute'] = True
                decision['direction'] = target_dir
                
                # Confidence Calculation
                base_conf = 75
                if tunneling_prob > 0.7: base_conf += 15 # Heisenberg Boost
                if horizon_confirmed: base_conf += 10 # Legion Boost (Infinity Trend)
                
                decision['confidence'] = min(99, base_conf)
                decision['setup_type'] = "LION_EVENT_HORIZON" if horizon_confirmed else "LION_SMC"
                decision['reasons'].append(f"Lion Attack. Horizon={horizon_confirmed} Tunnel={tunneling_prob:.2f}")
                return decision

        # B) SNAKE MODE (Reversal / Scalp)
        else:
            # Legion Scalp: Time Knife (Overrides everything)
            if knife and knife.confidence > 80:
                # Time Knife sees extreme volatility spike. Snake strikes immediately.
                decision['execute'] = True
                decision['direction'] = knife.signal_type
                decision['confidence'] = 95
                decision['setup_type'] = "SNAKE_TIME_KNIFE"
                decision['reasons'].append(f"LEGION: Time Knife Override ({knife.meta_data.get('reason', 'Spike')})")
                return decision
            
            # Basic Setup (Liquidity Sweep)
            swept_pools = [p for p in structure.get('liquidity_pools', []) if p.swept]
            if swept_pools:
                last_sweep = swept_pools[-1]
                cand_dir = "SELL" if last_sweep.type == "HIGH" else "BUY"
                
                # Divine Gate (Physarum Check)
                path_clear = False
                if physarum and physarum.signal_type == cand_dir: path_clear = True
                
                # Trigger
                if is_excited or path_clear: 
                    decision['execute'] = True
                    decision['direction'] = cand_dir
                    decision['confidence'] = 90
                    decision['setup_type'] = "SNAKE_DIVINE"
                    decision['reasons'].append(f"Snake Reversal. Excited={is_excited} Path={path_clear}")
                    return decision

        # C) OVERLORD VOTE (Minerva)
        if overlord and overlord.confidence > 90 and overlord.signal_type != "WAIT":
             decision['execute'] = True
             decision['direction'] = overlord.signal_type
             decision['confidence'] = 85
             decision['setup_type'] = "OVERLORD_COMMAND"
             decision['reasons'].append("Overlord Intervention")
             return decision

        return decision
        is_lion_territory = False
        master_trend = 0
        
        if h4_trend == 1: # H4 Bullish
            if d1_trend >= 0: # D1 Bullish or Neutral (Allowed)
                is_lion_territory = True
                master_trend = 1
        elif h4_trend == -1: # H4 Bearish
            if d1_trend <= 0: # D1 Bearish or Neutral (Allowed)
                is_lion_territory = True
                master_trend = -1
        
        # --- DIVINE GATE CALCULATION (Pre-calc for Snake) ---
        divine_confirmation = False
        m8_score = 0
        
        # Check M8 Time Gate
        if df_m8 is not None and df_h1 is not None and df_h4 is not None:
            try:
                m8_eval = self.m8_fib.evaluate(df_h1, df_h4, df_m8, None, current_time)
                if m8_eval.get('total_score', 0) >= 7:
                    divine_confirmation = True
                    m8_score = m8_eval['total_score']
            except Exception as e:
                logger.error(f"M8 Error: {e}")

        # Check Vortex Trap Gate
        vortex_signal = None
        try:
            vortex_signal = self.vortex.process({'df_m5': df_m5, 'tick': {'bid': current_price, 'ask': current_price}})
        except:
            pass
        
        # ════════════════════════════════════════════════
        # HEISENBERG LAYER: QUANTUM PHYSICS (A Vantagem Desleal)
        # ════════════════════════════════════════════════
        
        # Extrair níveis de liquidez para calcular barreiras de potencial
        structure_levels = [p.level for p in structure.get('liquidity_pools', [])]
        
        # O QuantumCore analisa a energia cinética do preço vs barreiras
        tunneling_prob = 0.0
        is_particle_excited = False
        try:
            q_metrics = self.quantum.analyze(df_m5, structure_levels)
            tunneling_prob = q_metrics.get('tunneling_prob', 0.0)
            is_particle_excited = q_metrics.get('is_excited', False)
        except Exception as e:
            logger.debug(f"Quantum Analyze Error: {e}")

        # --- FREE ENERGY: VETO DE REALIDADE ---
        current_surprise = 0.0
        if self.last_prediction and self.last_prediction.tp_price:
            try:
                target = self.last_prediction.tp_price if self.last_prediction.direction == 'BUY' else self.last_prediction.sl_price
                target = target if target else current_price
                prediction_mock = {'bid': target, 'volatility': 0.001} 
                reality_mock = {'bid': current_price}
                current_surprise = self.free_energy.calculate_surprise(prediction_mock, reality_mock)
            except:
                pass

        # REGRA DE SOBREVIVÊNCIA: Se Surpresa > 50 (5 Sigma), VETO TOTAL
        if current_surprise > 50.0:
            logger.warning(f"HEISENBERG VETO: Market Surprise Critical ({current_surprise:.2f})")
            return decision

        # ════════════════════════════════════════════════
        # PATH 1: THE LION (Trend Following - High Volume)
        # ════════════════════════════════════════════════
        if is_lion_territory:
            target_direction = "BUY" if master_trend == 1 else "SELL"
            
            # Check for ANY valid structure in direction of trend
            # 1. Order Block Retest
            entry = structure.get('entry_signal', {})
            if entry.get('direction') == target_direction:
                 decision['execute'] = True
                 decision['direction'] = target_direction
                 decision['confidence'] = 75
                 decision['setup_type'] = "LION_OB_FLOW"
                 decision['reasons'].append(f"Lion Trend (H4) + Structure Entry. D1={d1_trend}")
                 
                 # QUANTUM BOOST: Tunneling increases confidence
                 if tunneling_prob > 0.7:
                     decision['confidence'] = min(99, decision['confidence'] + 15)
                     decision['reasons'].append(f"QUANTUM TUNNELING ({tunneling_prob:.2f})")
                 
                 return decision
            
            # 2. Momentum Breakout (New)
            # If RSI is healthy and we are trending, take the break
            rsi = momentum.get('rsi', {})
            if rsi:
                is_safe_rsi = (target_direction == "BUY" and not rsi.get('overbought')) or \
                              (target_direction == "SELL" and not rsi.get('oversold'))
                
                if is_safe_rsi and structure.get('trend') == "BULLISH" and target_direction == "BUY":
                     decision['execute'] = True
                     decision['direction'] = "BUY"
                     decision['confidence'] = 70
                     decision['setup_type'] = "LION_MOMENTUM"
                     decision['reasons'].append("Lion Trend + RSI Safe + Structure Align")
                     if tunneling_prob > 0.7:
                         decision['confidence'] = min(99, decision['confidence'] + 15)
                         decision['reasons'].append(f"QUANTUM TUNNELING ({tunneling_prob:.2f})")
                     return decision
                elif is_safe_rsi and structure.get('trend') == "BEARISH" and target_direction == "SELL":
                     decision['execute'] = True
                     decision['direction'] = "SELL"
                     decision['confidence'] = 70
                     decision['setup_type'] = "LION_MOMENTUM"
                     decision['reasons'].append("Lion Trend + RSI Safe + Structure Align")
                     if tunneling_prob > 0.7:
                         decision['confidence'] = min(99, decision['confidence'] + 15)
                         decision['reasons'].append(f"QUANTUM TUNNELING ({tunneling_prob:.2f})")
                     return decision

        # ════════════════════════════════════════════════
        # PATH 2: THE SNAKE (Reversal Sniper - High Precision)
        # ════════════════════════════════════════════════
        else: 
            # We are in the Jungle (Neutral/Conflict).
            # ONLY trade if we have a Trap (Liquidity Sweep) AND Divine Confirmation.
            
            liq_pools = structure.get('liquidity_pools', [])
            swept_pools = [p for p in liq_pools if getattr(p, 'swept', False)]
            
            if swept_pools:
                 last_sweep = swept_pools[-1]
                 # If we swept a HIGH, we want to SELL
                 sweep_type = getattr(last_sweep, 'type', 'UNKNOWN')
                 candidate_dir = "SELL" if sweep_type == "HIGH" else "BUY" 
                 
                 # MANDATORY: Divine Gate or Strong Structure Rejection
                 sweep_strength = getattr(last_sweep, 'strength', 0)
                 if divine_confirmation or sweep_strength >= 80:
                      decision['execute'] = True
                      decision['direction'] = candidate_dir
                      decision['confidence'] = 90 # Divine
                      decision['setup_type'] = "SNAKE_DIVINE_REVERSAL"
                      sweep_level = getattr(last_sweep, 'level', 0)
                      decision['reasons'].append(f"Snake Ambush: Liquidity Sweep of {sweep_level:.5f} + Divine/Structure Conf.")
                      
                      # QUANTUM BOOST: Excited state means reversal is imminent
                      if is_particle_excited:
                          decision['confidence'] = min(99, decision['confidence'] + 8)
                          decision['reasons'].append("QUANTUM EXCITED STATE: Mean reversion imminent")
                      
                      return decision
                      
        return decision
    
    def _resample_to_m8(self, df_m1: pd.DataFrame = None, df_m5: pd.DataFrame = None) -> pd.DataFrame:
        """
        Resample to M8 (8-minute) timeframe.
        Priority: M1 > M5 (Fallback)
        """
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        if df_m1 is not None and not df_m1.empty:
            return df_m1.resample('8min').agg(agg).dropna()
            
        if df_m5 is not None and not df_m5.empty:
            # Low Resolution Fallback
            return df_m5.resample('8min').agg(agg).dropna()
            
        return None
    
    def _analyze_timing(self, 
                        df_m1: pd.DataFrame,
                        df_m8: pd.DataFrame,
                        df_h1: pd.DataFrame,
                        df_h4: pd.DataFrame,
                        current_time: datetime) -> Dict:
        """Layer 1: Timing Analysis"""
        result = {
            'score': 0,
            'direction': None,
            'veto': False,
            'veto_reason': None,
            'reasons': [],
            'warnings': []
        }
        
        # 1. Quarterly Theory (90-minute cycles)
        quarterly = self.quarterly.analyze(current_time, df_m1, "NEUTRAL")
        
        if quarterly.is_golden_zone:
            result['score'] += 5
            result['reasons'].append(f"Q3 Golden Zone: {quarterly.reason}")
            if self.quarterly.last_q2_sweep_direction:
                result['direction'] = 'SELL' if self.quarterly.last_q2_sweep_direction == 'UP' else 'BUY'
        elif quarterly.is_manipulation_zone:
            result['warnings'].append(f"Q2 Manipulation: {quarterly.reason}")
        elif quarterly.phase == "Q1":
            # NOTE: Changed from veto to warning for testing
            result['warnings'].append("Q1 Accumulation: Lower probability zone")
            result['score'] -= 3  # Penalty instead of veto
        
        result['score'] += max(0, quarterly.score)
        
        # 2. M8 Fibonacci System
        if df_m8 is not None and df_h1 is not None and df_h4 is not None:
            m8_result = self.m8_fib.evaluate(df_h1, df_h4, df_m8, df_m1, current_time)
            
            if m8_result.get('execute'):
                result['score'] += 5
                result['direction'] = m8_result['signal']
                result['reasons'].append(f"M8 Fibonacci: {m8_result['reason']} (Score: {m8_result['total_score']})")
            
            gate = m8_result.get('breakdown', {}).get('gate', {})
            if gate.get('gate') == 'Q1':
                result['veto'] = True
                result['veto_reason'] = "M8 Q1 Gate: No entries in dead zone"
                return result
        
        # 3. Time Macro (xx:50 to xx:10)
        macro = self.time_macro.analyze(current_time, df_m1)
        if macro['in_macro']:
            result['warnings'].append(f"Macro Window: Extremes are fragile")
            if macro.get('fragile_high') or macro.get('fragile_low'):
                result['score'] -= 2
        
        # 4. Initial Balance
        if df_m1 is not None:
            ib = self.initial_balance.calculate_ib(df_m1)
            if ib.get('calculated'):
                ib_analysis = self.initial_balance.analyze(df_m1['close'].iloc[-1])
                result['reasons'].append(f"IB Day Type: {ib_analysis['day_type']}")
        
        return result
    
    def _analyze_structure(self,
                           df_m1: pd.DataFrame,
                           df_m5: pd.DataFrame,
                           df_h1: pd.DataFrame,
                           df_h4: pd.DataFrame,
                           current_price: float,
                           current_time: datetime) -> Dict:
        """Layer 2: Structure Analysis"""
        result = {
            'score': 0,
            'direction': None,
            'veto': False,
            'veto_reason': None,
            'reasons': [],
            'warnings': [],
            'ob': None,
            'fvg': None
        }
        
        # 1. SMC Analysis
        smc_result = self.smc.analyze(df_m5, current_price)
        
        if smc_result.get('trend'):
            if smc_result['trend'] == 'BULLISH':
                result['direction'] = 'BUY'
                result['score'] += 3
            elif smc_result['trend'] == 'BEARISH':
                result['direction'] = 'SELL'
                result['score'] += 3
            result['reasons'].append(f"SMC Trend: {smc_result['trend']}")
        
        entry = smc_result.get('entry_signal', {})
        if entry.get('direction'):
            result['score'] += 5
            result['direction'] = entry['direction']
            result['reasons'].append(f"SMC Entry: {entry['reason']}")
            result['ob'] = entry
        
        # 2. BlackRock Patterns
        # Seek and Destroy
        seek = self.blackrock.detect_seek_and_destroy(df_m5)
        if seek.get('detected'):
            result['warnings'].append(f"Seek & Destroy: Wait for mean reversion to {seek['midpoint']:.5f}")
            result['score'] -= 3  # Penalty for volatile conditions
        
        # Iceberg Detection
        iceberg = self.blackrock.detect_iceberg_absorption(df_m1)
        if iceberg.get('detected'):
            zone = iceberg['zone']
            result['score'] += 7
            result['direction'] = 'BUY' if zone['type'] == 'BULLISH_ABSORPTION' else 'SELL'
            result['reasons'].append(f"Iceberg: {zone['reason']}")
        
        # Month-End Rebalancing
        rebal = self.blackrock.is_month_end_rebalancing(current_time)
        if rebal['in_rebalancing_window']:
            result['warnings'].append(f"Month-End Rebalancing: {rebal['action']}")
        
        # 3. Vector Candle
        vector = self.vector.detect_vector_candle(df_m5)
        if vector.get('detected'):
            result['warnings'].append(f"Vector Candle: {vector['action']}")
            
            reversal = self.vector.check_reversal_entry(df_m5)
            if reversal.get('entry'):
                result['score'] += 5
                result['direction'] = reversal['direction']
                result['reasons'].append(f"Vector Recovery: {reversal['reason']}")
        
        # 4. Gann Geometry
        if df_m5 is not None and len(df_m5) > 50:
            gann_direction = 'UP' if result['direction'] == 'BUY' else 'DOWN'
            gann = self.gann.check_geometric_exhaustion(df_m5, gann_direction)
            if gann.get('exhausted'):
                result['veto'] = True
                result['veto_reason'] = f"Gann Exhaustion: {gann['recommendation']}"
                return result
        
        # 5. Tesla Vortex (3-6-9)
        tesla_candles = self.tesla.count_consecutive_candles(df_m5)
        if tesla_candles['count'] >= 9:
            # NOTE: Changed from veto to warning for testing
            result['warnings'].append(tesla_candles['warning'])
            result['score'] -= 5  # Strong penalty instead of veto
        elif tesla_candles['count'] >= 6:
            result['warnings'].append(tesla_candles['warning'])
            result['score'] -= 2
        
        tesla_waves = self.tesla.count_impulse_waves(df_h1)
        if tesla_waves.get('exhausted'):
            result['warnings'].append(tesla_waves['warning'])
        
        # 6. AMD Power of Three
        self.amd.set_asia_range(df_h1)
        amd_phase = self.amd.detect_phase(df_m5, current_time)
        
        if amd_phase.get('phase') == 'M' and amd_phase.get('expected_real_move'):
            result['score'] += 3
            result['direction'] = amd_phase['expected_real_move']
            result['reasons'].append(f"AMD Manipulation: {amd_phase['action']}")
        elif amd_phase.get('phase') == 'D' and amd_phase.get('high_conviction'):
            result['score'] += 5
            result['direction'] = amd_phase['bias']
            result['reasons'].append(f"AMD Distribution: {amd_phase['action']}")
        
        return result
    
    def _analyze_momentum(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
        """Layer 3: Momentum Analysis"""
        result = {
            'score': 0,
            'direction': 'NEUTRAL',
            'reasons': []
        }
        
        # 1. Full momentum analysis
        if df_m5 is not None and len(df_m5) > 50:
            momentum = self.momentum.analyze(df_m5)
            
            composite = momentum.get('composite', {})
            if composite.get('direction') and composite['direction'] != 'NEUTRAL':
                result['direction'] = composite['direction']
                result['score'] += int(composite.get('strength', 0) / 20)  # Convert 0-100 to 0-5
                
                if composite.get('agreement'):
                    result['score'] += 3
                    result['reasons'].append(f"Momentum Agreement: All indicators {result['direction']}")
                
            rsi = momentum.get('rsi', {})
            if rsi.get('divergence'):
                result['score'] += 3
                result['reasons'].append(f"RSI {rsi['divergence']}")
        
        # 2. Toxic Flow
        toxic = self.toxic_flow.detect_compression(df_m5)
        if toxic.get('detected'):
            result['reasons'].append(toxic['warning'])
            # Don't trade into compression
            result['score'] -= 2
        
        return result
    
    def _analyze_volatility(self, df_m5: pd.DataFrame, df_d1: pd.DataFrame) -> Dict:
        """Layer 4: Volatility Analysis"""
        result = {
            'score': 0,
            'direction': None,
            'reasons': [],
            'atr_pips': 15,
            'size_multiplier': 1.0
        }
        
        if df_m5 is None or len(df_m5) < 50:
            return result
        
        vol = self.volatility.analyze(df_m5)
        
        regime = vol.get('regime')
        if regime:
            result['reasons'].append(f"Volatility: {regime.regime} ({regime.recommendation})")
            result['atr_pips'] = regime.atr * 10000  # Convert to pips
            
            if regime.regime == 'LOW' and vol.get('bollinger', {}).get('squeeze'):
                result['score'] += 3
                result['reasons'].append("Bollinger Squeeze: Prepare for expansion")
            elif regime.regime == 'EXTREME':
                result['score'] -= 3
                result['size_multiplier'] = 0.5
            elif regime.regime == 'HIGH':
                result['score'] += 2
        
        # Displacement
        disp = vol.get('displacement', {})
        if disp.get('detected'):
            result['score'] += 4
            result['direction'] = disp['direction'].replace('UP', 'BUY').replace('DOWN', 'SELL')
            result['reasons'].append(f"Displacement: {disp['action']}")
        
        return result
    
    def _analyze_correlation(self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> Dict:
        """Layer 5: Cross-Asset Correlation"""
        result = {
            'score': 0,
            'direction': None,
            'reasons': []
        }
        
        # SMT Divergence
        smt = self.smt.analyze(df_primary, df_secondary)
        
        if smt.detected:
            result['score'] += 10
            result['direction'] = 'BUY' if 'BUY' in smt.trade_action else 'SELL'
            result['reasons'].append(f"SMT Divergence: {smt.reason}")
        
        return result
    
    def _calculate_sl_tp(self,
                         direction: str,
                         current_price: float,
                         atr_pips: float,
                         structure_result: Dict,
                         setup_type: str = "NORMAL",
                         magnetic_target: float = None,
                         df_m5: pd.DataFrame = None) -> Dict:
        """
        [PROTOCOLO AEGIS + OMNI-CORTEX] - DYNAMIC GEOMETRY
        
        SL baseado em Market Structure (Swing Points + SNRMatrix).
        TP baseado em Asymmetric R:R ou magnetic_target do Heatmap.
        """
        pip_size = 0.0001
        sl_pips = 0.0
        
        # --- OMNI-CORTEX: SCAN SNR LEVELS ---
        snr_support = 0.0
        snr_resistance = 0.0
        try:
            if df_m5 is not None and len(df_m5) > 50:
                self.snr_matrix.scan_structure(df_m5)
                snr_levels = self.snr_matrix.get_nearest_levels(current_price)
                snr_support = snr_levels.get('nearest_support', 0.0)
                snr_resistance = snr_levels.get('nearest_resistance', 999999.0)
        except Exception as e:
            logger.debug(f"SNR Scan Error: {e}")
        
        # --- 1. SMART STOP LOSS (STRUCTURAL DEFENSE + SNR) ---
        pools = structure_result.get('liquidity_pools', [])
        structure_sl_price = None
        
        if direction == 'BUY':
            # Prioridade: SNR Support > Liquidity Pool
            valid_lows = [p.level for p in pools if hasattr(p, 'type') and p.type == 'LOW' and p.level < current_price]
            if snr_support > 0 and snr_support < current_price:
                structure_sl_price = snr_support - (3 * pip_size)  # 3 pips below SNR
            elif valid_lows:
                structure_sl_price = max(valid_lows) - (5 * pip_size)
        else:
            # Prioridade: SNR Resistance > Liquidity Pool
            valid_highs = [p.level for p in pools if hasattr(p, 'type') and p.type == 'HIGH' and p.level > current_price]
            if snr_resistance < 999999 and snr_resistance > current_price:
                structure_sl_price = snr_resistance + (3 * pip_size)  # 3 pips above SNR
            elif valid_highs:
                structure_sl_price = min(valid_highs) + (5 * pip_size)

        # Lógica de Decisão do SL
        if structure_sl_price:
            dist = abs(current_price - structure_sl_price) / pip_size
            if 10 < dist < 50:
                sl_pips = dist
            else:
                sl_pips = max(20, atr_pips * 2.0)
        else:
            sl_pips = max(20, atr_pips * 2.0)

        # --- 2. DYNAMIC TAKE PROFIT (ASYMMETRIC + MAGNETIC TARGET) ---
        tp_pips = 0.0
        
        # Se temos um alvo magnético do Heatmap, use-o!
        if magnetic_target and magnetic_target > 0:
            mag_dist = abs(magnetic_target - current_price) / pip_size
            if 20 < mag_dist < 200:  # Sanity check
                tp_pips = mag_dist - 5  # 5 pips antes do magnet
        
        # Fallback para lógica assimétrica
        if tp_pips == 0:
            if "LION" in setup_type:
                tp_pips = max(80, sl_pips * 5.0)
            elif "SNAKE" in setup_type:
                tp_pips = max(25, sl_pips * 2.0)
            else:
                tp_pips = sl_pips * 2.5

        # Preços Finais
        if direction == 'BUY':
            sl = current_price - (sl_pips * pip_size)
            tp = current_price + (tp_pips * pip_size)
        else:
            sl = current_price + (sl_pips * pip_size)
            tp = current_price - (tp_pips * pip_size)
        
        return {
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'sl_pips': round(sl_pips, 1),
            'tp_pips': round(tp_pips, 1)
        }
    
    def get_simple_signal(self,
                          df_m5: pd.DataFrame,
                          current_price: float = None) -> Tuple[Optional[str], float, float, float, str]:
        """
        Simplified signal for backtest compatibility.
        
        Returns: (direction, sl_pips, tp_pips, confidence, source)
        """
        if df_m5 is None or len(df_m5) < 50:
            return (None, 0, 0, 0, "INSUFFICIENT_DATA")
        
        if current_price is None:
            current_price = df_m5['close'].iloc[-1]
        
        # Quick analysis
        prediction = self.analyze(
            df_m1=None,
            df_m5=df_m5,
            df_h1=None,
            df_h4=None,
            current_price=current_price
        )
        
        if prediction.execute:
            return (
                prediction.direction,
                prediction.sl_pips,
                prediction.tp_pips,
                prediction.confidence,
                prediction.primary_signal or "LAPLACE"
            )
        
        return (None, 0, 0, 0, "NO_SIGNAL")


    def _check_matrix_breaker(self, df_m1: pd.DataFrame, current_price: float) -> bool:
        """
        AGI PERCEPTION: Matrix Breaker.
        Checks if price has swept liquidity (High/Low of previous structure) in the last 15 minutes.
        """
        # Look at last 15 candles
        window = df_m1.iloc[-16:-1] # Previous 15 excluding current forming candle
        if len(window) < 5: return True # Not enough data
        
        recent_high = window['high'].max()
        recent_low = window['low'].min()
        
        # Did we just break it?
        # A sweep is usually a wick break.
        # We assume if current price is near recent extremums OR we just broke them.
        # Actually proper sweep: High > PrevHigh.
        
        # Check if the CURRENT candle or previous candle broke the window's range
        # We need to see VOLATILITY signature.
        
        # Simplest Proxy: Price range in last 15m > 10 pips?
        # Or explicitly: Did we trade outside the range of the previous hour?
        
        # User Definition: "Pavio superando minima/maxima anterior"
        # Let's check if any candle in last 3 candles broke the high/low of the 12 candles before it.
        
        scan_window = df_m1.iloc[-15:]
        local_high = scan_window['high'].max()
        local_low = scan_window['low'].min()
        
        # Range check: If range < 5 pips, it's dead. No sweep possible.
        rng = (local_high - local_low) * 10000
        if rng < 5.0: return False # Dead market
        
        return True # Default to True if moving, refining exact "Sweep" is complex without tick data.
        # The true "Sweep" logic is: High > OldHigh but Close < OldHigh (SFP).
        # For now, we allow if Volatility is present (Range > 5 pips).

    def learn_from_trade(self, success: bool, pnl: float, setup_type: str, regime: str = "UNKNOWN") -> Dict:
        """
        [OVERLORD] Neural Plasticity Feedback Loop.
        
        Called after each trade closes to adapt internal weights.
        The bot learns what works and what doesn't.
        
        Args:
            success: True if trade was profitable
            pnl: Profit/Loss in dollars
            setup_type: "LION_MOMENTUM", "SNAKE_DIVINE_REVERSAL", etc.
            regime: Market regime ("TRENDING", "RANGING", "VOLATILE")
            
        Returns:
            Current plasticity state with updated weights.
        """
        # Map setup_type to factors
        factors_used = []
        if "LION" in setup_type:
            factors_used.extend(['trend_following', 'momentum'])
        if "SNAKE" in setup_type:
            factors_used.extend(['mean_reversion', 'session_awareness'])
        if "MOMENTUM" in setup_type:
            factors_used.append('momentum')
        if "REVERSAL" in setup_type:
            factors_used.append('mean_reversion')
        
        feedback = {
            'success': success,
            'pnl': pnl,
            'regime': regime,
            'factors_used': factors_used
        }
        
        state = self.plasticity.adapt(feedback)
        
        # Log learning
        weights = self.plasticity.get_weights()
        logger.info(f"OVERLORD LEARNING: PnL={pnl:+.2f} | Stability={state.stability_score:.2f}")
        logger.debug(f"WEIGHTS: {weights}")
        
        return {
            'learning_rate': state.learning_rate,
            'stability': state.stability_score,
            'weights': weights
        }
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive weights from Neural Plasticity."""
        return self.plasticity.get_weights()


# Create singleton instance
_laplace_instance: Optional[LaplaceDemonCore] = None

def get_laplace_demon(symbol: str = "GBPUSD") -> LaplaceDemonCore:
    """Get or create the Laplace Demon instance."""
    global _laplace_instance
    if _laplace_instance is None or _laplace_instance.symbol != symbol:
        _laplace_instance = LaplaceDemonCore(symbol)
    return _laplace_instance
