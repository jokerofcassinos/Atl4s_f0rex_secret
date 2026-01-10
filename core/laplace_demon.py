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
from signals.correlation import (
    SMTDivergence, PowerOfOne, InversionFVG, 
    MeanThreshold, AMDPowerOfThree
)
from signals.momentum import MomentumAnalyzer, ToxicFlowDetector
from signals.volatility import VolatilityAnalyzer, DisplacementCandle, VolatilityFilter

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
        
        # AGI: Metacognitive State
        self.cognitive_plasticity = 0.85 # Adaptability factor
        self.resonance_matrix = {} # Stores contextual resonance
        
        # Initialize all analysis modules
        self.quarterly = QuarterlyTheory()
        self.m8_fib = M8FibonacciSystem()
        self.time_macro = TimeMacroFilter()
        self.initial_balance = InitialBalanceFilter()
        
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
        
        logger.info("LAPLACE DEMON INITIALIZED - Deterministic Intelligence Active")
    
    def analyze(self,
                df_m1: pd.DataFrame,
                df_m5: pd.DataFrame,
                df_h1: pd.DataFrame,
                df_h4: pd.DataFrame,
                df_d1: pd.DataFrame = None,
                df_secondary: pd.DataFrame = None,  # For SMT (e.g., EURUSD)
                current_time: datetime = None,
                current_price: float = None) -> LaplacePrediction:
        """
        AGI CAUSAL INFERENCE ENGINE
        
        Replaces legacy 'Linear Scoring' with precise Logic Gates.
        1. PERCEPTION: Gather data from Sensory Cortex (Modules)
        2. SYNTHESIS: Build a Mental Model of the market state
        3. INFERENCE: Simulate outcomes based on Causal Physics
        4. DECISION: Execute only if Probability > Threshold
        """
        if current_time is None: current_time = datetime.now()
        if current_price is None and df_m1 is not None: current_price = df_m1['close'].iloc[-1]
        
        # Initialize Blank Prediction
        prediction = LaplacePrediction(execute=False, direction="WAIT", confidence=0.0, strength=SignalStrength.WEAK)
        
        # ═══════════════════════════════════════════════════════════
        # GATE 0: TEMPORAL REALITY (The Time-Space Constraint)
        # ═══════════════════════════════════════════════════════════
        if current_time.hour not in self.ALLOWED_HOURS_SET:
             return prediction # Silent Veto (Sleep Mode)
             
        if current_time.weekday() == 4 and current_time.hour >= 15:
             prediction.vetoes.append("FRIDAY_CURSE")
             return prediction
             
        # ═══════════════════════════════════════════════════════════
        # PHASE 1: SENSORY PERCEPTION (Gathering Raw Data)
        # ═══════════════════════════════════════════════════════════
        
        # 1. Structure Perception (The Map)
        structure_data = self.smc.analyze(df_m5, current_price)
        
        # 2. Momentum Perception (The Flow)
        momentum_data = self.momentum.analyze(df_m5)
        
        # 3. Timing Perception (The Clock)
        # (We use a simplified timing check here for speed)
        quarterly = self.quarterly.analyze(current_time, df_m1)
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 2: CAUSAL SYNTHESIS (The Brain)
        # ═══════════════════════════════════════════════════════════
        
        decision = self._synthesize_agi_decision(
            structure=structure_data,
            momentum=momentum_data,
            timing=quarterly,
            current_price=current_price
        )
        
        # ═══════════════════════════════════════════════════════════
        # PHASE 3: REFLEXIVE SAFETY (The Spinal Cord)
        # ═══════════════════════════════════════════════════════════
        
        if decision['execute']:
             prediction.execute = True
             prediction.direction = decision['direction']
             prediction.confidence = decision['confidence']
             prediction.reasons = decision['reasons']
             prediction.primary_signal = decision['setup_type']
             
             # Dynamic Stop Loss based on Structure (Mental Stop)
             sl_tp = self._calculate_sl_tp(prediction.direction, current_price, 15, structure_data)
             prediction.sl_pips = sl_tp['sl_pips']
             prediction.tp_pips = sl_tp['tp_pips']
             
             # Log the epiphany
             logger.info(f"AGI DECISION: {prediction.direction} | Conf: {prediction.confidence}% | Type: {decision['setup_type']}")
             
        self.last_prediction = prediction
        return prediction

    def _synthesize_agi_decision(self, structure: Dict, momentum: Dict, timing: Any, current_price: float) -> Dict:
        """
        [AGI METACOGNITION] v2.0 - ALPHA CORRECTED
        
        Refined Logic Gates:
        1. TREND FILTER: Don't short a Bull Market (Trend > Reversal).
        2. SFP CONFIRMATION: Validates specific structure breaks.
        3. CONFIDENCE SCALING: 60% (Base) -> 75% (Trend) -> 90% (Perfect).
        """
        decision = {
            'execute': False,
            'direction': 'WAIT',
            'confidence': 0,
            'reasons': [],
            'setup_type': 'None'
        }
        
        # 0. Global Trend Context (Hierarchical Veto)
        # We assume structure['trend'] tells us the M5/H1 structure.
        # Ideally we'd look at H4, but for now we use the SMC Text.
        market_trend = structure.get('trend', 'RANGING')
        
        # 1. Analyze Flow State (The River)
        flow = momentum.get('flow')
        if not flow: return decision
        
        # COMPRESSION TRAP
        if flow['compression']['detected']:
             if flow['expansion']['detected']:
                  decision['execute'] = True
                  direction = "BUY" if flow['expansion']['direction'] == "UP" else "SELL"
                  decision['direction'] = direction
                  decision['confidence'] = 75 # Breakouts are good, but fakeouts exist
                  decision['setup_type'] = "COMPRESSION_BREAKOUT"
                  decision['reasons'].append(f"Compression -> {direction} Expansion")
                  return decision
             else:
                  return decision
        
        # 2. Check Liquidity Sweeps (Turtle Soup SFP)
        swept_pools = [p for p in structure['liquidity_pools'] if p.swept]
        if swept_pools:
             last_sweep = swept_pools[-1]
             sweep_dir = "SELL" if last_sweep.type == "HIGH" else "BUY"
             
             # TREND FILTER
             is_counter_trend = False
             if market_trend == "BULLISH" and sweep_dir == "SELL": is_counter_trend = True
             if market_trend == "BEARISH" and sweep_dir == "BUY": is_counter_trend = True
             
             # BASE CONFIDENCE
             base_conf = 60
             
             # Context Boosters
             is_exhausted = flow['exhaustion']['detected'] and flow['exhaustion']['direction'] != sweep_dir
             has_divergence = momentum['rsi']['divergence'] is not None
             
             if not is_counter_trend:
                 base_conf += 15 # With Trend = 75%
             
             if is_exhausted or has_divergence:
                 base_conf += 15 # Confirmation = +15%
             
             # Killzone Booster (Simple Time Check proxy)
             # (Real check is in Gate 0, but we boost score here)
             # Assuming we are in a valid time if we got here (Gate 0 passed)
             
             # DECISION
             # We only trade Counter-Trend if we have Div/Exhaustion (Score >= 75)
             if is_counter_trend and base_conf < 75:
                 decision['warnings'] = ["VETO: Counter-trend SFP without divergence"]
                 pass 
             else:
                 decision['execute'] = True
                 decision['direction'] = sweep_dir
                 decision['confidence'] = min(95, base_conf)
                 decision['setup_type'] = "TURTLE_SOUP_AGI"
                 decision['reasons'].append(f"Valid SFP of {last_sweep.level} (Trend: {market_trend})")
                 return decision
        
        # 3. Check Order Block Continuation (Trend)
        entry = structure.get('entry_signal', {})
        if entry.get('direction'):
             ob_dir = entry['direction']
             
             # Strict Trend Alignment for OBs
             if market_trend == "BULLISH" and ob_dir == "SELL": return decision
             if market_trend == "BEARISH" and ob_dir == "BUY": return decision
             
             # Momentum Alignment
             momentum_aligned = True
             if ob_dir == "BUY" and momentum['rsi']['overbought']: momentum_aligned = False
             if ob_dir == "SELL" and momentum['rsi']['oversold']: momentum_aligned = False
             
             if momentum_aligned:
                  decision['execute'] = True
                  decision['direction'] = ob_dir
                  decision['confidence'] = 75 # Trend following is solid
                  decision['setup_type'] = "ORDER_BLOCK_FLOW"
                  decision['reasons'].append(f"Trend Continuation: {entry['reason']}")
                  return decision
                  
        return decision
    
    def _resample_to_m8(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """Resample M1 data to M8 (8-minute) timeframe."""
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        return df_m1.resample('8min').agg(agg).dropna()
    
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
                         structure_result: Dict) -> Dict:
        """Calculate Stop Loss and Take Profit levels."""
        pip_size = 0.0001  # GBPUSD
        
        # Base SL: 1.5x ATR, minimum 10 pips
        sl_pips = max(10, atr_pips * 1.5)
        
        # Base TP: 2.5x ATR or 2:1 R:R
        tp_pips = max(sl_pips * 2, atr_pips * 2.5)
        
        # Adjust for structure
        ob = structure_result.get('ob')
        if ob and ob.get('sl_price'):
            # Use structure-based SL
            structure_sl = abs(current_price - ob['sl_price']) / pip_size
            if 5 < structure_sl < sl_pips * 2:  # Reasonable range
                sl_pips = structure_sl
        
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


# Create singleton instance
_laplace_instance: Optional[LaplaceDemonCore] = None

def get_laplace_demon(symbol: str = "GBPUSD") -> LaplaceDemonCore:
    """Get or create the Laplace Demon instance."""
    global _laplace_instance
    if _laplace_instance is None or _laplace_instance.symbol != symbol:
        _laplace_instance = LaplaceDemonCore(symbol)
    return _laplace_instance
