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
    
    def __init__(self, symbol: str = "GBPUSD"):
        self.symbol = symbol
        
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
        MAIN ANALYSIS FUNCTION
        
        Runs all analysis modules and synthesizes into a single prediction.
        
        Args:
            df_m1: 1-minute data
            df_m5: 5-minute data
            df_h1: 1-hour data
            df_h4: 4-hour data
            df_d1: Daily data (optional)
            df_secondary: Secondary pair for SMT (optional)
            current_time: Current datetime
            current_price: Current market price
        
        Returns:
            LaplacePrediction with complete trading decision
        """
        if current_time is None:
            current_time = datetime.now()
        
        if current_price is None and df_m1 is not None and len(df_m1) > 0:
            current_price = df_m1['close'].iloc[-1]
        
        # Initialize prediction
        prediction = LaplacePrediction(
            execute=False,
            direction="WAIT",
            confidence=0.0,
            strength=SignalStrength.WEAK
        )
        
        # Create M8 timeframe (resample M1 to 8-minute)
        df_m8 = None
        if df_m1 is not None and len(df_m1) > 20:
            df_m8 = self._resample_to_m8(df_m1)
        
        # ═══════════════════════════════════════════════════════════
        # LAYER 1: TIMING ANALYSIS (Max +15 points)
        # ═══════════════════════════════════════════════════════════
        
        timing_result = self._analyze_timing(
            df_m1, df_m8, df_h1, df_h4, current_time
        )
        
        prediction.timing_score = timing_result['score']
        prediction.reasons.extend(timing_result.get('reasons', []))
        prediction.warnings.extend(timing_result.get('warnings', []))
        
        # Timing veto check
        if timing_result.get('veto'):
            prediction.vetoes.append(timing_result['veto_reason'])
            prediction.execute = False
            prediction.direction = "WAIT"
            self.last_prediction = prediction
            return prediction
        
        # ═══════════════════════════════════════════════════════════
        # LAYER 2: STRUCTURE ANALYSIS (Max +20 points)
        # ═══════════════════════════════════════════════════════════
        
        structure_result = self._analyze_structure(
            df_m1, df_m5, df_h1, df_h4, current_price, current_time
        )
        
        prediction.structure_score = structure_result['score']
        prediction.reasons.extend(structure_result.get('reasons', []))
        prediction.warnings.extend(structure_result.get('warnings', []))
        
        # Structure veto check
        if structure_result.get('veto'):
            prediction.vetoes.append(structure_result['veto_reason'])
            prediction.execute = False
            prediction.direction = "WAIT"
            self.last_prediction = prediction
            return prediction
        
        # ═══════════════════════════════════════════════════════════
        # LAYER 3: MOMENTUM ANALYSIS (Max +10 points)
        # ═══════════════════════════════════════════════════════════
        
        momentum_result = self._analyze_momentum(df_m5, df_h1)
        
        prediction.momentum_score = momentum_result['score']
        prediction.reasons.extend(momentum_result.get('reasons', []))
        
        # ═══════════════════════════════════════════════════════════
        # LAYER 4: VOLATILITY ANALYSIS (Max +10 points)
        # ═══════════════════════════════════════════════════════════
        
        volatility_result = self._analyze_volatility(df_m5, df_d1)
        
        prediction.volatility_score = volatility_result['score']
        prediction.reasons.extend(volatility_result.get('reasons', []))
        prediction.position_multiplier = volatility_result.get('size_multiplier', 1.0)
        
        # ═══════════════════════════════════════════════════════════
        # LAYER 5: CROSS-ASSET CORRELATION (Max +15 points)
        # ═══════════════════════════════════════════════════════════
        
        correlation_score = 0
        if df_secondary is not None:
            correlation_result = self._analyze_correlation(df_m5, df_secondary)
            correlation_score = correlation_result['score']
            prediction.reasons.extend(correlation_result.get('reasons', []))
        
        # ═══════════════════════════════════════════════════════════
        # SYNTHESIS: Calculate Final Decision
        # ═══════════════════════════════════════════════════════════
        
        total_score = (
            prediction.timing_score +
            prediction.structure_score +
            prediction.momentum_score +
            prediction.volatility_score +
            correlation_score
        )
        
        # Determine direction from strongest signal
        direction_votes = {
            'BUY': 0,
            'SELL': 0,
            'WAIT': 0
        }
        
        if structure_result.get('direction'):
            direction_votes[structure_result['direction']] += structure_result['score']
        
        if timing_result.get('direction'):
            direction_votes[timing_result['direction']] += timing_result['score'] * 0.5
        
        if momentum_result.get('direction') and momentum_result['direction'] != 'NEUTRAL':
            direction_votes[momentum_result['direction']] += momentum_result['score']
        
        # Determine winner
        max_votes = max(direction_votes.values())
        if max_votes > 0:
            for d, v in direction_votes.items():
                if v == max_votes and d != 'WAIT':
                    prediction.direction = d
                    break
        
        # Calculate confidence
        max_score = 70  # Maximum possible
        prediction.confidence = min(95, (total_score / max_score) * 100)
        
        # Confluence count
        prediction.confluence_count = sum([
            1 if prediction.timing_score > 5 else 0,
            1 if prediction.structure_score > 5 else 0,
            1 if prediction.momentum_score > 3 else 0,
            1 if prediction.volatility_score > 3 else 0,
            1 if correlation_score > 5 else 0
        ])
        
        # Determine strength
        if prediction.confluence_count >= 5:
            prediction.strength = SignalStrength.DIVINE
        elif prediction.confluence_count >= 4:
            prediction.strength = SignalStrength.EXTREME
        elif prediction.confluence_count >= 3:
            prediction.strength = SignalStrength.STRONG
        elif prediction.confluence_count >= 2:
            prediction.strength = SignalStrength.MODERATE
        else:
            prediction.strength = SignalStrength.WEAK
        
        # Final execution decision
        min_confluence = 2
        min_confidence = 60
        
        if (prediction.confluence_count >= min_confluence and 
            prediction.confidence >= min_confidence and
            prediction.direction != 'WAIT' and
            len(prediction.vetoes) == 0):
            
            prediction.execute = True
            
            # Calculate SL/TP
            sl_tp = self._calculate_sl_tp(
                prediction.direction,
                current_price,
                volatility_result.get('atr_pips', 15),
                structure_result
            )
            
            prediction.entry_price = current_price
            prediction.sl_price = sl_tp['sl']
            prediction.tp_price = sl_tp['tp']
            prediction.sl_pips = sl_tp['sl_pips']
            prediction.tp_pips = sl_tp['tp_pips']
        
        self.last_prediction = prediction
        return prediction
    
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
            result['veto'] = True
            result['veto_reason'] = "Q1 Accumulation: Dead zone - no trades"
            return result
        
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
            direction = 'UP' if result['direction'] == 'BUY' else 'DOWN'
            gann = self.gann.check_geometric_exhaustion(df_m5, direction)
            if gann.get('exhausted'):
                result['veto'] = True
                result['veto_reason'] = f"Gann Exhaustion: {gann['recommendation']}"
                return result
        
        # 5. Tesla Vortex (3-6-9)
        tesla_candles = self.tesla.count_consecutive_candles(df_m5)
        if tesla_candles['count'] >= 9:
            result['veto'] = True
            result['veto_reason'] = tesla_candles['warning']
            return result
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


# Create singleton instance
_laplace_instance: Optional[LaplaceDemonCore] = None

def get_laplace_demon(symbol: str = "GBPUSD") -> LaplaceDemonCore:
    """Get or create the Laplace Demon instance."""
    global _laplace_instance
    if _laplace_instance is None or _laplace_instance.symbol != symbol:
        _laplace_instance = LaplaceDemonCore(symbol)
    return _laplace_instance
