import asyncio
import pandas as pd
import numpy as np
import ta # Biblioteca de Análise Técnica
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from enum import Enum

# --- MODULE IMPORTS ---
from signals.timing import QuarterlyTheory, M8FibonacciSystem
from signals.structure import SMCAnalyzer
from analysis.trend_architect import TrendArchitect
from signals.momentum import MomentumAnalyzer
from signals.volatility import VolatilityAnalyzer
from analysis.m8_fibonacci_system import M8FibonacciSystem
from analysis.swarm.vortex_swarm import VortexSwarm

# --- OMNI & HEISENBERG ---
from core.agi.big_beluga.snr_matrix import SNRMatrix
from core.agi.microstructure.flux_heatmap import FluxHeatmap
from analysis.quantum_core import QuantumCore
from core.agi.active_inference.free_energy import FreeEnergyMinimizer
from analysis.chaos_engine import ChaosEngine

# --- LEGION ELITE ---
from analysis.swarm.time_knife_swarm import TimeKnifeSwarm
from analysis.swarm.physarum_swarm import PhysarumSwarm
from analysis.swarm.event_horizon_swarm import EventHorizonSwarm
from analysis.swarm.overlord_swarm import OverlordSwarm

# AGI Components (for initialization)
from core.holographic_memory import HolographicMemory
from core.agi.infinite_why_engine import InfiniteWhyEngine

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
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    primary_signal: str = ""

class LaplaceDemonCore:
    # Horários de Alta Volatilidade (London + NY)
    ALLOWED_HOURS_SET = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17} 

    def __init__(self, symbol: str = "GBPUSD", contrarian_mode: bool = False):
        self.symbol = symbol
        self.daily_trades = {'date': None, 'count': 0}
        
        # Advanced AGI Components
        self.holographic_memory = HolographicMemory()
        self.infinite_why = InfiniteWhyEngine(
            max_depth=1,  # REDUCED from 4 for backtest performance
            max_branches=6,  # REDUCED from 12
            parallel_workers=1,  # REDUCED from 2
            enable_meta_reasoning=False  # DISABLED for speed
        )
        # BASIC MODULES
        self.quarterly = QuarterlyTheory()
        self.m8_fib = M8FibonacciSystem()
        self.trend_architect = TrendArchitect(symbol=symbol)
        self.smc = SMCAnalyzer()
        self.momentum = MomentumAnalyzer()
        self.volatility = VolatilityAnalyzer()
        self.vortex = VortexSwarm()
        
        # ADVANCED MODULES
        self.snr_matrix = SNRMatrix()
        self.heatmap = FluxHeatmap()
        self.quantum = QuantumCore()
        self.free_energy = FreeEnergyMinimizer()
        self.chaos = ChaosEngine()
        
        # LEGION ELITE
        self.time_knife = TimeKnifeSwarm()
        self.physarum = PhysarumSwarm()
        self.event_horizon = EventHorizonSwarm()
        self.overlord = OverlordSwarm()
        
        self.last_prediction = None
        logger.info("SYSTEM ONLINE: Laplace Sniper Protocol (70% WR Target)")

    async def analyze(self,
                df_m1: pd.DataFrame,
                df_m5: pd.DataFrame,
                df_h1: pd.DataFrame,
                df_h4: pd.DataFrame,
                current_time: datetime = None,
                current_price: float = None,
                **kwargs) -> LaplacePrediction:
        
        if current_time is None: current_time = datetime.now()
        if current_price is None and df_m5 is not None: current_price = df_m5['close'].iloc[-1]
        
        # 1. PERCEPTION (Síncrona)
        structure_data = self.smc.analyze(df_m5, current_price)
        trend_context = self.trend_architect.analyze(df_m5, df_h1, df_h4, None)
        quarterly = self.quarterly.analyze(current_time, df_m5)
        
        # 2. LEGION PARALLEL PROCESSING (Async)
        swarm_ctx = {
            'df_m1': df_m1, 'df_m5': df_m5, 'df_h1': df_h1, 
            'tick': {'bid': current_price, 'ask': current_price},
            'data_map': {'M5': df_m5, 'M1': df_m1}
        }
        
        tasks = [
            self.time_knife.process(swarm_ctx),
            self.event_horizon.process(swarm_ctx),
            self.physarum.process(swarm_ctx),
            self.overlord.process(swarm_ctx)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        legion_intel = {
            'knife': results[0] if not isinstance(results[0], Exception) else None,
            'horizon': results[1] if not isinstance(results[1], Exception) else None,
            'physarum': results[2] if not isinstance(results[2], Exception) else None,
            'overlord': results[3] if not isinstance(results[3], Exception) else None
        }

        # 3. SYNTHESIS (Decisão Central - PROTOCOLO SNIPER)
        decision = self._synthesize_agi_decision(
            df_m5=df_m5,
            current_price=current_price,
            trend_context=trend_context,
            legion_intel=legion_intel
        )
        
        # 4. EXECUTION
        prediction = LaplacePrediction(
            execute=decision['execute'], 
            direction=decision['direction'], 
            confidence=decision['confidence'],
            strength=SignalStrength.STRONG,
            reasons=decision['reasons'],
            primary_signal=decision['setup_type']
        )
        
        if prediction.execute:
            # Risco Conservador para construir a conta (1-2%)
            # Aumentamos apenas se a confiança for extrema
            prediction.risk_pct = 5.0 if prediction.confidence > 85 else 2.0
            
            # Dynamic SL/TP (Protocolo Sniper: Alvos Curtos)
            vol = self.volatility.analyze(df_m5)
            
            # ✅ BACKTEST FIX: VolatilityState is a dataclass, not a dict
            vol_regime = vol.get('regime')  # Returns VolatilityState object
            if vol_regime and hasattr(vol_regime, 'atr'):
                atr = vol_regime.atr * 10000  # Convert to pips
            else:
                atr = 20.0  # Default 20 pips if analysis fails
            
            sl_tp = self._calculate_sl_tp(prediction.direction, current_price, atr, structure_data, decision['setup_type'])
            
            prediction.sl_pips = sl_tp['sl_pips']
            prediction.tp_pips = sl_tp['tp_pips']
            prediction.sl_price = sl_tp['sl']
            prediction.tp_price = sl_tp['tp']
            
        self.last_prediction = prediction
        return prediction

    def _synthesize_agi_decision(self, df_m5: pd.DataFrame, current_price: float, trend_context: Dict, legion_intel: Dict) -> Dict:
        """
        THE SNIPER BRAIN
        Focado em Alta Probabilidade (Trend Pullback).
        """
        decision = {
            'execute': False, 'direction': 'WAIT', 'confidence': 0, 
            'reasons': [], 'setup_type': 'None', 'magnetic_target': None
        }
        
        if df_m5 is None or len(df_m5) < 200: return decision

        # --- 1. INDICADORES TÉCNICOS (O Motor Sniper) ---
        close = df_m5['close']
        
        # EMA 200 (A Muralha da Tendência)
        ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator().iloc[-1]
        
        # RSI 14 (O Gatilho)
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        
        # --- 2. LÓGICA DE EXECUÇÃO ---
        
        # SETUP A: SNIPER BULL (Compra na Tendência)
        # Preço > EMA200 (Tendência de Alta) E RSI < 40 (Oversold - Pullback)
        # PHASE 2: Relaxed from 35 to 40 for more signals
        if current_price > ema200:
            if rsi < 40:
                decision['execute'] = True
                decision['direction'] = "BUY"
                decision['confidence'] = 80
                decision['setup_type'] = "SNIPER_PULLBACK_BUY"
                decision['reasons'].append(f"Trend UP (Price > EMA200) + RSI Oversold ({rsi:.1f})")
        
        # SETUP B: SNIPER BEAR (Venda na Tendência)
        # Preço < EMA200 (Tendência de Baixa) E RSI > 60 (Overbought - Pullback)
        # PHASE 2: Relaxed from 65 to 60 for more signals
        elif current_price < ema200:
            if rsi > 60:
                decision['execute'] = True
                decision['direction'] = "SELL"
                decision['confidence'] = 80
                decision['setup_type'] = "SNIPER_PULLBACK_SELL"
                decision['reasons'].append(f"Trend DOWN (Price < EMA200) + RSI Overbought ({rsi:.1f})")

        # --- 3. CONFIRMAÇÃO DA LEGIÃO (Boost) ---
        knife = legion_intel.get('knife')
        horizon = legion_intel.get('horizon')
        
        if decision['execute']:
            # Se a Legião concorda, aumentamos a mão
            if knife and knife.signal_type == decision['direction']:
                decision['confidence'] = 95
                decision['reasons'].append("Legion TimeKnife Confirm")
            if horizon and horizon.signal_type == decision['direction']:
                decision['confidence'] = 90
                decision['reasons'].append("Legion EventHorizon Confirm")
                
        # --- 4. OVERRIDE DE EMERGÊNCIA (TimeKnife Scalp) ---
        # Se não temos setup Sniper, mas o TimeKnife vê um pico absurdo (Reversão M1)
        if not decision['execute']:
            if knife and knife.confidence > 85:
                decision['execute'] = True
                decision['direction'] = knife.signal_type
                decision['confidence'] = 85
                decision['setup_type'] = "LEGION_KNIFE_SCALP"
                decision['reasons'].append(f"TimeKnife Volatility Spike ({knife.meta_data.get('reason')})")

        return decision

    def _calculate_sl_tp(self, direction, price, atr, structure, setup_type):
        pip = 0.0001
        
        # SL TÉCNICO: Baseado em Estrutura Recente (Swing Point)
        # Para 70% WR, o SL não pode ser curto demais.
        sl_pips = max(20, atr * 2.0) # Mínimo 20 pips para respirar
        
        # TP ALVO: 1:1 a 1.5:1 para garantir o Win Rate
        # Se for Sniper Pullback, buscamos a continuação da tendência (30-40 pips)
        # Se for Scalp (Knife), pegamos rápido (10-15 pips)
        
        if "SNIPER" in setup_type:
            tp_pips = sl_pips * 1.2 # R:R 1.2 conservador para bater a meta de WR
        else:
            tp_pips = 10.0 # Scalp rápido fixo
            
        sl = price - (sl_pips*pip) if direction == 'BUY' else price + (sl_pips*pip)
        tp = price + (tp_pips*pip) if direction == 'BUY' else price - (tp_pips*pip)
        return {'sl': sl, 'tp': tp, 'sl_pips': sl_pips, 'tp_pips': tp_pips}

    def _resample_to_m8(self, df_m1, df_m5):
        return None

# Singleton
_laplace_instance: Optional[LaplaceDemonCore] = None
def get_laplace_demon(symbol: str = "GBPUSD") -> LaplaceDemonCore:
    global _laplace_instance
    if _laplace_instance is None or _laplace_instance.symbol != symbol:
        _laplace_instance = LaplaceDemonCore(symbol)
    return _laplace_instance
