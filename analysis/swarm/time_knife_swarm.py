
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("TimeKnifeSwarm")

class TimeKnifeSwarm(SubconsciousUnit):
    """
    The Time Knife (formerly Quantum Grid / 13th Eye).
    Specialized for M1 Micro-structure scalping during "Knife" events (Spikes).
    Logic:
    1. Detects "Boom" candles (2.5 Sigma Move).
    2. Calculates Reversion Target (EMA 5).
    3. Fires a signal if RSI is extreme (Overbought/Oversold).
    """
    def __init__(self):
        super().__init__("TimeKnifeSwarm")
        self.last_candle_time = None

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        df_m5 = context.get('df_m5')
        tick_data = context.get('tick')

        if df_m1 is None or len(df_m1) < 20: return None

        # --- ANALYSIS ---
        last_candle = df_m1.iloc[-1]
        close = last_candle['close']
        
        # 1. Volatility Expansion (The Knife)
        # Check standard deviation of last 20 candles
        std_20 = df_m1['close'].rolling(20).std().iloc[-1]
        ma_20 = df_m1['close'].rolling(20).mean().iloc[-1]
        
        # Bollinger Extreme
        upper = ma_20 + (std_20 * 2.5)
        lower = ma_20 - (std_20 * 2.5)
        
        # 2. RSI Approximation
        delta = df_m1['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (gain/loss))) if loss > 0 else 50
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""

        # --- LOGIC ---
        # If Price Pierces 2.5 Sigma Band + RSI Extreme -> REVERSION TRADE
        
        if close > upper and rsi > 80:
            signal = "SELL"
            confidence = 90.0
            reason = f"Time Knife: 2.5 SD Break (Top) + RSI {rsi:.1f}"
            
        elif close < lower and rsi < 20:
             signal = "BUY"
             confidence = 90.0
             reason = f"Time Knife: 2.5 SD Break (Bottom) + RSI {rsi:.1f}"
             
        # Machine Gun Mode (Trend Continuation)
        # If Price is hugging the EMA 5 and moving fast, we join.
        else:
             ema5 = df_m1['close'].ewm(span=5).mean().iloc[-1]
             ema10 = df_m1['close'].ewm(span=10).mean().iloc[-1]
             
             # Bullish Flow
             if close > ema5 and ema5 > ema10 and rsi > 55 and rsi < 75:
                 signal = "BUY"
                 confidence = 65.0
                 reason = "Time Knife: Laminar Flow (Up)"
                 
             # Bearish Flow
             elif close < ema5 and ema5 < ema10 and rsi < 45 and rsi > 25:
                 signal = "SELL"
                 confidence = 65.0
                 reason = "Time Knife: Laminar Flow (Down)"

        return SwarmSignal(
            source="TimeKnifeSwarm",
            signal_type=signal,
            confidence=confidence,
            timestamp=0,
            meta_data={"reason": reason}
        )
