
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("ArchitectSwarm")

class ArchitectSwarm(SubconsciousUnit):
    """
    The Architect (Strategic Commander).
    Replaces the legacy 'TenthEye'.
    Function:
    1. Determines Global Directive (ATTACK, DEFEND, SURVIVE) based on Chaos/Hurst.
    2. Outputs a Meta-Signal that doesn't just say BUY/SELL, but suggests INTENSITY.
    """
    def __init__(self):
        super().__init__("ArchitectSwarm")
        self.current_directive = "NEUTRAL"

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        data_map = context.get('data_map', {})
        df_m1 = data_map.get('M1')
        df_m5 = data_map.get('M5')
        df_h1 = data_map.get('H1')
        df_h4 = data_map.get('H4')
        tick_data = context.get('tick')

        if df_m5 is None or len(df_m5) < 50: return None

        # --- HTF BIAS (The Macro View) ---
        htf_bias = "NEUTRAL"
        if df_h4 is not None and not df_h4.empty:
            sma50_h4 = df_h4['close'].rolling(50).mean().iloc[-1]
            last_h4 = df_h4['close'].iloc[-1]
            if last_h4 > sma50_h4: htf_bias = "BULLISH"
            elif last_h4 < sma50_h4: htf_bias = "BEARISH"

        # --- REGIME CALCULATION (Micro View) ---
        # 1. Hurst Exponent (Trend Strength on M5)
        hurst = 0.5
        series = df_m5['close'].tail(50).values
        if len(series) > 20:
             # Fast Hurst Proxy
             lags = range(2, 20)
             tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
             poly = np.polyfit(np.log(lags), np.log(tau), 1)
             hurst = poly[0] * 2.0 

        # --- DIRECTIVE LOGIC ---
        directive = "BALANCED"
        
        # Confluence: If HTF is Bullish AND Hurst is high -> ATTACK
        if hurst > 0.60:
            directive = "AGGRESSIVE_TREND"
        elif hurst < 0.40:
            directive = "SNIPER_SCALP" # Mean reversion
            
        # Override: If HTF is strong, don't short the dip, buy it.
        # This nuance is handled in signal generation below.
        
        # Volatility Check
        atr = (df_m5['high'] - df_m5['low']).mean()
        curr_range = df_m5['high'].iloc[-1] - df_m5['low'].iloc[-1]
        
        if curr_range > (atr * 3.0):
             directive = "SURVIVAL" 
             
        self.current_directive = directive

        # --- SIGNAL GENERATION ---
        ema20 = df_m5['close'].ewm(span=20).mean().iloc[-1]
        price = tick_data.get('last', tick_data.get('bid'))
        
        signal = "WAIT"
        confidence = 0.0
        
        if directive == "AGGRESSIVE_TREND":
            if price > ema20:
                if htf_bias in ["BULLISH", "NEUTRAL"]:
                    signal = "BUY"
                    confidence = 85.0
            else:
                if htf_bias in ["BEARISH", "NEUTRAL"]:
                    signal = "SELL"
                    confidence = 85.0
                
        elif directive == "SNIPER_SCALP":
            # Counter-Trend
            delta = df_m5['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            if rsi > 70:
                # Only short overbought if we are NOT in a super Bullish HTF trend
                if htf_bias != "BULLISH": 
                    signal = "SELL"
                    confidence = 80.0
            elif rsi < 30:
                if htf_bias != "BEARISH":
                    signal = "BUY"
                    confidence = 80.0
        
        elif directive == "SURVIVAL":
            signal = "VETO" 
            confidence = 100.0

        reason = f"Regime: {directive} (Hurst {hurst:.2f}) | HTF: {htf_bias}"
        
        return SwarmSignal(
            source="ArchitectSwarm",
            signal_type=signal,
            confidence=confidence,
            timestamp=0,
            meta_data={"reason": reason, "directive": directive, "htf": htf_bias}
        )
