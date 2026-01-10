
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("VortexSwarm")

class VortexSwarm(SubconsciousUnit):
    """
    Phase 66: The Vortex Swarm (Trap Detector).
    
    Specializes in analyzing 'Candle Geometry' to detect Market Maker Traps.
    It does not care about Trend or Momentum. It cares about REJECTION.
    
    Logic:
    - Analyzes the shape of the current M1/M5 candle in real-time.
    - Large Upper Wick = Rejection (Bull Trap) -> Signal SELL/VETO BUY.
    - Large Lower Wick = Rejection (Bear Trap) -> Signal BUY/VETO SELL.
    - Uses 'Vortex Depth' (Volume * Wick Size) to estimate trap magnitude.
    """
    def __init__(self):
        super().__init__("Vortex_Swarm")

    def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5') # Switch to M5 for reliability
        tick = context.get('tick')
        
        if df_m5 is None or len(df_m5) < 20: return None
        
        # Calculate ATR (Volatility Threshold)
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        closes = df_m5['close'].values
        
        # Simple ATR-14
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        atr = np.mean(tr[-14:])
        
        # Analyze the LAST closed M5 candle
        last_closed = df_m5.iloc[-1]
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # 1. Analyze Last Closed M5
        high = last_closed['high']
        low = last_closed['low']
        close = last_closed['close']
        open_p = last_closed['open']
        
        total_range = high - low
        
        # PLANCK SCALE FILTER (Phase 72)
        # Verify if the action is "Macroscopic" enough to matter.
        # If the candle is tiny (Noise), ignore its geometry.
        if total_range < (atr * 0.4): 
            return None # Quantum Foam (Noise)
            
        upper_wick = high - max(open_p, close)
        lower_wick = min(open_p, close) - low
        body = abs(close - open_p)
        
        # Ratio of Wicks to Range
        upper_ratio = upper_wick / total_range
        lower_ratio = lower_wick / total_range
        
        # TRAP DETECTION (M5)
        
        # A. Bull Trap (Shooting Star / Grave Doji)
        # Price tried to go up (Long Wick) but closed near low.
        if upper_ratio > 0.45 and (close < open_p or body < (total_range * 0.2)):
             signal = "SELL"
             confidence = 88.0
             reason = f"VORTEX [M5]: Bull Trap (Upper Wick {upper_ratio:.2f}). Rejection from High."
             
        # B. Bear Trap (Hammer / Dragonfly)
        # Price tried to go down (Long Wick) but closed near high.
        elif lower_ratio > 0.45 and (close > open_p or body < (total_range * 0.2)):
             signal = "BUY"
             confidence = 88.0
             reason = f"VORTEX [M5]: Bear Trap (Lower Wick {lower_ratio:.2f}). Rejection from Low."
             
        # 2. Live Rejection (Current Tick vs M5 High) -> Intrabar Trap
        # If Current Price is dropping fast from M5 High...
        # Skipped for now to strictly trust Closed M5 signals (less noise).
        
        if signal != "WAIT":
            
            upper_wick = high - max(open_p, close)
            lower_wick = min(open_p, close) - low
            body = abs(close - open_p)
            
            # Trap Detection 1: The "Shooting Star" / "Pinbar" (Bull Trap)
            # Upper Wick > 2x Body AND Upper Wick > 40% of Range
            if upper_wick > (body * 2.0) and (upper_wick / total_range) > 0.4:
                # Strong Rejection from High
                signal = "SELL"
                confidence = 85.0
                reason = "VORTEX: Shooting Star Pattern (Bull Trap) Detected."
                
            # Trap Detection 2: The "Hammer" (Bear Trap)
            elif lower_wick > (body * 2.0) and (lower_wick / total_range) > 0.4:
                # Strong Rejection from Low
                signal = "BUY" # Reversal Up
                confidence = 85.0
                reason = "VORTEX: Hammer Pattern (Bear Trap) Detected."
                
        # Real-time wick check (if current price is way below High of current session)
        # This requires tracking High since Open. Harder without state.
        
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'reason': reason}
            )
            
        return None
