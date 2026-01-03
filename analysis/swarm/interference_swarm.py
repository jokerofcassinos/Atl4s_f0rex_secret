
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("InterferenceSwarm")

class InterferenceSwarm(SubconsciousUnit):
    """
    The Quantum Wave Function (Interference).
    Phase 45 Innovation.
    Logic:
    1. Models Price and Volume as Waveforms.
    2. Concept: Constructive vs Destructive Interference.
       - Constructive: Price Rise + Volume Rise (High Amplitude).
       - Destructive: Price Rise + Volume Fall (Divergence / Damping).
    3. Math: Phase Difference Calculation.
    """
    def __init__(self):
        super().__init__("InterferenceSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        # 1. Normalize Trends to Waves (-1 to 1)
        # Use simple Moving Average deviation
        closes = df['close'].values
        # Handle different volume column names (yfinance 'volume' vs mt5 'tick_volume')
        if 'tick_volume' in df.columns:
            volumes = df['tick_volume'].values
        elif 'volume' in df.columns:
            volumes = df['volume'].values
        else:
            return None # No volume data
        
        # Smooth them
        period = 10
        price_ma = pd.Series(closes).rolling(period).mean().values
        vol_ma = pd.Series(volumes).rolling(period).mean().values
        
        # Wave = Value - MA
        # Normalized by MA
        price_wave = (closes - price_ma) / (price_ma + 1e-9)
        vol_wave = (volumes - vol_ma) / (vol_ma + 1e-9)
        
        # Focus on the last period
        p_last = price_wave[-1]
        v_last = vol_wave[-1]
        
        # 2. Interference Calculation
        # Constructive: Both same sign.
        # Destructive: Opposite signs.
        
        # Amplitude = P + V
        amp = p_last + v_last
        
        signal = "WAIT"
        confidence = 0.0
        
        # Scale: Typical wave amp is 0.001 (0.1%). 
        # Normalize roughly.
        
        # LOGIC:
        # Bullish Interference: Price is ABOVE MA (Positive) AND Volume is ABOVE MA (Positive)
        if p_last > 0 and v_last > 0:
            # Constructive Bullish
            signal = "BUY"
            # Magnify confidence by amplitude
            confidence = min(100, (p_last * 10000) * 5 + 30) 
            
        # Bearish Interference: Price is BELOW MA (Negative) AND Volume is ABOVE MA (Positive)
        # Wait, usually high volume confirms trend.
        # So for Bearish Trend, we want Price < 0 AND Volume > 0?
        # That means "Strong Selling".
        # If Price < 0 and Volume < 0 (Low Volume), it's weak selling (drift).
        
        elif p_last < 0 and v_last > 0:
            # Constructive Bearish (Strong Downward Push)
            signal = "SELL"
            confidence = min(100, (abs(p_last) * 10000) * 5 + 30)
            
        # Divergence (Price Up, Vol Down) -> Destructive
        # We don't trade divergence here, we WAIT.
        
        if signal != "WAIT":
             return SwarmSignal(
                source="InterferenceSwarm",
                signal_type=signal,
                confidence=min(100.0, confidence),
                timestamp=0,
                meta_data={
                    "price_wave": float(p_last),
                    "vol_wave": float(v_last),
                    "interference_type": "Constructive"
                }
            )
            
        return None
