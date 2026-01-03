
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("TrendingSwarm")

class TrendingSwarm(SubconsciousUnit):
    """
    Trend Consensus Engine.
    Role: Identify the Macro and Micro River (Flow).
    Concepts: Deep Causality, Multi-Timeframe Alignment.
    """
    def __init__(self):
        super().__init__("Trending_Swarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        market_state = context.get('market_state', {}) # From OpportunityFlow or Architect
        
        if df_m5 is None or len(df_m5) < 50:
            return None

        # 1. H1 River Check (Macro Causality)
        # If H1 is bullish, we have a "Source of Flow".
        target_river = market_state.get('river', 0)
        
        # 2. M5 Micro Structure (Immediate Pressure)
        # Check EMA Cross + ADX
        close = df_m5['close'].iloc[-1]
        ema_short = df_m5['close'].ewm(span=9).mean().iloc[-1]
        ema_long = df_m5['close'].ewm(span=21).mean().iloc[-1]
        
        micro_trend = 0
        if ema_short > ema_long: micro_trend = 1
        elif ema_short < ema_long: micro_trend = -1
        
        # 3. Deep Causality: Volume Validation
        # "Why is price moving?" -> If Volume is rising with price = Valid.
        # If Price rising but Volume falling = Divergence (Fake).
        vol_ma = df_m5['volume'].rolling(20).mean().iloc[-1]
        curr_vol = df_m5['volume'].iloc[-1]
        
        volume_support = False
        if curr_vol > vol_ma:
            volume_support = True
            
        # 4. Synthesis
        confidence = 0.0
        signal = "WAIT"
        
        # Alignment: Macro (River) == Micro (EMA)
        if target_river == 1 and micro_trend == 1:
            signal = "BUY"
            confidence = 80.0
            if volume_support: confidence += 10.0
            
        elif target_river == -1 and micro_trend == -1:
            signal = "SELL"
            confidence = 80.0
            if volume_support: confidence += 10.0
            
        # Counter-Trend (Pullback) - Lower Confidence
        elif target_river == 1 and micro_trend == -1:
            # Pullback in Uptrend?
            # Only if price is touching specific support... simpler logic for now:
            signal = "WAIT" # Let Sniper handle counter-trend entries
            confidence = 0.0
            
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'reason': f"Trend Alignment (River={target_river}, Micro={micro_trend})"}
            )
            
        return None
