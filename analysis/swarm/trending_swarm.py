
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

        # --- NEW: PARABOLIC & EXHAUSTION SAFETY ---
        # Calculate ATR (Approximate 14 period)
        high_low = df_m5['high'] - df_m5['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr == 0: atr = 0.0010 # Default fallback
        
        # A. Parabolic Check: Price moved > 3x ATR in last 3 candles?
        accel = abs(df_m5['close'].iloc[-1] - df_m5['close'].iloc[-4])
        is_parabolic = accel > (3.0 * atr)
        
        # B. Climax / Churn: High Volume (> 2x) but Small Body (< 0.5x ATR)
        body = abs(df_m5['close'].iloc[-1] - df_m5['open'].iloc[-1])
        is_churn = (curr_vol > 2.0 * vol_ma) and (body < 0.5 * atr)
        
        # C. Volume Climax: Extreme Volume (> 3x)
        is_climax = curr_vol > 3.0 * vol_ma
        
        exhaustion_state = False
        exhaustion_reason = ""
        
        if is_parabolic:
             exhaustion_state = True
             exhaustion_reason = "PARABOLIC_EXTENTION"
        elif is_churn:
             exhaustion_state = True
             exhaustion_reason = "VOLUME_CHURN"
        elif is_climax:
             exhaustion_state = True
             exhaustion_reason = "VOLUME_CLIMAX"

        # 4. Synthesis
        confidence = 0.0
        signal = "WAIT"
        
        # Alignment: Macro (River) == Micro (EMA)
        if target_river == 1 and micro_trend == 1:
            if exhaustion_state:
                signal = "WAIT" # Forced Cool-down
                confidence = 0.0
                logger.warning(f"TREND EXHAUSTION: Bullish Signal Blocked via {exhaustion_reason}")
            else:
                signal = "BUY"
                confidence = 80.0
                if volume_support: confidence += 10.0
            
        elif target_river == -1 and micro_trend == -1:
            if exhaustion_state:
                signal = "WAIT" # Forced Cool-down
                confidence = 0.0
                logger.warning(f"TREND EXHAUSTION: Bearish Signal Blocked via {exhaustion_reason}")
            else:
                signal = "SELL"
                confidence = 80.0
                if volume_support: confidence += 10.0
            
        # Counter-Trend (Pullback) - Lower Confidence
        elif target_river == 1 and micro_trend == -1:
            # Pullback in Uptrend?
            # Only if price is touching specific support... simpler logic for now:
            signal = "WAIT" # Let Sniper handle counter-trend entries
            confidence = 0.0
            
        if signal != "WAIT" or exhaustion_state:
            meta = {'reason': f"Trend Alignment (River={target_river}, Micro={micro_trend})"}
            if exhaustion_state:
                meta['exhaustion'] = True
                meta['exhaustion_reason'] = exhaustion_reason
                meta['reason'] = f"BLOCKED: {exhaustion_reason}"
            
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data=meta
            )
            
        return None
