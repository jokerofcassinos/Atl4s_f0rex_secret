
import logging
import numpy as np
from typing import Dict, Any
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("QuantSwarm")

class QuantSwarm(SubconsciousUnit):
    """
    Mathematical Probability Engine.
    Role: Apply Physics/Stats to validate moves.
    """
    def __init__(self):
        super().__init__("Quant_Swarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 100:
            return None
            
        # 1. Z-Score (Mean Reversion)
        # Is price statistically overextended?
        close = df_m5['close']
        window = 50
        roll_mean = close.rolling(window).mean().iloc[-1]
        roll_std = close.rolling(window).std().iloc[-1]
        price = close.iloc[-1]
        
        z_score = (price - roll_mean) / (roll_std + 1e-9)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Extreme Extension -> Reversion Likely
        if z_score > 3.0:
            signal = "SELL"
            confidence = 85.0
            reason = f"Z-Score Extreme (+{z_score:.2f}) - Statistical Reversion"
        elif z_score < -3.0:
            signal = "BUY"
            confidence = 85.0
            reason = f"Z-Score Extreme ({z_score:.2f}) - Statistical Reversion"
            
        # 2. Volatility Breakout (Expansion)
        # If Z-Score is just starting to expand (e.g. 1.5) AND volume is high -> Continuation
        
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'reason': reason, 'z_score': z_score}
            )
            
        return None
