
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("SentimentSwarm")

class SentimentSwarm(SubconsciousUnit):
    """
    The Empath.
    Detects Fear, Greed, Panic, and FOMO.
    """
    def __init__(self):
        super().__init__("Sentiment_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Pain Index (Price vs VWAP)
        # Approximation of VWAP for recent session
        df_m5['tp'] = (df_m5['high'] + df_m5['low'] + df_m5['close']) / 3
        df_m5['cum_vol'] = df_m5['volume'].cumsum()
        df_m5['cum_vol_price'] = (df_m5['tp'] * df_m5['volume']).cumsum()
        vwap = df_m5['cum_vol_price'] / df_m5['cum_vol']
        
        last_price = df_m5['close'].iloc[-1]
        last_vwap = vwap.iloc[-1]
        
        deviation = (last_price - last_vwap) / last_vwap * 100
        
        # 2. Acceleration (Second derivative of price)
        # panic = velocity of price change increasing downwards
        velocity = df_m5['close'].diff()
        acceleration = velocity.diff()
        acc_val = acceleration.iloc[-1]
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        # Logic:
        # If Price is WAY below VWAP (> 2% deviation) and Acceleration is stabilizing (Panic exhaustion) -> BUY
        # If Price is WAY above VWAP (> 2%) and Acceleration slowing -> SELL (Top)
        
        if deviation < -1.5: # Deep underwater (Pain)
            if acc_val > 0: # Deceleration of the drop (Turnaround)
                signal = "BUY"
                confidence = 85
                reason = f"Max Pain Reversal: Dev {deviation:.1f}% + Deceleration"
            else:
                # Still crashing
                pass
                
        elif deviation > 1.5: # Euphoria
            if acc_val < 0: # Deceleration of the pump
                signal = "SELL"
                confidence = 85
                reason = f"Max Euphoria Exhaustion: Dev {deviation:.1f}%"
                
        # 3. FOMO Spike
        # High Velocity Up + High Volume
        
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'deviation': deviation, 'acceleration': acc_val}
            )
            
        return None
