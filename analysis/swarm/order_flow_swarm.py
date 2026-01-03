
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("OrderFlowSwarm")

class OrderFlowSwarm(SubconsciousUnit):
    """
    The Tape Reader.
    Analyzes Volume Delta and Absorption to find Hidden Walls.
    """
    def __init__(self):
        super().__init__("Order_Flow_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        if df_m1 is None or len(df_m1) < 20: return None
        
        last = df_m1.iloc[-1]
        prev = df_m1.iloc[-2]
        
        # 1. Infer Delta (Proxy for Buying/Selling Pressure)
        # If Close > Open, assume Volume is heavily Buy-side
        # We can refine this: (High - Low) range vs Volume.
        
        body = abs(last['close'] - last['open'])
        range_ = last['high'] - last['low']
        volume = last['volume']
        
        # Avoid division by zero
        if range_ == 0: range_ = 0.00001
        
        # Effort (Volume) vs Result (Body)
        # Absorption Ratio: Volume / Body. High Ratio = Absorption.
        absorption_ratio = 0
        if body > 0:
            absorption_ratio = volume / body
        else:
            absorption_ratio = volume # Massive absorption (Doji)
            
        # Normalize Ratio (Simple baseline)
        avg_vol = df_m1['volume'].iloc[-20:].mean()
        avg_body = abs(df_m1['close'] - df_m1['open']).iloc[-20:].mean()
        baseline_ratio = avg_vol / (avg_body + 0.00001)
        
        is_absorbing = absorption_ratio > (baseline_ratio * 3.0) # 3x normal density
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        # Logic: Absorption at Highs = Bearish Wall. Absorption at Lows = Bullish Wall.
        
        if is_absorbing:
            # Check context: Where did it happen?
            # At a Swing Low?
            if last['close'] < df_m1['low'].iloc[-10:].min() * 1.0005: 
                # Preventing price from going lower
                signal = "BUY"
                confidence = 80
                reason = "Passive Buyer Wall (Absorption)"
            
            # At a Swing High?
            elif last['close'] > df_m1['high'].iloc[-10:].max() * 0.9995:
                # Preventing price from going higher
                signal = "SELL"
                confidence = 80
                reason = "Passive Seller Wall (Absorption)"
                
        # 2. Delta Divergence (Price Up, Delta Down) - Hard without real tick data, using CVD proxy
        # Placeholder for CVD Logic
        
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'absorption_ratio': absorption_ratio}
            )
            
        return None
