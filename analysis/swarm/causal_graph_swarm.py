
import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("CausalGraphSwarm")

class CausalGraphSwarm(SubconsciousUnit):
    """
    The Architect.
    maintains a Causal DAG to distinguish Correlation from Causation.
    """
    def __init__(self):
        super().__init__("Causal_Graph_Swarm")
        # Simplified DAG Structure: Force -> Movement
        # Nodes: Volume -> Volatility -> Price
        self.edges = {
            ('volume', 'volatility'): 0.5,
            ('volatility', 'price_change'): 0.5
        }

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Update Causal Weights (Granger Proxy)
        # Does Lagged Volume predict Volatility?
        vol = df_m5['volume']
        # Approx volatility as High-Low range
        volatility = df_m5['high'] - df_m5['low']
        price_change = df_m5['close'].diff().abs()
        
        # Simple Correlation of lags as proxy for Causality (Fast)
        # Real-time Granger is too slow python-side without acceleration
        corr_vol_volat = vol.shift(1).corr(volatility)
        corr_volat_price = volatility.shift(1).corr(price_change)
        
        self.edges[('volume', 'volatility')] = corr_vol_volat
        self.edges[('volatility', 'price_change')] = corr_volat_price
        
        # 2. Causality Check for Current State
        # If we see a Price Spike, is it CAUSED by the graph?
        last_vol = vol.iloc[-1]
        avg_vol = vol.iloc[-20:].mean()
        
        last_volatility = volatility.iloc[-1]
        
        is_volume_spike = last_vol > avg_vol * 1.5
        is_volatility_spike = last_volatility > volatility.iloc[-20:].mean() * 1.5
        
        # Logic: If Price moved, but Volume/Volatility didn't preceed/accompany it, it's 'Acausal' (Noise/Manipulation)
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        if is_volume_spike and is_volatility_spike:
            # The "Force" is present. The movement is Causal.
            # We endorse the trend.
            trend = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-1]
            if trend > 0:
                signal = "BUY"
                confidence = 80 + (corr_vol_volat * 10) # Boost by causal strength
                reason = "Causal Validation: vol->price link strong"
            elif trend < 0:
                signal = "SELL"
                confidence = 80 + (corr_vol_volat * 10)
                reason = "Causal Validation: vol->price link strong"
        else:
            # Weak Causality.
            # If price moved heavily without this, it might be a Stop Hunt (which LiquiditySwarm handles)
            # or just noise.
            pass
            
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'edges': str(self.edges)}
            )
            
        return None
