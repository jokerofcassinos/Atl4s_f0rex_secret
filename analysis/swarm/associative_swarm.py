
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal
from core.hyper_dimensional import HDCEncoder, HyperVector

logger = logging.getLogger("AssociativeSwarm")

class AssociativeSwarm(SubconsciousUnit):
    """
    The Librarian.
    Uses Hyperdimensional Computing (HDC) to recognize market states via Vector Similarity.
    """
    def __init__(self):
        super().__init__("Associative_Swarm")
        self.encoder = HDCEncoder()
        
        # Long-Term Concepts (Prototypes)
        self.concepts = {
            'BULL': HyperVector(), # Sum of all Bullish states
            'BEAR': HyperVector(), # Sum of all Bearish states
            'CHAOS': HyperVector() # Sum of Noise states
        }
        
        # Buffer for delayed learning (outcome verification)
        self.memory_buffer = [] # (Vector, timestamp, price)

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        last = df_m5.iloc[-1]
        
        # 1. Normalize Inputs for Encoder (0-100 Scale)
        # Close % Rank in last 20
        recent_closes = df_m5['close'].iloc[-20:].values
        close_rank = (np.searchsorted(np.sort(recent_closes), last['close']) / 20.0) * 100
        
        # Vol % Rank
        recent_vols = df_m5['volume'].iloc[-20:].values
        vol_rank = (np.searchsorted(np.sort(recent_vols), last['volume']) / 20.0) * 100
        
        # RSI (Assume 50 if not calculated)
        rsi = 50 
        
        # 2. Encode Current State
        current_hv = self.encoder.encode_state(close_rank, vol_rank, rsi)
        
        # 3. Learning (Delayed)
        # Check buffer for past states that can now be labeled
        # If State T-5 led to Price Increase, add T-5 Vector to BULL Concept.
        if len(self.memory_buffer) > 5:
            past_hv, _, past_price = self.memory_buffer.pop(0)
            
            price_delta = last['close'] - past_price
            if price_delta > 0.05: # Significant Up
                self.concepts['BULL'] = self.concepts['BULL'].bundle(past_hv)
            elif price_delta < -0.05: # Significant Down
                self.concepts['BEAR'] = self.concepts['BEAR'].bundle(past_hv)
                
        self.memory_buffer.append((current_hv, 0, last['close']))
        
        # 4. Inference
        sim_bull = current_hv.similarity(self.concepts['BULL'])
        sim_bear = current_hv.similarity(self.concepts['BEAR'])
        
        # Normalize Similarity (since it's -1 to 1 and starts random near 0)
        # We look for relative dominance
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        diff = sim_bull - sim_bear
        
        # HDC Similarity is subtle. 0.05 diff is significant for orthogonal spaces.
        if diff > 0.02:
            signal = "BUY"
            confidence = 65 + (diff * 100)
            reason = f"HDC Match: Bull ({sim_bull:.2f}) > Bear ({sim_bear:.2f})"
        elif diff < -0.02:
            signal = "SELL"
            confidence = 65 + (abs(diff) * 100)
            reason = f"HDC Match: Bear ({sim_bear:.2f}) > Bull ({sim_bull:.2f})"
            
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=min(95, confidence),
                timestamp=0,
                meta_data={'hdc_sim': diff}
            )
        
        return None
