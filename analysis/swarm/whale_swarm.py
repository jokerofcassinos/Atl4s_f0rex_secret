
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("WhaleSwarm")

class WhaleSwarm(SubconsciousUnit):
    """
    The Heavyweights.
    Agents tracking Institutional Footprints (Smart Money).
    """
    def __init__(self):
        super().__init__("Whale_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Iceberg Agent
        ice_score, ice_dir = self._detect_iceberg(df_m5)
        
        # 2. Smart Money Structure Agent (Simplified Imbalance)
        smc_score, smc_dir = self._detect_structure(df_m5)
        
        # Consensus
        final_dir = 0
        confidence = 0
        
        # If Iceberg and Structure align -> Whale Move
        if ice_dir != 0 and smc_dir != 0 and ice_dir == smc_dir:
            final_dir = ice_dir
            confidence = (ice_score + smc_score) / 2
            reason = f"Whale Confluence: Iceberg + Structure ({confidence:.1f})"
            
        # If Massive Iceberg -> Solo move
        elif ice_score > 80:
            final_dir = ice_dir
            confidence = ice_score
            reason = f"Whale Solo: Massive Iceberg ({ice_score:.1f})"
            
        if final_dir != 0:
            action = "BUY" if final_dir == 1 else "SELL"
            return SwarmSignal(
                source=self.name,
                signal_type=action,
                confidence=confidence,
                timestamp=0,
                meta_data={'reason': reason}
            )
            
        return None

    def _detect_iceberg(self, df):
        last = df.iloc[-1]
        
        # Vol checks
        vol_col = 'volume'
        avg_vol = df[vol_col].rolling(20).mean().iloc[-1]
        curr_vol = last[vol_col]
        
        # Range checks
        df['rng'] = df['high'] - df['low']
        avg_rng = df['rng'].rolling(20).mean().iloc[-1]
        curr_rng = df['rng'].iloc[-1]
        
        if avg_rng == 0 or avg_vol == 0: return 0, 0
        
        vol_ratio = curr_vol / avg_vol
        range_ratio = curr_rng / avg_rng
        
        # High Vol / Low Range
        if vol_ratio > 1.5 and range_ratio < 0.8:
            score = min(100, 50 * (vol_ratio / range_ratio))
            # Direction: Close relative to open
            direction = 1 if last['close'] > last['open'] else -1
            return score, direction
            
        return 0, 0

    def _detect_structure(self, df):
        # Placeholder for complex SMC (Order Block) logic
        # For now, check Break of Structure (Close > High[-2])
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish BOS
        if last['close'] > prev['high']:
             return 60, 1
        # Bearish BOS
        elif last['close'] < prev['low']:
             return 60, -1
             
        return 0, 0
