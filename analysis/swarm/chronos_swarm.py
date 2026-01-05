import logging
import asyncio
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.twelfth_eye import TwelfthEye

logger = logging.getLogger("ChronosSwarm")

class ChronosSwarm(SubconsciousUnit):
    """
    The Temporal Orchestrator.
    Wraps The Twelfth Eye to provide Time-based signals.
    """
    def __init__(self):
        super().__init__("Chronos_Swarm")
        self.engine = TwelfthEye()
        self.last_sync_idx = 0.0
        
    async def process(self, context) -> SwarmSignal:
        data_map = context.get('data_map')
        if not data_map: return None
        
        # 1. Analyze Time Structure
        result = self.engine.analyze(data_map)
        
        sync_index = result.get('synchronization_index', 0.0)
        crystal = result.get('time_crystal')
        
        # 2. Decision Logic
        signal_type = "WAIT"
        confidence = 0.0
        meta = result
        
        # Condition A: The Singularity (Perfect Sync)
        if sync_index > 0.85:
            # Everything is moving together.
            # We don't know direction solely from Sync, but we know Magnitude is coming.
            # We check the dominant trend of the H1 to bias.
            
            df_h1 = data_map.get('H1')
            if df_h1 is not None:
                sma20 = df_h1['close'].rolling(20).mean().iloc[-1]
                price = df_h1['close'].iloc[-1]
                
                if price > sma20:
                    signal_type = "BUY"
                    confidence = 85.0 + (sync_index * 10) # Max 95
                else:
                    signal_type = "SELL"
                    confidence = 85.0 + (sync_index * 10)
                    
            meta['desc'] = f"TEMPORAL SINGULARITY (Sync: {sync_index:.2f})"
            
        # Condition B: Time Crystal (Broken Symmetry)
        elif crystal and crystal['is_crystal']:
            # Period doubling often precedes chaos (Trend change).
            # If we are in a trend, this signals Reversal.
            signal_type = "EXIT_ALL" # Safety First?
            confidence = 80.0
            meta['desc'] = "TIME CRYSTAL DETECTED: Phase Transition Imminent"
            
            # Or if finding opportunities
            # "Chaos is a ladder" - Petyr Baelish
            # If sub-harmonic is strong, it means the market is vibrating at 2 frequencies.
            # This is a specialized state.
            
        if signal_type != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=0,
                meta_data=meta
            )
            
        return None
