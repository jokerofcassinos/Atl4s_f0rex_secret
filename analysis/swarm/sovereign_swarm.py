
import logging
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.eighth_eye import EighthEye
import time

logger = logging.getLogger("SovereignSwarm")

class SovereignSwarm(SubconsciousUnit):
    """
    Eighth Eye Swarm Wrapper.
    The Sovereign: Fractal Coherence Logic.
    """
    def __init__(self):
        super().__init__("Sovereign_Swarm")
        self.engine = EighthEye()

    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        data_map = context.get('data_map')
        if not data_map: return None
        
        try:
            res = self.engine.deliberate(data_map)
            decision = res.get('decision', 'WAIT')
            score = res.get('score', 0)
            
            signal_type = "WAIT"
            if "BUY" in decision: signal_type = "BUY"
            elif "SELL" in decision: signal_type = "SELL"
            
            if signal_type != "WAIT":
                return SwarmSignal(
                    source=self.name,
                    signal_type=signal_type,
                    confidence=min(100.0, abs(score)),
                    timestamp=time.time(),
                    meta_data=res
                )
        except Exception as e:
             logger.debug(f"Sovereign Logic Error: {e}")
             
        return None
