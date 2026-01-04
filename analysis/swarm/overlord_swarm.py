
import logging
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.seventh_eye import SeventhEye
import time

logger = logging.getLogger("OverlordSwarm")

class OverlordSwarm(SubconsciousUnit):
    """
    Seventh Eye Swarm Wrapper.
    The Overlord: Synthesis & Latent Space Logic.
    """
    def __init__(self):
        super().__init__("Overlord_Swarm")
        self.engine = SeventhEye()

    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        data_map = context.get('data_map')
        if not data_map: return None
        
        try:
            res = self.engine.deliberate(data_map)
            decision = res.get('decision', 'WAIT')
            score = res.get('score', 0)
            
            if decision != "WAIT":
                return SwarmSignal(
                    source=self.name,
                    signal_type=decision,
                    confidence=min(100.0, abs(score)),
                    timestamp=time.time(),
                    meta_data=res
                )
        except Exception as e:
            logger.debug(f"Overlord Logic Error: {e}")
            
        return None
