
import logging
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.sixth_eye import SixthEye
import time

logger = logging.getLogger("CouncilSwarm")

class CouncilSwarm(SubconsciousUnit):
    """
    Sixth Eye Swarm Wrapper.
    The Council: Secular Trend & Macro Fundamental Logic.
    """
    def __init__(self):
        super().__init__("Council_Swarm")
        self.engine = SixthEye()

    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        # Sixth Eye expects 'data_map' with 'MN', 'W1'
        # Orchestrator provides 'data_map' in context?
        # If not, we skip or try to use what we have.
        
        # SwarmOrchestrator.process_tick passes `state` which might contain data_map
        # If not, we might fail to act.
        # But let's assume context has access to data.
        
        data_map = context.get('data_map')
        if not data_map: return None
        
        try:
            res = self.engine.deliberate(data_map)
            decision = res.get('anchor', 'WAIT')
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
            logger.debug(f"Council Logic Error: {e}")
            
        return None
