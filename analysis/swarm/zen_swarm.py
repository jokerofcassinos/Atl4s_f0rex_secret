from typing import Dict, Any
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

class ZenSwarm(SubconsciousUnit):
    """
    The Inner Peace (System Health).
    
    Monitors the 'agi_health' and 'neuroplasticity_state'.
    If the system is confused (high entropy, low health), it votes WAIT/VETO.
    If the system is in Flow, it boosts confidence.
    """

    def __init__(self, name: str = "Zen_Swarm"):
        super().__init__(name)

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        tick = context.get('tick', {})
        health_metrics = tick.get('agi_health', {})
        
        if not health_metrics:
            return None
            
        health_score = health_metrics.get('health_score', 100)
        
        if health_score < 50:
             # System is sick/confused.
             return SwarmSignal(
                 signal_type="VETO",
                 confidence=100.0,
                 source=self.name,
                 meta_data={"reason": "System Health Critical"},
                 timestamp=time.time()
             )
             
        # If health is perfect, we don't necessarily vote BUY/SELL, 
        # but we could output a META signal to boost others.
        # Currently SwarmSignal doesn't support "BOOST", so we abstain unless Vetoing.
        
        return None
