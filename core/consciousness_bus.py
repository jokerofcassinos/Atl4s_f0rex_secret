
import asyncio
import logging
from typing import List, Callable, Dict, Any
from .interfaces import SwarmSignal

logger = logging.getLogger("ConsciousnessBus")

class ConsciousnessBus:
    """
    The Nervous System.
    Facilitates zero-latency event propagation between the Eye (Data) and the Mind (Swarm).
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {
            'TICK_EVENT': [],
            'CANDLE_EVENT': [],
            'RISK_EVENT': []
        }
        self.memory_stream: List[SwarmSignal] = [] # Short-term memory of signals

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        logger.debug(f"[Bus] New synapse connected to {event_type}")

    async def publish(self, event_type: str, payload: Any):
        """Fire the synapse."""
        if event_type in self.subscribers:
            # Broadcast to all agents in parallel (Fire-and-Forget style for speed)
            tasks = [func(payload) for func in self.subscribers[event_type]]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    def register_thought(self, signal: SwarmSignal):
        """Store a thought in short-term memory."""
        self.memory_stream.append(signal)
        if len(self.memory_stream) > 1000:
            self.memory_stream.pop(0)

    def get_recent_thoughts(self) -> List[SwarmSignal]:
        """Retrieves and clears recent thoughts."""
        thoughts = list(self.memory_stream)
        self.memory_stream = [] # Flush short-term memory after reading
        return thoughts

    def peek_thoughts(self) -> List[SwarmSignal]:
        """Retrieves recent thoughts WITHOUT clearing (ReadOnly)."""
        return list(self.memory_stream)

print("ConsciousnessBus Online.")
