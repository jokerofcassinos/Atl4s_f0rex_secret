
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

@dataclass
class SwarmSignal:
    """
    Standardized Protocol for Agent Communication.
    The 'Thought' produced by a Subconscious Unit.
    """
    source: str          # Name of the agent (e.g., 'Sniper_Wick_Detector')
    signal_type: str     # 'BUY', 'SELL', 'WAIT', 'VETO', 'URGENT_CLOSE'
    confidence: float    # -100.0 to 100.0
    timestamp: float
    meta_data: Dict[str, Any] # 'Why' context: {'reason': 'Liquidity Grab', 'level': 2045.50}

class SubconsciousUnit(ABC):
    """
    Base Template for all Swarm Agents.
    Mimics a neural cluster in the brain.
    """
    def __init__(self, name: str):
        self.name = name
        self.active = True
        self.weight = 1.0 # Synaptic Weight (Dynamic)

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        """
        The Thinking Process.
        Args:
            context: The Global Hardware State (OHLC data, Order Flow, Macro State)
        Returns:
            A SwarmSignal or None (if no synapse fired)
        """
        pass

    def adjust_weight(self, performance_metric: float):
        """Neuroplasticity: Adjust weight based on success/failure."""
        self.weight *= (1.0 + (performance_metric * 0.01))
        self.weight = max(0.1, min(self.weight, 5.0)) # Clamp weight
