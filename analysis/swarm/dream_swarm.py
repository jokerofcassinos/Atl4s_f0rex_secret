from typing import Dict, Any
from core.interfaces import SubconsciousUnit, SwarmSignal
from core.agi.omni_cortex import OmniCortex
import time
import random

class DreamSwarm(SubconsciousUnit):
    """
    The Dreamer (Active Imagination).
    
    When the market is quiet, this Swarm 'dreams' of potential futures using the Omni-Cortex.
    It runs counterfactual simulations (What if price spikes? What if it crashes?)
    to find hidden opportunities that standard technicals miss.
    """

    def __init__(self, name: str = "Dream_Swarm"):
        super().__init__(name)
        self.omni_cortex = OmniCortex()
        self.last_dream_time = 0
        self.dream_interval = 60 # Dream every minute if stable

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        # 1. Perception
        prices = []
        if 'history' in context:
             prices = context['history']
        
        if prices:
             self.omni_cortex.perceive({'prices': prices})
             
        # 2. Check Regime
        # We only dream if the mind is calm (STABLE regime)
        if self.omni_cortex.current_regime != "STABLE":
             # If chaotic, we don't daydream; we focus on survival (Red Team handles this)
             return None

        # 3. Active Imagination (The Dream Loop)
        if time.time() - self.last_dream_time < self.dream_interval:
             return None
             
        self.last_dream_time = time.time()
        
        # Scenario Generation: "What if?"
        # We simulate a sudden movement to see if the MCTS finds a profitable path
        # Bias: Randomly choose Upside or Downside
        bias_dir = 1 if random.random() > 0.5 else -1
        
        current_price = prices[-1] if prices else 0
        if current_price == 0: return None
        
        # "Dreaming of a move..."
        thought = self.omni_cortex.bridge.mcts.run_guided_mcts(
            current_price=current_price,
            entry_price=current_price,
            direction=bias_dir, # Assume we follow the dream
            volatility=context.get('volatility', 0.005),
            drift=0.0,
            iterations=5000,
            depth=50,
            bias_strength=0.5, # Strong bias to force the scenario
            bias_direction=bias_dir
        )
        
        expected_value = thought.get('expected_value', 0.0)
        
        # If the dream result is exceptionally good, we treat it as a Premonition
        if expected_value > 2.0: # High expected return in this dream scenario
             signal_type = "BUY" if bias_dir == 1 else "SELL"
             return SwarmSignal(
                 signal_type=signal_type,
                 confidence=75.0, # Speculative
                 source=self.name,
                 meta_data={
                     "type": "PREMONITION",
                     "scenario": f"Imagined {signal_type} Scenario",
                     "expected_value": expected_value
                 },
                 timestamp=time.time()
             )
             
        return None
