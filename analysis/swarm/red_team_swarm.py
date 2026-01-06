from typing import Dict, Any
from core.interfaces import SubconsciousUnit, SwarmSignal
from core.agi.adversarial_network import AdversarialCritic
from core.agi.omni_cortex import OmniCortex
import time

class RedTeamSwarm(SubconsciousUnit):
    """
    The Adversary (Red Team).
    
    Acts as the 'Devil's Advocate' within the Swarm.
    Now enhanced with Omni-Cortex (AGI Hybrid Engine) to simulate crashes.
    """

    def __init__(self, name: str = "Red_Team_Swarm"):
        super().__init__(name)
        self.critic = AdversarialCritic()
        self.omni_cortex = OmniCortex() # Initialize the AGI Brain Region

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        # 1. Omni-Cortex Perception (Physics Check)
        # Update internal state with latest prices if available
        current_price = 0.0
        
        # Extract price history from context if available
        # Assuming context['history'] is a list or we use 'last_close'
        prices = []
        if 'history' in context:
             prices = context['history']
             if len(prices) > 0: current_price = prices[-1]

        if prices:
             self.omni_cortex.perceive({'prices': prices})

        # 2. Deep Thought (Guided MCTS)
        # If the Cortex detects a "Regime Shift" (High Fisher Info), it will simulate disaster.
        thought = self.omni_cortex.run_deep_thought({
             'current_price': current_price,
             'volatility': context.get('volatility', 0.005)
        })

        if thought and thought.get('recommendation') == "VETO_LONG":
             return SwarmSignal(
                signal_type="VETO",
                confidence=100.0,
                source=self.name,
                meta_data={
                    "reason": "OmniCortex Crash Simulation (Expectation: {:.2f}%)".format(thought['crash_expectation']*100),
                    "fisher_info": thought['fisher_info']
                },
                timestamp=time.time()
            )

        # 3. Standard Critic (Fallback)
        failure_prob = self.critic.critique("BUY", context) 
        
        if failure_prob > 0.7:
            return SwarmSignal(
                signal_type="VETO",
                confidence=100.0,
                source=self.name,
                meta_data={"reason": f"Red Team Detected Trap (Prob: {failure_prob:.2f})"},
                timestamp=time.time()
            )
            
        return None
