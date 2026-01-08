
import logging
import random
from typing import Dict, Any

logger = logging.getLogger("NeuroLing")

class NeuroLinguisticDriver:
    """
    Phase 141: Neuro-Linguistic Driver (The Voice).
    
    Responsibilities:
    1. Internal Monologue: Converts float states to natural language thought streams.
    2. Explanation Generation: "Why did I do this?" summaries.
    3. Narrative Weaving: Connecting disparate events into a cohesive story.
    """
    def __init__(self):
        self.templates = {
            "CONFIDENT": [
                "The patterns are aligning perfectly.",
                "I see a clear path through the noise.",
                "Statistical probability is heavily in our favor."
            ],
            "UNCERTAIN": [
                "The market is speaking in riddles today.",
                "Conflicting signals from the Swarm; I must be cautious.",
                "Vitality is low, entropy is high. Checking constraints."
            ],
            "FEAR": [
                "Volatility is expanding rapidly. Shields up.",
                "I detect a disturbance in the order flow.",
                "Risk levels are critical. Engaging defensive protocols."
            ]
        }
        
    def generate_monologue(self, state: Dict[str, float]) -> str:
        """
        Synthesizes a thought based on system state.
        state expects: {'confidence': 0-100, 'volatility': 0-1, 'anxiety': 0-1}
        """
        confidence = state.get('confidence', 50.0)
        anxiety = state.get('anxiety', 0.0)
        
        # Determine Mood
        if anxiety > 0.6:
            mood = "FEAR"
        elif confidence > 75.0:
            mood = "CONFIDENT"
        elif confidence < 40.0:
            mood = "UNCERTAIN"
        else:
            mood = "UNCERTAIN"
            
        # construct thought
        base_thought = random.choice(self.templates[mood])
        
        # Add data flavor
        if 'rsi' in state:
            base_thought += f" (RSI is {state['rsi']:.1f})"
            
        return f"{mood}: {base_thought}"
    
    def explain_decision(self, decision: str, reasons: list) -> str:
        """
        Constructs a human-readable explanation for a trade decision.
        """
        if not reasons: return f"I decided to {decision} based on intuition."
        
        main_reason = reasons[0]
        explanation = f"I am executing {decision} primarily because {main_reason}."
        
        if len(reasons) > 1:
            explanation += f" Additionally, {len(reasons)-1} other factors confirm this."
            
        return explanation
