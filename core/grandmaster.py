from typing import Dict, Any, Optional
import logging
from core.agi.omni_cortex import OmniCortex
from core.agi.dialectic import DialecticEngine
from core.memory.holographic import HolographicMemory
from core.agi.augur import Augur

logger = logging.getLogger("GrandMaster")

class GrandMaster:
    """
    The GrandMaster: Apex Decision Engine.
    Coordinates the Omni-Cortex and Dialectic Engine to produce
    'Ultra-Complex Reasoning' decisions.
    """
    
    def __init__(self):
        logger.info("Initializing GrandMaster AI...")
        
        # Phase 5: The Holographic Nexus
        self.memory = HolographicMemory() 
        
        # Phase 7: The Augur (Advanced Perception)
        self.augur = Augur()
        
        self.cortex = OmniCortex(memory=self.memory)
        self.dialectic = DialecticEngine(self.cortex)
        self.current_mindset = "NEUTRAL"
        self.user_anxiety = 0.0 # 0.0 to 1.0
        self.risk_modifier = 1.0
        
    def update_user_anxiety(self, level: float):
        """
        Adjust internal state based on User's emotional resonance.
        High Anxiety -> High Risk Aversion.
        """
        self.user_anxiety = level
        if self.user_anxiety > 0.7:
             self.current_mindset = "DEFENSIVE"
             self.risk_modifier = 0.5 # Halve the risk
             logger.info(f"EMPATHIC RESONANCE: High Anxiety Detected ({level}). Switching to DEFENSIVE Mode.")
        else:
             self.current_mindset = "NEUTRAL"
             self.risk_modifier = 1.0

    def perceive_and_decide(self, market_data: Dict[str, Any]) -> str:
        """
        Main cognitive loop.
        Returns: "BUY", "SELL", or "WAIT".
        """
        # 1. Perceive (Physics/Regime Check)
        self.cortex.perceive(market_data)
        
        # 2. Check for Stability (Skip heavy compute if flat)
        if self.cortex.current_regime == "STABLE":
            return "WAIT"
            
        # 3. Trigger Dialectic Debate
        # We need to construct a context for the engines
        # Assuming market_data contains 'close' price directly or df
        current_price = 0.0
        if 'close' in market_data: 
             current_price = market_data['close']
        elif 'df' in market_data and not market_data['df'].empty:
             current_price = market_data['df']['close'].iloc[-1]
             
        context = {
            'current_price': float(current_price),
            'volatility': 0.005, # Adaptive later
            'drift': 0.0
        }
        
        # Run The Debate
        result = self.dialectic.resolve_market_debate(context)
        
        # 4. Final Judgment
        if result.decision != "WAIT":
             logger.info(f"GRANDMASTER VERDICT: {result.decision} | Confidence: {result.confidence:.1f}%")
             logger.info(f"Reasoning: {result.winner} => {result.reasoning}")
             
        return result.decision
        
    def get_status(self):
        return f"Regime: {self.cortex.current_regime} | Mindset: {self.current_mindset}"

    def get_recursive_depth(self) -> int:
        """
        Phase 11: AGI Depth Calculation for Hydra.
        Returns a simulated recursion depth (1-100) based on Regime Complexity.
        """
        base_depth = 10
        
        # 1. Fisher Complexity Audit
        complexity = self.cortex.fisher_metric # 0.0 to 10.0+
        base_depth += int(complexity * 10)
        
        # 2. Empathic Resonance (Anxiety = Deep Thought)
        if self.user_anxiety > 0.5:
             base_depth += 20
             
        # 3. Intuition Boost
        if hasattr(self.cortex, 'last_intuition') and abs(self.cortex.last_intuition) > 0.5:
             base_depth += 15
             
        return min(100, base_depth)
