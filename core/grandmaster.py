from typing import Dict, Any, Optional
import logging
from core.agi.omni_cortex import OmniCortex
from core.agi.dialectic import DialecticEngine

logger = logging.getLogger("GrandMaster")

class GrandMaster:
    """
    The GrandMaster: Apex Decision Engine.
    Coordinates the Omni-Cortex and Dialectic Engine to produce
    'Ultra-Complex Reasoning' decisions.
    """
    
    def __init__(self):
        logger.info("Initializing GrandMaster AI...")
        self.cortex = OmniCortex()
        self.dialectic = DialecticEngine(self.cortex)
        self.current_mindset = "NEUTRAL"
        
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
