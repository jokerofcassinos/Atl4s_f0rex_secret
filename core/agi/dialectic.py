from dataclasses import dataclass
from typing import Dict, Optional
import logging
from core.agi.omni_cortex import OmniCortex

logger = logging.getLogger("DialecticEngine")

@dataclass
class DialecticResult:
    decision: str # BUY, SELL, or WAIT
    confidence: float
    thesis_score: float # Bull Strength
    antithesis_score: float # Bear Strength
    winner: str # THESIS or ANTITHESIS or STALEMATE
    reasoning: str

class DialecticEngine:
    """
    The Dialectic Engine: AGI Reasoning v2.0.
    Simulates a debate between a Proponent (Thesis) and an Opponent (Antithesis).
    Reduces Confirmation Bias by actively seeking falsification.
    """
    
    def __init__(self, cortex: OmniCortex):
        self.cortex = cortex
        
    def resolve_market_debate(self, context: Dict) -> DialecticResult:
        """
        Conducts a debate on the current market state.
        Thesis: Market goes UP (Buy).
        Antithesis: Market goes DOWN (Sell).
        Synthesis: The stronger argument wins.
        """
        price = context.get('price', 0)
        
        # 1. The Thesis (Bull Argument)
        # Force the dream to look for Upside
        thesis_dream = self.cortex.run_deep_thought(
            context,
            force_bias=1 # Bias Direction: UP
        )
        
        # 2. The Antithesis (Bear Argument)
        # Force the dream to look for Downside
        antithesis_dream = self.cortex.run_deep_thought(
            context,
            force_bias=-1 # Bias Direction: DOWN
        )
        
        # 3. Evaluate Arguments (Synthesis)
        bull_strength = thesis_dream.get('expected_value', 0) * (thesis_dream.get('confidence', 0) / 100.0)
        bear_strength = abs(antithesis_dream.get('expected_value', 0)) * (antithesis_dream.get('confidence', 0) / 100.0)
        
        # Calculate Net Strength (Bull - Bear)
        net_score = bull_strength - bear_strength
        
        # Decision Thresholds
        # If the gap is small, it's a STALEMATE (Ambiguous market)
        debate_threshold = 0.5 
        
        result = DialecticResult(
            decision="WAIT",
            confidence=0.0,
            thesis_score=bull_strength,
            antithesis_score=bear_strength,
            winner="STALEMATE",
            reasoning="Market Ambiguous. Both Bull and Bear cases have merit."
        )
        
        if net_score > debate_threshold:
            result.decision = "BUY"
            result.winner = "THESIS (BULL)"
            result.confidence = min(99.0, (net_score * 10) + 50)
            result.reasoning = f"Bull Thesis Dominant (EV: {bull_strength:.2f} vs {bear_strength:.2f})"
            
        elif net_score < -debate_threshold:
            result.decision = "SELL"
            result.winner = "ANTITHESIS (BEAR)"
            result.confidence = min(99.0, (abs(net_score) * 10) + 50)
            result.reasoning = f"Bear Antithesis Dominant (EV: {bear_strength:.2f} vs {bull_strength:.2f})"
            
        logger.info(f"DIALECTIC RESOLUTION: {result.winner} | Net: {net_score:.2f}")
        return result
