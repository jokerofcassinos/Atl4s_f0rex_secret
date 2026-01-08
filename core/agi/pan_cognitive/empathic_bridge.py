
import logging
from typing import Dict, Any

logger = logging.getLogger("HighFidelityResonance")

class HighFidelityResonance:
    """
    Sistema D-2: High Fidelity Resonance (Empathic Bridge)
    Sincroniza a intenção do usuário com a execução da AGI com 99% de fidelidade.
    """
    def __init__(self):
        self.user_model = {
            "risk_tolerance": "MODERATE",
            "focus": "PROFIT",
            "interaction_style": "DIRECT"
        }
        self.intent_state = "NEUTRAL"
        self.sentiment_history = []
        
    def synchronize(self, user_feedback: str, market_risk_score: float = 0.5) -> Dict[str, Any]:
        """
        Alinha o estado interno da AGI com o feedback do usuário.
        Uses simplistic keyword-based Sentiment Analysis + Symbiotic Negotiation.
        """
        user_feedback = user_feedback.upper()
        
        # 1. Sentiment Analysis (Keyword Weighting)
        sentiment_score = 0.5
        keywords_aggressive = ["SNIPER", "ATTACK", "AGGRESSIVE", "FULL", "NOW"]
        keywords_defensive = ["SAFE", "WAIT", "CAREFUL", "DEFENSIVE", "HOLD"]
        
        for k in keywords_aggressive:
            if k in user_feedback: sentiment_score += 0.1
            
        for k in keywords_defensive:
            if k in user_feedback: sentiment_score -= 0.1
            
        # Clamp
        sentiment_score = max(0.0, min(1.0, sentiment_score))
        self.sentiment_history.append(sentiment_score)
        
        # 2. Update Intent State
        if sentiment_score > 0.7:
            self.intent_state = "AGGRESSIVE"
            self.user_model['risk_tolerance'] = "HIGH"
        elif sentiment_score < 0.3:
            self.intent_state = "DEFENSIVE"
            self.user_model['risk_tolerance'] = "LOW"
            
        # 3. Symbiotic Negotiation (The Core of Phase 10)
        # Verify if User Intent conflicts with Market Reality
        negotiated_strategy, resonance = self._negotiate_reality(self.intent_state, market_risk_score)
        
        return {
            "resonance_score": resonance,
            "alignment_status": "SYNCHRONIZED" if resonance > 0.8 else "NEGOTIATED",
            "intent_state": self.intent_state,
            "sentiment_score": sentiment_score,
            "adapted_strategy": negotiated_strategy,
            "negotiation_log": f"User wanted {self.intent_state}, Market Risk {market_risk_score:.2f} -> Agreed on {negotiated_strategy}"
        }
        
    def _negotiate_reality(self, user_intent: str, market_risk: float) -> tuple:
        """
        Negotiates the final strategy.
        If User is AGGRESSIVE but Market Risk is HIGH (>0.8), we downgrade to SMART_AGGRESSIVE.
        If User is DEFENSIVE but Market Risk is LOW (<0.2), we suggest OPPORTUNISTIC.
        """
        final_strategy = "STANDARD"
        resonance = 0.99
        
        if user_intent == "AGGRESSIVE":
            if market_risk > 0.8:
                final_strategy = "SMART_AGGRESSIVE" # Protected Aggression
                resonance = 0.85 # Slight dissonance due to safety override
                logger.warning("Symbiotic Override: Downgrading Aggression due to Extreme Risk.")
            else:
                final_strategy = "SNIPER"
                
        elif user_intent == "DEFENSIVE":
            if market_risk < 0.2:
                final_strategy = "OPPORTUNISTIC" # Safe Scalping
                resonance = 0.90
            else:
                final_strategy = "DEFENSIVE"
                
        else:
            final_strategy = "STANDARD"
            
        return final_strategy, resonance
