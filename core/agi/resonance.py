
import logging
import time
import numpy as np
import random
from typing import Dict, Any

logger = logging.getLogger("ResonanceEngine")

class ResonanceEngine:
    """
    Phase 135: Resonance Engine (Empathy Module).
    
    Responsibilities:
    1. Anxiety Detection: Estimates user stress based on PnL volatility and Drawdown duration.
    2. Empathic Modulation: Adjusts System Risk ("Calm Down" vs "Push It") to align with user psychology.
    3. Symbiotic Feedback: Learns from manual interventions (e.g. panic closes).
    """
    def __init__(self):
        self.user_anxiety_level = 0.0 # 0.0 (Zen) to 1.0 (Panic)
        self.max_drawdown_tolerance = 0.05 # 5% default
        self.last_intervention_time = 0
        
        # Psychological Profiling
        self.profile = {
            "risk_aversion": 0.5, # 0=Gambler, 1=Conservative
            "patience": 0.5 # 0=Impatient, 1=Zen Master
        }
        
    def assess_user_state(self, pnl_percent: float, open_duration_minutes: float, is_in_drawdown: bool):
        """
        Infers user state from market conditions.
        "If I were a human, how would I feel?"
        """
        stress = 0.0
        
        # Factor 1: Drawdown magnitude vs Tolerance
        if pnl_percent < 0:
            dd_severity = abs(pnl_percent) / self.max_drawdown_tolerance
            stress += dd_severity * 0.6 * self.profile['risk_aversion']
            
        # Factor 2: Duration of pain (Time in Drawdown)
        if is_in_drawdown and open_duration_minutes > 15:
            # Pain increases with time
            time_stress = min(1.0, (open_duration_minutes - 15) / 60.0)
            stress += time_stress * 0.4 * (1.0 - self.profile['patience'])
            
        # Factor 3: Recent Interventions (Residual Anxiety)
        if time.time() - self.last_intervention_time < 3600: # 1 Hour decay
            stress += 0.2
            
        self.user_anxiety_level = min(1.0, stress)
        
        if self.user_anxiety_level > 0.7:
            logger.warning(f"RESONANCE: Detected High User Anxiety ({self.user_anxiety_level:.2f}). Initiating Calming Protocols.")
            
    def get_risk_modifier(self) -> float:
        """
        Returns a multiplier for position sizing (0.0 to 1.0).
        High Anxiety -> Lower Risk.
        """
        if self.user_anxiety_level < 0.3:
            return 1.0 # Full throttle
        elif self.user_anxiety_level > 0.8:
            return 0.25 # Defense mode
        else:
            # Linear scaling
            return 1.0 - (self.user_anxiety_level * 0.7)
            
    def register_manual_intervention(self, action_type: str):
        """
        Called when user manually closes/modifies a trade.
        """
        self.last_intervention_time = time.time()
        logger.info(f"RESONANCE: Observed Human Intervention ({action_type}). Learning preference...")
        
        if action_type == "PANIC_CLOSE":
            # User closed a losing trade manually. They are Risk Averse.
            self.profile['risk_aversion'] = min(1.0, self.profile['risk_aversion'] + 0.1)
            logger.info("RESONANCE: Increased Risk Aversion Profile.")
            
        elif action_type == "EARLY_PROFIT":
            # User took profit early. Low Patience.
            self.profile['patience'] = max(0.0, self.profile['patience'] - 0.1)
            logger.info("RESONANCE: Decreased Patience Profile.")
