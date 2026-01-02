import logging
import time
import numpy as np
import config

logger = logging.getLogger("Atl4s-EleventhEye")

class EleventhEye:
    """
    The Gap Exploiter (The Opportunist).
    
    Role:
    - Identifies "Cognitive Dissonance" between the Architect (Strategic Command) and the Consensus (The Crowd/Body).
    - Exploits the time lag where the Strategy is right but the indicators haven't caught up.
    - Operates in "Chaotic Dynamic Spaces".
    """
    def __init__(self):
        self.name = "The Gap Exploiter"
        self.active_gaps = 0
        self.last_trade_time = 0
        self.base_cooldown = 300 # 5 Minutes Base
        
    def calculate_temporal_dilation(self, market_state, divergence):
        """
        Calculates the 'Time Dilation' factor based on Relativistic Market Physics.
        
        Logic:
        - High Chaos (Lyapunov > 0.5) -> Time Dilates (We wait longer to avoid fakeouts).
        - High Order/Trend (Hurst > 0.7) -> Time Contracts (We strike faster).
        - Massive Gravity (Divergence > 40) -> Time Contracts (Urgency).
        """
        lyapunov = market_state.get('lyapunov', 0)
        hurst = market_state.get('hurst', 0.5)
        
        dilation = 1.0
        
        # 1. Chaos Penalty (Time Dilation)
        if lyapunov > 0.2: # Low threshold for chaos awareness
            # Nonlinear penalty: 0.6 Lyapunov -> +0.8 delay
            dilation += (lyapunov * 2.0) 
            
        # 2. Trend Integrity Boost (Time Contraction)
        if hurst > 0.65:
            dilation -= 0.3 # 30% faster in strong trends
            
        # 3. Urgency Boost (Gravity Well)
        if abs(divergence) > 40:
             dilation -= 0.5 # 50% faster if the gap is massive
        
        # 4. Volatility Scaling
        volatility = market_state.get('volatility', 50)
        if volatility > 80:
             dilation += 0.5 # High Vol -> Slow down slightly (whipsaw protection)
             
        dilation = max(0.1, dilation) # Minimum 10% of base time
        
        return dilation

    def scan_for_divergence(self, architect_data, consensus_score, market_state):
        """
        Scans for the "Gap".
        """
        # 1. Cooldown Check
        current_time = time.time()
        time_since_last = current_time - self.last_trade_time
        
        # We need to calculate dilation even for "checking" if we are in cooldown, 
        # but we don't know divergence yet. 
        # Optimization: Use a default or last known divergence for preliminary check?
        # Better: Calculate divergence first, THEN apply cooldown logic.
        
        # 1. Unpack Data
        arch_score = architect_data.get('score', 0)
        arch_directive = architect_data.get('directive', 'NEUTRAL')
        
        gap_action = "WAIT"
        gap_score = 0
        reason = ""
        
        # 2. Define the Dissonance
        divergence = arch_score - consensus_score
        entropy = market_state.get('entropy', 0.5)
        
        # Calculate Dynamic Cooldown required for THIS specific market state
        dilation_factor = self.calculate_temporal_dilation(market_state, divergence)
        required_wait = self.base_cooldown * dilation_factor
        
        if time_since_last < required_wait:
            # We are cooling down.
            # But... if this is a NEW much bigger gap, maybe override?
            # User wants "Perfect Timing". Let's stick to the cooldown to prevent spam.
            return {
                'action': 'WAIT',
                'score': 0,
                'reason': f"cooling down (Need {required_wait:.0f}s, Has {time_since_last:.0f}s)",
                'divergence': divergence
            }
        
        # 3. Gap Logic
        # We only act if Chaos is present (Entropy > 0.4) because that's where indicators lag.
        if entropy > 0.4:
            
            # CASE 1: Architect Bullish (Forward looking) vs Consensus Lagging
            if arch_score > 15 and consensus_score < 10:
                if divergence > 15:
                    gap_action = "BUY"
                    gap_score = 80 + (divergence * 0.5) 
                    reason = f"GAP DETECTED: Architect ({arch_score:.1f}) sees Alpha. Div: {divergence:.1f}"
                    
            # CASE 2: Architect Bearish vs Consensus Lagging
            elif arch_score < -15 and consensus_score > -10:
                if divergence < -15: 
                    gap_action = "SELL"
                    gap_score = -80 + (divergence * 0.5) 
                    reason = f"GAP DETECTED: Architect ({arch_score:.1f}) sees Doom. Div: {divergence:.1f}"
                    
        if gap_action != "WAIT":
             # [!] Symbol replaced for unicode safety
            logger.info(f"[!] ELEVENTH EYE TRIGGER: {reason}")
            self.last_trade_time = current_time
            
        return {
            'action': gap_action,
            'score': gap_score,
            'reason': reason,
            'divergence': divergence
        }
