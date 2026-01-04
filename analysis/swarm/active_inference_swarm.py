
import logging
import numpy as np
import time
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("ActiveInferenceSwarm")

class ActiveInferenceSwarm(SubconsciousUnit):
    """
    Phase 109: The Friston Particle (Active Inference).
    
    Based on Karl Friston's Free Energy Principle.
    Uses a Hierarchical Gaussian Filter (HGF) to model Price and Volatility.
    
    Goal: Minimize 'Surprise' (Variational Free Energy).
    If Surprise is high and irreducible, we must ACT (Trade) to change our sensation 
    (or rather, capitalize on the regime shift).
    """
    
    def __init__(self):
        super().__init__("Active_Inference_Swarm")
        
        # Level 1: Value x (Price)
        self.mu_1 = 0.0 # Mean belief
        self.pi_1 = 1.0 # Precision (Inverse Variance)
        
        # Level 2: Volatility x_2 (Log-Volatility of x_1)
        self.mu_2 = 0.0 # Mean log-vol
        self.pi_2 = 1.0 # Precision of vol
        
        # Level 3: Phasic Volatility (Meta-Vol) - Fixed for now
        self.v_3 = 0.01 # Evolution variance of level 2
        
        self.initialized = False
        self.learning_rate = 0.1 # Kappa
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        tick = context.get('tick')
        if not tick: return None
        
        u = tick['bid'] # Sensory Input (Observation)
        
        # Initialization
        if not self.initialized:
            self.mu_1 = u
            self.initialized = True
            return None
            
        # --- HGF UPDATE STEPS (Simplification of Mathys et al.) ---
        
        # 1. Prediction Error (PE) at Level 1
        # delta_1 = u - mu_1
        delta_1 = u - self.mu_1
        
        # 2. Precision weighting for Level 1
        # Estimated Volatility from Level 2: sigma_1 = exp(mu_2)
        # pi_1_hat = 1 / sigma_1
        sigma_1 = np.exp(self.mu_2)
        pi_1_hat = 1.0 / (sigma_1 + 1e-9)
        
        # 3. Precision-Weighted PE (epsi_1)
        # Sensation of "Surprise" relative to uncertainty
        epsi_1 = delta_1 * pi_1_hat
        
        # 4. Update Level 1 Belief (Value)
        # mu_1_new = mu_1 + alpha * epsi_1
        # alpha is like Kalman Gain
        self.mu_1 += self.learning_rate * epsi_1
        
        # 5. Prediction Error at Level 2 (Volatility PE)
        # Did we underestimate the volatility?
        # Observed Volatility energy = delta_1^2
        # Expected Volatility energy = sigma_1
        # PE ~ (delta_1^2 / sigma_1) - 1
        # If delta is huge, delta^2 >> sigma, ratio > 1, PE > 0 (Volatility underestimated)
        delta_2 = (delta_1**2 * pi_1_hat) - 1.0
        
        # 6. Update Level 2 Belief (Volatility)
        # Precision of Level 2 (pi_2) depends on Level 3 (fixed)
        pi_2_hat = 1.0 / self.v_3
        epsi_2 = delta_2 * pi_2_hat
        
        self.mu_2 += (self.learning_rate * 0.1) * epsi_2 # Slower update for Vol
        
        # Stability Clamp (Phase 114 Fix)
        # Avoid explosion of volatility belief
        self.mu_2 = np.clip(self.mu_2, -5.0, 5.0) 
        
        # --- INFERENCE ---
        
        # We track "Surprise" (Free Energy). 
        # F ~ PE^2 * Precision
        # F = 0.5 * (delta_1^2 / sigma_1)
        free_energy = 0.5 * (delta_1**2 * pi_1_hat)
        
        # Analyze the Structure of Surprise
        # If Free Energy is LOW (< 1.0), the model predicts well. Noise.
        # If Free Energy is HIGH (> 4.0), huge surprise.
        
        signal = "WAIT"
        conf = 0.0
        meta = {'F': free_energy, 'vol': sigma_1}
        
        # Active Inference Logic:
        # A "Regime Shift" occurs when Volatility (Level 2) spikes AND Price (Level 1) moves.
        # The model's beliefs are "Broken". Price is finding a new equilibrium.
        
        # Thresholds (Heuristic)
        SURPRISE_THRESHOLD = 3.0 # > 3 Sigma event effectively
        
        if free_energy > SURPRISE_THRESHOLD:
            # Huge Surprise.
            # Direction?
            # If PE (delta_1) is positive, Price > Prediction. Breakout UP.
            
            if delta_1 > 0:
                signal = "BUY"
                conf = min(99.0, 70 + free_energy * 2)
                meta['reason'] = f"Friston: High Surprise (F={free_energy:.2f}). Model Broken Upwards."
            else:
                signal = "SELL"
                conf = min(99.0, 70 + free_energy * 2)
                meta['reason'] = f"Friston: High Surprise (F={free_energy:.2f}). Model Broken Downwards."
                
        # What if Free Energy is Low?
        # Mean Reversion? Model assumes stationary.
        # If F is very low, we ignore.
        
        if signal == "WAIT": return None
        
        return SwarmSignal(
            source=self.name,
            signal_type=signal,
            confidence=conf,
            timestamp=time.time(),
            meta_data=meta
        )
