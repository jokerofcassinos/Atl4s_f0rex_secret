import logging
import numpy as np
import pandas as pd
from analysis.kinematics import Kinematics

logger = logging.getLogger("Atl4s-RecursiveReasoner")

class RecursiveReasoner:
    """
    The Internal Debater (Logical System).
    Performs 'Recursive Self-Reflection' on decisions.
    - Path Simulation (Kinetic Projection)
    - Adversarial Market Testing (Veto Logic)
    - Recursive Refinement (3 Iterations)
    """
    def __init__(self):
        self.kin_engine = Kinematics()

    def simulate_future_path(self, df, steps=5):
        """
        Projects likely future price path using Kinematics (Velocity/Acceleration).
        """
        if df is None or len(df) < 10:
            return df['close'].iloc[-1] * np.ones(steps)
            
        # Get current Physics
        _, _, _, angle, energy = self.kin_engine.analyze(df)
        
        # Simple projection: Next price = Current + Velocity + 0.5 * Accel
        # We'll use the ROC of the last few bars as a proxy for velocity if needed,
        # but Kinematics already gives us 'phy_score' which we can use.
        
        last_price = df['close'].iloc[-1]
        returns = df['close'].pct_change().dropna().values
        velocity = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        
        path = []
        curr_price = last_price
        for i in range(steps):
             # Decay the velocity to be conservative 
             curr_price *= (1 + velocity * (0.9 ** i))
             path.append(curr_price)
             
        return np.array(path)

    def debate(self, consensus_decision, score, data_map, iterations=3):
        """
        Recursively questions a decision.
        If the decision is 'BUY', it asks: 'Does the future path support this?'
        """
        df_m5 = data_map.get('M5')
        if df_m5 is None or consensus_decision == "WAIT":
             return consensus_decision, score, "No debate needed."

        current_decision = consensus_decision
        current_score = score
        history = []
        
        for i in range(iterations):
            # 1. Project Future
            future_path = self.simulate_future_path(df_m5, steps=5)
            avg_future = np.mean(future_path)
            last_price = df_m5['close'].iloc[-1]
            
            # 2. Criticize
            adversarial_veto = False
            reason = ""
            
            if current_decision == "BUY":
                if avg_future < last_price:
                    adversarial_veto = True
                    reason = "Future projection shows price decay."
            elif current_decision == "SELL":
                if avg_future > last_price:
                    adversarial_veto = True
                    reason = "Future projection shows price ascent."
                    
            # 3. Refine
            if adversarial_veto:
                # ENLIGHTENED PIVOT PROTOCOL
                # If the Debate proves the original decision wrong, we don't just stop.
                # We align with the stronger truth (The Future Path).
                
                # Check magnitude of disagreement
                path_delta = (avg_future - last_price) / last_price
                
                if abs(path_delta) > 0.0005: # Significant drift (Relaxed to 5 bps for Aggressive Mode)
                    history.append(f"Iter {i}: Conflict detected. {reason}")
                    history.append("Result: ENLIGHTENED PIVOT. Switching Decision.")
                    
                    # Flip Decision
                    current_decision = "SELL" if current_decision == "BUY" else "BUY"
                    current_score = max(50.0, current_score) # Reset score to baseline confidence
                    
                    # Break loop, we found a new truth
                    break
                else:
                    # Weak conflict, just reduce score (Caution)
                    current_score *= 0.95 # Relaxed penalty (was 0.8)
                    history.append(f"Iter {i}: Minor Conflict ({reason}). Reducing confidence.")
                    if current_score < 40:
                         current_decision = "WAIT"
                         history.append("Result: Veto Entry (Weak Conviction).")
                         break
            else:
                current_score *= 1.1 # Reinforce confidence
                history.append(f"Iter {i}: Path confirms direction.")
                
        final_reason = " | ".join(history) if history else "Confirmed."
        current_score = min(current_score, 100) # Cap at 100
        
        return current_decision, current_score, final_reason
