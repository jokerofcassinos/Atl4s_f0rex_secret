
import logging
import numpy as np
import time
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("ZeroPointSwarm")

class ZeroPointSwarm(SubconsciousUnit):
    """
    Phase 108: The Zero-Point Field (Kalman State Estimator).
    
    Uses a Recursive Kalman Filter to separate 'Vacuum Energy' (Noise) from 'Physical Matter' (Trend).
    
    Model: Constant Velocity
    State x = [Price, Velocity]^T
    """
    
    def __init__(self):
        super().__init__("Zero_Point_Swarm")
        
        # 1. State Vector [p, v]
        self.x = np.zeros((2, 1)) 
        
        # 2. State Covariance P
        self.P = np.eye(2) * 1000.0 # High uncertainty initially
        
        # 3. State Transition Matrix F (Constant Velocity)
        # x_k = x_{k-1} + v_{k-1}*dt
        # v_k = v_{k-1}
        self.dt = 1.0 # 1 minute/step normalized
        self.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ])
        
        # 4. Measurement Matrix H
        # We only observe Price (p)
        self.H = np.array([[1.0, 0.0]])
        
        # 5. Process Noise Covariance Q
        # Confidence in our model. Lower = Trust model line (smoother). Higher = Follow price jumps.
        q_pos = 0.1
        q_vel = 0.1
        self.Q = np.array([
            [q_pos, 0.0],
            [0.0, q_vel]
        ])
        
        # 6. Measurement Noise Covariance R
        # Confidence in data. Higher R = Noisy data ( Trust model more).
        self.R = np.array([[10.0]]) # To be adaptive
        
        self.initialized = False
        self.innovations = [] # Store recent residuals for Z-Score
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        tick = context.get('tick')
        if not tick: return None
        
        z = tick['bid'] # Measurement
        
        # Initialization
        if not self.initialized:
            self.x[0, 0] = z
            self.x[1, 0] = 0.0 # Zero velocity start
            self.initialized = True
            return None
            
        # --- A. PREDICT STEP ---
        # x_{k|k-1} = F * x_{k-1|k-1}
        x_pred = np.dot(self.F, self.x)
        
        # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        P_pred = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # --- B. UPDATE STEP ---
        # Innovation y = z - H * x_pred
        y = z - np.dot(self.H, x_pred)
        
        # Innovation Covariance S = H * P_pred * H^T + R
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
        
        # Kalman Gain K = P_pred * H^T * S^-1
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))
        
        # State Update x = x_pred + K * y
        self.x = x_pred + np.dot(K, y)
        
        # Covariance Update P = (I - K * H) * P_pred
        I = np.eye(2)
        self.P = np.dot(I - np.dot(K, self.H), P_pred)
        
        # --- C. ANALYSIS ---
        
        # Get scalar residual
        residual = y[0, 0]
        self.innovations.append(residual)
        if len(self.innovations) > 100: self.innovations.pop(0)
        
        if len(self.innovations) < 20: return None # Tuning
        
        # Adaptive R calculation?
        # If residuals are consistently high, increase R (Market is noisy).
        std_res = np.std(self.innovations)
        self.R[0,0] = max(1.0, std_res * std_res) # Update R estimate roughly
        
        # Z-Score of the Innovation
        if std_res == 0: std_res = 1.0
        z_score = residual / std_res
        
        estimated_value = self.x[0,0]
        velocity = self.x[1,0]
        
        signal = "WAIT"
        conf = 0.0
        meta = {'kalman_val': estimated_value, 'innov_z': z_score}
        
        # LOGIC 1: ANOMALY (Casimir Effect)
        # If Price is WAY above estimate (Z > 3), it's a bubble. Revert.
        if z_score > 3.0:
            signal = "SELL"
            conf = min(99.0, 70 + (z_score * 5))
            meta['reason'] = f"Zero Point: Positive Anomaly (Z={z_score:.2f}). Expect Reversion."
            
        elif z_score < -3.0:
            signal = "BUY"
            conf = min(99.0, 70 + (abs(z_score) * 5))
            meta['reason'] = f"Zero Point: Vacuum Collapse (Z={z_score:.2f}). Expect Reversion."
            
        # LOGIC 2: MOMENTUM (Zero Point Drive)
        # If Price is close to Estimate (Z < 1) AND Velocity is Strong
        elif abs(z_score) < 1.0:
            if velocity > 0.05: # Strong Up Trend
                signal = "BUY"
                conf = 60.0 + (min(velocity, 0.5) * 50)
                meta['reason'] = f"Zero Point: Coherent Flow (Vel={velocity:.4f})"
            elif velocity < -0.05: # Strong Down Trend
                signal = "SELL"
                conf = 60.0 + (min(abs(velocity), 0.5) * 50)
                meta['reason'] = f"Zero Point: Coherent Flow (Vel={velocity:.4f})"
                
        if signal == "WAIT": return None
        
        return SwarmSignal(
            source=self.name,
            signal_type=signal,
            confidence=conf,
            timestamp=time.time(),
            meta_data=meta
        )
