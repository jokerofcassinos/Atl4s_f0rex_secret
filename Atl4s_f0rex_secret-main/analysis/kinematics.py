import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Atl4s-Kinematics")

class Kinematics:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Calculates Phase Space Kinematics.
        Velocity vs Acceleration.
        Phase Angle determines position in the cycle (0-360).
        """
        if df is None or len(df) < 10:
            return 0, 0, 0, 0, 0
            
        df = df.copy()
        
        # Velocity (1st Derivative)
        df['velocity'] = df['close'].diff()
        
        # Acceleration (2nd Derivative)
        df['acceleration'] = df['velocity'].diff()
        
        # We need smoothed values for Phase Space to look good
        # Simple SMA 3 smoothing
        df['vel_smooth'] = df['velocity'].rolling(3).mean()
        df['acc_smooth'] = df['acceleration'].rolling(3).mean()
        
        current_vel = df['vel_smooth'].iloc[-1]
        current_acc = df['acc_smooth'].iloc[-1]
        
        if np.isnan(current_vel): current_vel = 0
        if np.isnan(current_acc): current_acc = 0
        
        # Phase Angle
        # theta = arctan2(Velocity, Acceleration)
        # We normalize them first to be on similar scales?
        # Typically Accel is smaller than Vel.
        # Let's use raw for relative angle, or standardize z-scores.
        
        # Standardizing gives a cleaner unit circle
        vel_std = df['vel_smooth'].std()
        acc_std = df['acc_smooth'].std()
        
        if vel_std == 0: vel_std = 1
        if acc_std == 0: acc_std = 1
        
        norm_vel = current_vel / vel_std
        norm_acc = current_acc / acc_std
        
        # Phase Angle in Degrees
        # 0 degrees = East (High Acc, Zero Vel) -> Boom start?
        # 90 degrees = North (High Vel, Zero Acc) -> Peak Velocity
        # 180 degrees = West (Neg Acc, Zero Vel) -> Deceleration / Top
        # 270 degrees = South (Neg Vel, Zero Acc) -> Peak Drop
        
        angle_rad = np.arctan2(norm_vel, norm_acc)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0: angle_deg += 360
        
        # Orbit Energy (Distance from origin)
        # High Energy = Strong Trend/Volatility
        orbit_energy = np.sqrt(norm_vel**2 + norm_acc**2)
        
        score = 0
        
        # Momentum Burst based on Orbit Energy
        if orbit_energy > 2.0:
            # Significant move
            if 0 <= angle_deg < 90:
                logger.info(f"Phase Space: ACCELERATING UP (Angle {angle_deg:.0f}). Boom.")
                score = 80
            elif 90 <= angle_deg < 180:
                logger.info(f"Phase Space: DECELERATING UP (Angle {angle_deg:.0f}). Topping?")
                score = 40
            elif 180 <= angle_deg < 270:
                logger.info(f"Phase Space: ACCELERATING DOWN (Angle {angle_deg:.0f}). Crash.")
                score = -80
            elif 270 <= angle_deg < 360:
                logger.info(f"Phase Space: DECELERATING DOWN (Angle {angle_deg:.0f}). Bottoming?")
                score = -40
                
        return current_vel, current_acc, score, angle_deg, orbit_energy
