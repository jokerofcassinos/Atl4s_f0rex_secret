import logging
import time
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.fourteenth_eye import FourteenthEye

logger = logging.getLogger("RicciSwarm")

class RicciSwarm(SubconsciousUnit):
    """
    The Geometric Sentinel.
    Wraps The Fourteenth Eye to monitor the Curvature of the Market Manifold.
    """
    def __init__(self):
        super().__init__("Ricci_Swarm")
        self.engine = FourteenthEye()
        
    async def process(self, context) -> SwarmSignal:
        data_map = context.get('data_map')
        if not data_map: return None
        
        result = self.engine.calculate_curvature(data_map)
        if not result: return None
        
        R = result.get('ricci_scalar', 0.0)
        vol_ratio = result.get('expansion_ratio', 1.0)
        
        signal_type = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Interpret Curvature
        
        # CASE A: Positive Curvature (Spherical) -> Compression / Mean Reversion
        if R > 2.0:
            # Space is closing in. Geodesics converge.
            # Price should revert to mean or stall.
            # If we are in extended trend, this signals exhaustion.
            
            # Logic: If trending UP, and R > 0, Predict Reversal (SELL).
            df = data_map.get('M5')
            trend_dir = 0
            if df is not None:
                sma50 = df['close'].rolling(50).mean().iloc[-1]
                price = df['close'].iloc[-1]
                trend_dir = 1 if price > sma50 else -1
                
            if trend_dir == 1:
                signal_type = "SELL" # Reversal Short
                confidence = 75.0 + min(R * 5, 20)
                reason = f"GEOMETRY: Spherical Curvature (R={R:.2f}). Space Compressing -> Reversal."
            elif trend_dir == -1:
                signal_type = "BUY" # Reversal Long
                confidence = 75.0 + min(R * 5, 20)
                reason = f"GEOMETRY: Spherical Curvature (R={R:.2f}). Space Compressing -> Reversal."
                
        # CASE B: Negative Curvature (Hyperbolic) -> Expansion / Explosion
        elif R < -2.0:
            # Space is tearing. Diagonals diverge exponentially.
            # Trend Acceleration.
            
            # Logic: Trade WITH the breakout.
            # R is negative. abs(R) is the magnitude.
            magnitude = abs(R)
            
            df = data_map.get('M5')
            impulse = 0
            if df is not None:
                # Immediate Momentum
                impulse = df['close'].iloc[-1] - df['open'].iloc[-3]
                
            if impulse > 0:
                signal_type = "BUY"
                confidence = 80.0 + min(magnitude * 5, 19)
                reason = f"GEOMETRY: Hyperbolic Curvature (R={R:.2f}). Space Expanding -> Acceleration UP."
            else:
                signal_type = "SELL"
                confidence = 80.0 + min(magnitude * 5, 19)
                reason = f"GEOMETRY: Hyperbolic Curvature (R={R:.2f}). Space Expanding -> Acceleration DOWN."
                
        if signal_type != "WAIT":
            meta = result
            meta['desc'] = reason
            return SwarmSignal(
                source=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=time.time(),
                meta_data=meta
            )
            
        return None
