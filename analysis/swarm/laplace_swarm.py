
import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("LaplaceSwarm")

class LaplaceSwarm(SubconsciousUnit):
    """
    Phase 98: The Laplace Demon (Deterministic Trajectory Engine).
    
    "We may regard the present state of the universe as the effect of its past and the cause of its future."
    - Pierre-Simon Laplace
    
    This Swarm treats Price as a physical particle with Mass and Velocity.
    It runs a forward simulation to find the point of rest (Terminal Price).
    """
    def __init__(self):
        super().__init__("Laplace_Swarm")
        self.mass_window = 20 # Average mass over 20 candles
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        tick = context.get('tick')
        df_m1 = context.get('df_m1')
        
        if tick is None or df_m1 is None or len(df_m1) < 50:
            return None
            
        current_price = tick['bid']
        
        # 1. Calculate Physical Constants
        
        # MASS (m): Proportional to Relative Volume
        # Heavier mass = Harder to stop (High Momentum Persistence)
        recent_vol = df_m1['volume'].iloc[-5:].mean()
        avg_vol = df_m1['volume'].iloc[-50:].mean()
        if avg_vol == 0: avg_vol = 1
        
        mass = max(0.1, recent_vol / avg_vol) # Normalized Mass around 1.0
        
        # VELOCITY (v): Price Delta per Second (approximated by M1 close-close)
        # We take the derivative of the last few candles
        closes = df_m1['close'].values
        velocity = (closes[-1] - closes[-2]) # Points per minute approx
        
        # Smoothed Velocity
        v_smooth = (closes[-1] - closes[-4]) / 3.0
        
        # ACCELERATION (a): Change in velocity
        v_prev = (closes[-2] - closes[-5]) / 3.0
        acceleration = v_smooth - v_prev
        
        # FRICTION (mu): Resistance of the medium (Order Book)
        # Approximated by Spread and Volatility
        spread = tick['ask'] - tick['bid']
        volatility = df_m1['high'].iloc[-10:].max() - df_m1['low'].iloc[-10:].min()
        
        # Higher spread = Higher Friction
        # Higher Volatility = Lower Viscosity (Easier to move)? Or Turbulence?
        # Let's say Friction scales with Spread.
        friction_coeff = max(0.01, spread * 10.0) 
        
        # 2. The Simulation (Forward Integration) -- The "Demon" Logic
        terminal_price = current_price
        
        # Try C++ Acceleration
        cpp_active = False
        try:
            import ctypes
            from core.cpp_loader import load_dll
            
            lib = load_dll("physics_core.dll")
            
            class TrajectoryResult(ctypes.Structure):
                _fields_ = [
                    ("terminal_price", ctypes.c_double),
                    ("max_deviation", ctypes.c_double),
                    ("steps_taken", ctypes.c_int),
                    ("total_distance", ctypes.c_double)
                ]
            
            lib.simulate_trajectory.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int
            ]
            lib.simulate_trajectory.restype = TrajectoryResult
            
            # Call C++
            res = lib.simulate_trajectory(
                current_price, v_smooth, acceleration, 
                mass, friction_coeff, 1.0, 100
            )
            
            terminal_price = res.terminal_price
            cpp_active = True
            # logger.debug(f"Laplace [C++]: Terminal {terminal_price:.2f}")
                 
        except Exception as e:
            # logger.warning(f"Laplace C++ Error: {e}")
            pass
            
        if not cpp_active:
            # Fallback Python Simulation
            sim_p = current_price
            sim_v = v_smooth
            sim_a = acceleration # Initial impulse
            time_step = 1.0 # Virtual Seconds? Steps?
            
            max_steps = 100
            
            for t in range(max_steps):
                # Force Balance: F_net = F_inertial - F_friction
                
                # Crash Protection: Clamp velocity to prevent overflow
                sim_v = max(-1000.0, min(1000.0, sim_v))
                
                # Friction Force (Hybrid Model)
                # Low speed = Linear (Stokes), High speed = Quadratic (Turbulent)
                if abs(sim_v) < 10:
                    f_friction = -1.0 * friction_coeff * sim_v
                else:
                    f_friction = -1.0 * friction_coeff * (abs(sim_v) ** 1.5) * (1 if sim_v > 0 else -1)
                
                # Update Velocity (v = u + at) -> but force based
                # a = F/m
                # We assume initial acceleration decays, new forces are just friction
                
                # Decay the initial drive
                sim_a *= 0.9 
                
                # Effective Acceleration this step
                eff_a = sim_a + (f_friction / mass)
                
                if np.isnan(eff_a) or np.isinf(eff_a): eff_a = 0.0
                
                sim_v += eff_a * time_step
                sim_p += sim_v * time_step
                
                # Terminal condition
                if abs(sim_v) < 0.001 * current_price: 
                    break
                    
            terminal_price = sim_p
        
        # 3. The Prophecy
        # If terminal price is significantly away from current price, we enter
        
        delta = terminal_price - current_price
        
        # Validating the trajectory
        # How far is the "Inevitability Point"?
        # E.g. +5.00 points away
        
        signal = "WAIT"
        conf = 0.0
        meta = {}
        
        threshold = volatility * 0.5 # Must move at least half a volatility range
        
        if abs(delta) > threshold:
            if delta > 0:
                signal = "BUY"
                # Confidence scales with Mass (Momentum) and Distance
                conf = 80.0 + (min(mass, 3.0) * 5.0)
                meta = {'reason': f"Deterministic Trajectory: Target {terminal_price:.2f} (+{delta:.2f})"}
            else:
                signal = "SELL"
                conf = 80.0 + (min(mass, 3.0) * 5.0)
                meta = {'reason': f"Deterministic Trajectory: Target {terminal_price:.2f} ({delta:.2f})"}
        
        if signal == "WAIT": return None
        
        return SwarmSignal(
            source=self.name,
            signal_type=signal,
            confidence=min(conf, 99.0),
            timestamp=time.time(),
            meta_data=meta
        )
