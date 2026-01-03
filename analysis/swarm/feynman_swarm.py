
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("FeynmanSwarm")

class FeynmanSwarm(SubconsciousUnit):
    """
    Phase 87: The Feynman Swarm (Path Integral Optimization).
    
    Calculates the 'Action' (S) of the current trade path using the Principle of Least Action.
    
    Physics:
    - A particle takes the path that minimizes the Action S = Integral(L dt).
    - L (Lagrangian) = Kinetic Energy (T) - Potential Energy (V).
    - In Quantum Mechanics, paths with high S have rapidly oscillating phases that cancel out (Destructive Interference).
    - Only paths near the minimum S contribute to the probability amplitude.
    
    Logic:
    - We treat the Trade as a Particle.
    - Kinetic Energy (T): "Cost of Motion" -> Volatility / Chop / Time Decay.
      - High Chop = Wasted Energy = High T.
    - Potential Energy (V): "Distance to Attractor" -> Distance to TP.
      - Ideally, V decreases linearly. If V stays high or oscillates, it increases Action.
    - Action Accumulation: S += (T + V) * dt.
    - SIGNAL:
      - If S exceeds a critical threshold (The "classical path" limit), the trade is energetically inefficient.
      - Probability of hitting TP drops.
      - ACTION: FEYNMAN EXIT (Cut the path).
    """
    def __init__(self):
        super().__init__("Feynman_Swarm")
        self.action_history = {} # Ticket -> Accumulated Action S

    async def process(self, context) -> SwarmSignal:
        tick = context.get('tick')
        df_m1 = context.get('df_m1')
        
        # We need active trade info to calculate path action
        positions = tick.get('positions', 0)
        best_ticket = tick.get('best_ticket', 0)
        current_profit = tick.get('profit', 0.0) # Global profit, rough proxy if 1 trade
        
        if positions == 0:
            self.action_history.clear()
            return None
            
        if df_m1 is None or len(df_m1) < 10: return None
        
        # 1. Calculate Lagrangian Components
        
        # Kinetic (T): Volatility / Instability of the moment
        # Ideally, we want smooth flow (Laminar). High Volatility without progress is high Cost.
        last_candle = df_m1.iloc[-1]
        high_low = last_candle['high'] - last_candle['low']
        open_close = abs(last_candle['close'] - last_candle['open'])
        
        # Chop Factor: How much movement was wasted?
        # If High-Low is big but Open-Close is small, T is high (Wasted Energy).
        kinetic_t = (high_low - open_close) * 1000 # Scaling factor
        
        # Potential (V): Distance to Goal
        # We assume goal is +$5.00 profit (Virtual TP) or similar.
        # V = (Target - Current)
        target_profit = 5.0
        dist_to_target = max(0, target_profit - current_profit)
        potential_v = dist_to_target * 5 # Scaling
        
        # Lagrangian L = T + V (Here we sum costs, so it's really Cost Function)
        # In physics L = T - V, but we are minimizing Cost. Let's call it Hamiltonian H = T + V.
        # We want to minimize H.
        
        step_action = (kinetic_t + potential_v) 
        
        # 2. Accumulate Action (S) for the Ticket
        if best_ticket not in self.action_history:
            self.action_history[best_ticket] = 0.0
            
        self.action_history[best_ticket] += step_action
        
        total_action = self.action_history[best_ticket]
        
        # 3. Threshold Logic
        # If Action is too high, the path is "Toxic".
        # Threshold depends on timeframe. Let's say max 1000 'units' of pain.
        
        action_threshold = 2000.0 
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Dynamic Difficulty:
        # If Profit is negative, Action accumulates faster (Potential V is high).
        # If Profit is positive but reversing, Action accumulates.
        
        if total_action > action_threshold:
             # PATH INTERFERENCE DETECTED.
             # The Probability Amplitude is collapsing.
             
             if current_profit > 0.5:
                 signal = "EXIT_SPECIFIC"
                 confidence = 95.0
                 reason = f"FEYNMAN: Least Action Limit Exceeded (S={total_action:.0f}). Path too difficult. Bank ${current_profit:.2f}."
             elif current_profit > -2.0:
                 # Small loss, but path is torture. Cut it.
                 signal = "EXIT_SPECIFIC"
                 confidence = 85.0
                 reason = f"FEYNMAN: Least Action Limit Exceeded (S={total_action:.0f}). Path too difficult. Cut ${current_profit:.2f}."
        
        # Also could use S to VETO new entries if market S is high generally (High Entropy).
        
        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'s_action': total_action, 'ticket': best_ticket, 'reason': reason}
            )
            
        return None
