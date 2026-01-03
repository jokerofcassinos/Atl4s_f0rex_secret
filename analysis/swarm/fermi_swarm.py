
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("FermiSwarm")

class FermiSwarm(SubconsciousUnit):
    """
    Phase 83: The Fermi Swarm (Golden Rule of Decay).
    
    Models the 'Half-Life' of a Trade Opportunity using Nuclear Decay Physics.
    
    Physics:
    - N(t) = N0 * e^(-lambda * t)
    - The probability of a trend continuing (N) decays exponentially over time (t).
    - lambda: Decay Constant (Volatility dependent).
    
    Logic:
    - We calculate the 'Half-Life' of profitable moves in the current regime.
    - If a trade remains open past its Half-Life without hitting TP, its probability of success drops < 50%.
    - ACTION: 'Decay Exit'.
      - Instead of waiting for Full TP, we accept 'Partial Decay' profit.
      - "Time is eating your Probability."
    """
    def __init__(self):
        super().__init__("Fermi_Swarm")
        self.default_half_life = 6 # Candles (30 mins at M5)

    async def process(self, context) -> SwarmSignal:
        tick = context.get('tick')
        df_m5 = context.get('df_m5')
        
        # We need Trade Duration information.
        # This assumes the 'tick' contains info about open positions or we infer it.
        # Since we don't have direct access to 'Order Open Time' in the tick structure universally,
        # we rely on the bridge sending 'best_ticket_time' or similar, OR we use Market State Decay.
        
        # Let's model Market Structure Decay generally.
        if df_m5 is None or len(df_m5) < 20: return None
        closes = df_m5['close'].values
        
        # 1. Calculate Regime Decay Constant (Lambda)
        # High Volatility = Fast Decay (Short Half-Life).
        # Low Volatility = Slow Decay (Long Half-Life).
        
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        atr = np.mean(highs[-10:] - lows[-10:])
        
        # Normalize ATR by Price
        vol_ratio = atr / closes[-1]
        
        # Base Lambda. Higher Vol = Higher Lambda.
        # Heuristic: Lambda = VolRatio * 1000.
        decay_constant = vol_ratio * 1000
        
        # Calculate Half Life in Candles: t_1/2 = ln(2) / lambda
        # We clamp this to reasonable values (3 to 12 candles).
        half_life_candles = np.log(2) / (decay_constant + 1e-9)
        half_life_candles = max(3, min(half_life_candles, 12))
        
        # 2. Analyze Current Micro-Trend Age
        # How long has the current directional move been alive?
        # We count consecutive candles of the same color or structure.
        
        current_trend_age = 0
        direction = 0 # 1 UP, -1 DOWN
        
        for i in range(1, 20):
            idx = -i
            open_p = df_m5['open'].values[idx]
            close_p = closes[idx]
            
            candle_dir = 1 if close_p > open_p else -1
            
            if i == 1:
                direction = candle_dir
                current_trend_age += 1
            else:
                if candle_dir == direction:
                    current_trend_age += 1
                else:
                    break
                    
        # 3. Decay Calculation
        # Probability remaining = e^(-lambda * age)
        # Actually using the half-life formulation directly: P = (0.5)^(age / half_life)
        
        prob_remaining = 0.5 ** (current_trend_age / half_life_candles)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        best_profit = tick.get('best_profit', 0.0)
        positions = tick.get('positions', 0)
        
        # 4. Decision Logic
        
        # If we have positions, and the Trend Age > Half Life, we are in "Radioactive Decay".
        # The probability of continuation is low.
        
        if positions > 0:
            if current_trend_age > half_life_candles:
                 # Trend is Dying.
                 if best_profit > 1.0: # Any profit is better than decay loss.
                     signal = "EXIT_ALL"
                     confidence = 85.0 + (100 * (1 - prob_remaining)) # Confidence increases as Prob drops
                     reason = f"FERMI: Radioactive Decay. Trend Age ({current_trend_age}) > Half-Life ({half_life_candles:.1f}). Prob: {prob_remaining:.2f}"
            
            elif prob_remaining < 0.25: # 2 Half-Lives passed. Dead.
                 if best_profit > -2.0: # Cut losses too if stalled forever
                     signal = "EXIT_ALL"
                     confidence = 90.0
                     reason = f"FERMI: Trend Dead (2x Half-Life). Exit to recycle capital."
                     
        else:
            # Entry Filter: Don't enter an old trend.
            if current_trend_age > half_life_candles:
                 signal = "VETO"
                 confidence = 70.0
                 reason = f"FERMI: Trend too old ({current_trend_age} bars). Half-Life {half_life_candles:.1f}."

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'half_life': half_life_candles, 'trend_age': current_trend_age, 'prob': prob_remaining, 'reason': reason}
            )
            
        return None
