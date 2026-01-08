
import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("GameSwarm")

class GameSwarm(SubconsciousUnit):
    """
    Phase 92: Nash Equilibrium 3.0 (The Tri-State Game).
    
    Models the market as a non-cooperative game between:
    1. Trend Followers (TF) - Bet on Continuation.
    2. Mean Reverters (MR) - Bet on Reversal.
    3. Market Makers (MM) - Bet on Liquidity (Stop Hunts).
    """
    def __init__(self):
        super().__init__("Game_Swarm")

    def normalize(self, val):
        return max(0.0, min(1.0, val))

    async def process(self, context) -> SwarmSignal:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        # 1. Player Incentives (Payoff Potential)
        
        # A. Trend Followers (TF)
        # Incentive: High Momentum, Moving Average Slope
        sma20 = df['close'].rolling(20).mean()
        slope = (sma20.iloc[-1] - sma20.iloc[-5])
        tf_incentive = self.normalize(abs(slope) * 1000) # Norm
        
        # B. Mean Reverters (MR)
        # Incentive: Overbought/Oversold (RSI)
        # 0.5 = Neutral, 1.0 = Extreme Deviation
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        dist_from_50 = abs(rsi - 50)
        mr_incentive = self.normalize(dist_from_50 / 40.0) 
        
        # C. Market Makers (MM)
        # Incentive: Proximity to Liquidity (Highs/Lows)
        last_high = df['high'].iloc[-20:].max()
        last_low = df['low'].iloc[-20:].min()
        curr_price = df['close'].iloc[-1]
        
        dist_h = abs(curr_price - last_high)
        dist_l = abs(curr_price - last_low)
        # Closer to level = Higher Incentive to Hunt
        mm_incentive = self.normalize(1.0 - (min(dist_h, dist_l) / (last_high - last_low + 0.0001)))
        
        # 2. Nash Equilibrium Solver (Simplified)
        # We assume the Market moves continuously to the state of Highest Total Pain (or Max Liquidity).
        # Which player has the highest "Force" (Incentive * Volume)?
        # For this prototype, we treat Incentives as Force.
        
        # --- FIX: TREND FILTER FOR REVERSION ---
        # If Trend is Strong, Reversion is dangerous (catching a falling knife).
        # We penalize Reversion Score by the Trend Score.
        
        # Determine Trend Direction vs Reversion Direction
        trend_dir = "BUY" if slope > 0 else "SELL"
        revert_dir = "SELL" if rsi > 50 else "BUY"
        
        if trend_dir != revert_dir:
             # Conflict!
             # If Trend Incentive is high (> 0.5), dampen Reversion
             if tf_incentive > 0.5:
                  mr_incentive *= (1.0 - tf_incentive) # Dampener
                  # Example: TF=0.8, MR=1.0 -> MR becomes 0.2. Trend wins.
        
        scores = {
            'TREND': tf_incentive,
            'REVERT': mr_incentive,
            'HUNT': mm_incentive
        }
        
        winner = max(scores, key=scores.get)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        if winner == 'TREND':
            # Follow the slope
            direction = "BUY" if slope > 0 else "SELL"
            signal = direction
            confidence = 85.0 * tf_incentive
            reason = f"NASH: Trend Dominance (Score: {tf_incentive:.2f})"
            
        elif winner == 'REVERT':
            # Fade the move
            direction = "SELL" if rsi > 50 else "BUY"
            signal = direction
            confidence = 80.0 * mr_incentive
            reason = f"NASH: Mean Reversion Dominance (Score: {mr_incentive:.2f})"
            
        elif winner == 'HUNT':
            # Predatory Logic: If near High, Hunt High (Buy then Revert?)
            # Usually MM pushes TO the level.
            if dist_h < dist_l: # Near High
                signal = "BUY" # Push to sweep
                reason = f"NASH: Market Maker Hunt for Highs (Score: {mm_incentive:.2f})"
            else: # Near Low
                signal = "SELL" # Push to sweep
                reason = f"NASH: Market Maker Hunt for Lows (Score: {mm_incentive:.2f})"
            confidence = 90.0 * mm_incentive

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'scores': scores, 'reason': reason}
            )
            
        return None
