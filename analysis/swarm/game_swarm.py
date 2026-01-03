
import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("GameSwarm")

class GameSwarm(SubconsciousUnit):
    """
    The Strategist.
    Treats the market as a Multiplayer Game.
    Identify:
    1. Nash Equilibrium (Point of Control)
    2. Trapped Traders (Pain Thresholds)
    """
    def __init__(self):
        super().__init__("Game_Swarm")
        self.lookback = 100

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < self.lookback: return None

        # 1. Nash Equilibrium (POC)
        equilibrium_price, dominance = self._calculate_nash(df_m5)
        
        # 2. Analyze Current State relative to Equilibrium
        current_price = df_m5.iloc[-1]['close']
        dist_pct = (current_price - equilibrium_price) / equilibrium_price * 100
        
        # LOGIC:
        # If Price is significantly FAR from Equilibrium, it wants to revert (Gravity)
        # UNLESS Dominance supports the move (Breakout)
        
        signal_type = "WAIT"
        confidence = 0
        reason = ""
        
        # Reversion Logic (Elastic Band)
        if abs(dist_pct) > 0.3: # > 0.3% deviation (significant for M5)
            if current_price > equilibrium_price:
                # Overextended UP. Check Dominance.
                if dominance < 0: # Bearish Flow dominating
                     signal_type = "SELL"
                     confidence = 80 + (abs(dominance) * 20)
                     reason = f"Nash Reversion: Price Extended (+{dist_pct:.2f}%) vs Bearish Delta"
            else:
                # Overextended DOWN
                if dominance > 0: # Bullish Flow dominating
                     signal_type = "BUY"
                     confidence = 80 + (dominance * 20)
                     reason = f"Nash Reversion: Price Extended ({dist_pct:.2f}%) vs Bullish Delta"
                     
        # Trapped Trader Logic (Breakout/Defense)
        # If we are AT Equilibrium, and Dominance is High -> Breakout Initiating
        if abs(dist_pct) < 0.05:
            if dominance > 0.3:
                 signal_type = "BUY"
                 confidence = 70
                 reason = "Nash Breakout: Stability + Bullish Accumulation"
            elif dominance < -0.3:
                 signal_type = "SELL"
                 confidence = 70
                 reason = "Nash Breakout: Stability + Bearish Accumulation"
                 
        if signal_type != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=0,
                meta_data={'reason': reason, 'nash_price': equilibrium_price, 'dominance': dominance}
            )
            
        return None

    def _calculate_nash(self, df):
        data = df.iloc[-self.lookback:]
        
        # Volume Profile Binning
        price_min = data['low'].min()
        price_max = data['high'].max()
        
        # Create 30 bins
        bins = np.linspace(price_min, price_max, 30)
        vol_profile = np.zeros(len(bins)-1)
        
        buy_vol = 0.0
        sell_vol = 0.0
        
        for idx, row in data.iterrows():
            avg_p = (row['open'] + row['close']) / 2
            bin_idx = np.digitize(avg_p, bins) - 1
            if 0 <= bin_idx < len(vol_profile):
                vol_profile[bin_idx] += row['volume']
                
            # Delta Approximation
            if row['close'] > row['open']:
                buy_vol += row['volume']
            elif row['close'] < row['open']:
                sell_vol += row['volume']
                
        # Nash Eq = Max Volume Node
        max_idx = np.argmax(vol_profile)
        nash_price = (bins[max_idx] + bins[max_idx+1]) / 2
        
        # Dominance (-1 to 1)
        total_vol = buy_vol + sell_vol
        dominance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
        
        return nash_price, dominance
