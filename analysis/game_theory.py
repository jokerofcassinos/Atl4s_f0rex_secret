import numpy as np
import logging

logger = logging.getLogger("Atl4s-GameTheory")

class GameTheoryCore:
    def __init__(self):
        self.lookback = 100 

    def calculate_nash_equilibrium(self, df):
        """
        Calculates the Nash Equilibrium price (Point of Control) and Dominance.
        Returns:
            equilibrium_price (float): The price where max volume occurred.
            dominance_score (float): +1 (Total Bull) to -1 (Total Bear).
            is_stable (bool): True if price is near equilibrium.
        """
        if df is None or len(df) < 50:
            return {'equilibrium_price': 0, 'dominance_score': 0, 'is_stable': True}
            
        # Use recent history
        data = df.iloc[-self.lookback:]
        
        # 1. Volume Profile (Simplified)
        # We bin prices and sum volume
        price_min = data['low'].min()
        price_max = data['high'].max()
        
        # Determine dominance based on Delta (Buying Vol vs Selling Vol)
        # We approximate: Close > Open = Buy Vol, Close < Open = Sell Vol
        buy_vol = 0
        sell_vol = 0
        
        # Volume Profile Bins
        bins = np.linspace(price_min, price_max, 20)
        vol_profile = np.zeros(len(bins)-1)
        
        for index, row in data.iterrows():
            # Add to profile
            # Find bin
            p = (row['open'] + row['close']) / 2
            bin_idx = np.digitize(p, bins) - 1
            if 0 <= bin_idx < len(vol_profile):
                vol_profile[bin_idx] += row['volume']
                
            # Delta
            if row['close'] > row['open']:
                buy_vol += row['volume']
            elif row['close'] < row['open']:
                sell_vol += row['volume']
                
        # Nash Equilibrium is the Point of Control (Max Volume Bin)
        max_vol_idx = np.argmax(vol_profile)
        equilibrium_price = (bins[max_vol_idx] + bins[max_vol_idx+1]) / 2
        
        # Dominance
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            dominance_score = (buy_vol - sell_vol) / total_vol # -1 to 1
        else:
            dominance_score = 0
            
        # Stability
        current_price = df.iloc[-1]['close']
        dist_to_eq = abs(current_price - equilibrium_price)
        
        # If we are within 0.1% of Nash Eq, we are stable
        threshold = current_price * 0.001
        is_stable = dist_to_eq < threshold
        
        # Game State
        # If Dominance is High (>0.3) but Price is BELOW Equilibrium -> Inefficiency (Undervalued)
        # If Dominance is Low (<-0.3) but Price is ABOVE Equilibrium -> Inefficiency (Overvalued)
        
        return {
            'equilibrium_price': equilibrium_price,
            'dominance_score': dominance_score,
            'is_stable': is_stable,
            'dist_to_nash': dist_to_eq
        }
