
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque

class NanoBlockAnalyzer:
    """
    Implements the 'Nano Algo Protocol'.
    Detects 'Nano Blocks' (High Density Liquidity Zones) and 'Icebergs' (Absorption).
    """
    def __init__(self, price_bins=100, memory_ticks=2000):
        # Dynamic Price Grid
        self.price_bins = {} # {price_level: volume}
        self.red_blocks = [] # List of resistance levels
        self.green_blocks = [] # List of support levels
        
        # Iceberg History
        self.icebergs = deque(maxlen=20)
        
        # Volatility Tracking for Algo Detection
        self.std_dev_window = deque(maxlen=50)
        
        # Configuration
        self.tick_size = 0.00001 # 0.1 pip
        self.block_threshold = 2.0 # Standard deviations above mean volume to qualify as a block
        
    def on_tick(self, tick: Dict):
        """
        Ingest live tick data to update Nano Structure.
        tick: {'bid': float, 'ask': float, 'volume': float, 'time': float}
        """
        price = (tick['bid'] + tick['ask']) / 2
        volume = tick.get('volume', 1.0)
        
        # 1. Update Volumetric Map
        # Round to nearest tick bin
        bin_price = round(price / self.tick_size) * self.tick_size
        self.price_bins[bin_price] = self.price_bins.get(bin_price, 0) + volume
        
        # Decay old volume (Liquidity isn't static)
        for p in self.price_bins:
            self.price_bins[p] *= 0.995 # Fast decay for "Nano" structure (short-term memory)
            
        # 2. Iceberg Detection (Absorption)
        # High Volume + Low Displacement
        # In this tick, if volume is abnormally high but price didn't move much from last tick
        # (Note: Requires last_price state, which we assume is handled by caller or inferred)
        # Simplified: If we see repeated hits to same bin with high volume -> Iceberg
        
        # 3. Block Analysis (Periodic, not every tick for perf)
        # We'll do a quick check every 10 volume units
        pass 
        
    def analyze(self, current_price: float) -> Dict:
        """
        Returns adjacent blocks and algo signals.
        """
        # Calculate Volume Statistics
        vols = list(self.price_bins.values())
        if not vols: return {}
        
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        threshold = mean_vol + (std_vol * self.block_threshold)
        
        # Identify Blocks
        red_zones = []
        green_zones = []
        
        for p, v in self.price_bins.items():
            if v > threshold:
                # Classify into Red (Res) or Green (Sup)
                # If block is above current price -> Resistance (Red)
                # If block is below current price -> Support (Green)
                if p > current_price:
                    red_zones.append({'price': p, 'strength': v})
                elif p < current_price:
                    green_zones.append({'price': p, 'strength': v})
                    
        # Sort by proximity
        red_zones.sort(key=lambda x: x['price']) # Ascending (Nearest resistance is first)
        green_zones.sort(key=lambda x: x['price'], reverse=True) # Descending (Nearest support is first)
        
        # Detect Algo Activity
        # "Sell Algos" usually manifest as precise, recurring resistance taps
        algo_sell = False
        if red_zones:
            nearest = red_zones[0]
            if abs(nearest['price'] - current_price) < (self.tick_size * 50): # Within 5 pips
                algo_sell = True
                
        # "Buy Algos"
        algo_buy = False
        if green_zones:
            nearest = green_zones[0]
            if abs(current_price - nearest['price']) < (self.tick_size * 50):
                algo_buy = True

        return {
            "red_blocks": red_zones[:3],   # Top 3 nearest resistance
            "green_blocks": green_zones[:3], # Top 3 nearest support
            "algo_sell_detected": algo_sell,
            "algo_buy_detected": algo_buy,
            "mean_volume": mean_vol,
            "block_threshold": threshold
        }

    def check_scale_in(self, trade_direction: str, current_price: float, entry_price: float) -> bool:
        """
        Logic for 3+3+3 Scale-In.
        Scale in when price crosses institutional buy algos + sell volume blocks.
        """
        analysis = self.analyze(current_price)
        
        if trade_direction == "BUY":
            # Scale in if we broke ABOVE a Red Block (Resistance turned Support)
            # Or if we just bounced off a Green Block (Algo Buy)
            
            # Logic: If Current Price > Entry Price + X (Profit) AND Algo Buy Detected
            if current_price > entry_price and analysis['algo_buy_detected']:
                return True
                
        elif trade_direction == "SELL":
             # Scale in if we broke BELOW a Green Block
             # Or rejected from Red Block (Algo Sell)
             if current_price < entry_price and analysis['algo_sell_detected']:
                 return True
                 
        return False
