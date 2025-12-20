import pandas as pd
import numpy as np
import logging
from collections import deque
import time
from scipy.stats import entropy

logger = logging.getLogger("Atl4s-MicroStructure")

class MicroStructure:
    def __init__(self, max_ticks=1000):
        self.ticks = deque(maxlen=max_ticks)
        self.last_price = None
        self.current_candle_delta = 0
        self.last_candle_time = None
        
    def on_tick(self, tick):
        """
        Ingest a live tick.
        tick: {'time': ms, 'bid': float, 'ask': float, 'last': float, 'volume': int, 'flags': int}
        """
        current_price = tick['last']
        current_time = tick['time'] / 1000.0 # Convert to seconds
        
        # Initialize
        if self.last_price is None:
            self.last_price = current_price
            return

        # Determine Aggressor (Simplified)
        direction = 0
        if current_price > self.last_price:
            direction = 1 # Buy
        elif current_price < self.last_price:
            direction = -1 # Sell
            
        volume = tick.get('volume', 1)
        
        self.current_candle_delta += (direction * volume)
        
        self.ticks.append({
            'price': current_price,
            'time': current_time,
            'direction': direction,
            'volume': volume
        })
        
        self.last_price = current_price

    def analyze(self):
        """
        Returns micro-structure metrics:
        - Instant Velocity
        - Tick Frequency
        - Flow Delta
        """
        if len(self.ticks) < 10:
            return {'velocity': 0, 'frequency': 0, 'delta': 0, 'rejection': False, 'imbalance': 0, 'entropy': 1.0}
            
        # 1. Tick Frequency (Activity Level)
        now = time.time()
        # Optimize: Start from end of deque
        count = 0
        for i in range(len(self.ticks)-1, -1, -1):
            if self.ticks[i]['time'] > now - 5:
                count += 1
            else:
                break
        frequency = count / 5.0
        
        # 2. Instant Velocity
        start_price = self.ticks[-10]['price']
        end_price = self.ticks[-1]['price']
        velocity = end_price - start_price
        
        # 3. Flow Delta (Last 50 ticks)
        subset_len = min(50, len(self.ticks))
        buy_vol = 0
        sell_vol = 0
        for i in range(len(self.ticks)-1, len(self.ticks)-1-subset_len, -1):
            t = self.ticks[i]
            if t['direction'] == 1: buy_vol += t['volume']
            elif t['direction'] == -1: sell_vol += t['volume']
            
        total_vol = buy_vol + sell_vol
        delta = buy_vol - sell_vol
        imbalance = 0
        if total_vol > 0:
            imbalance = (buy_vol - sell_vol) / total_vol
        
        # 5. Tick Entropy
        tick_entropy = 0
        if count > 5:
            # Use only the ticks from the last 5 seconds (already counted)
            recent_times = [self.ticks[i]['time'] for i in range(len(self.ticks)-count, len(self.ticks))]
            intervals = np.diff(recent_times)
            try:
                hist, _ = np.histogram(intervals, bins=5, density=True)
                hist = hist[hist > 0]
                tick_entropy = entropy(hist)
            except:
                tick_entropy = 0
            
        # 6. Wick Rejection Detection
        rejection = False
        if len(self.ticks) >= 20:
            subset_wick = [self.ticks[i] for i in range(len(self.ticks)-20, len(self.ticks))]
            prices = [t['price'] for t in subset_wick]
            times = [t['time'] for t in subset_wick]
            min_p = min(prices)
            max_p = max(prices)
            curr_p = prices[-1]
            time_span = times[-1] - times[0]
            if time_span < 10: 
                range_p = max_p - min_p
                if range_p > 0.05: # Changed threshold for Gold?
                    if (curr_p - min_p) > (range_p * 0.8): 
                        rejection = "BULLISH_REJECTION"
        return {
            'velocity': velocity,
            'frequency': frequency,
            'delta': delta,
            'rejection': rejection,
            'imbalance': imbalance,
            'entropy': tick_entropy
        }

    def reset_candle(self):
        self.current_candle_delta = 0
