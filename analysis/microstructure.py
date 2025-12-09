import pandas as pd
import numpy as np
import logging
from collections import deque
import time

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
        # If price went up, assume aggressive buy.
        # If price went down, assume aggressive sell.
        # If price same, use flag or ignore.
        
        direction = 0
        if current_price > self.last_price:
            direction = 1 # Buy
        elif current_price < self.last_price:
            direction = -1 # Sell
            
        volume = tick['volume']
        
        # Update Candle Delta (Reset on new candle handled externally or by time check)
        # For now, we just track a rolling delta or reset manually.
        # Let's assume this is a rolling flow analyzer.
        self.current_candle_delta += (direction * volume)
        
        # Store tick for velocity calc
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
        - Instant Velocity (price change / sec over last N ticks)
        - Tick Frequency (ticks / sec)
        - Flow Delta (Net Volume Direction)
        """
        if len(self.ticks) < 10:
            return {'velocity': 0, 'frequency': 0, 'delta': 0, 'rejection': False, 'imbalance': 0, 'entropy': 1.0}
            
        # 1. Tick Frequency (Activity Level)
        now = time.time()
        recent_ticks = [t for t in self.ticks if t['time'] > now - 5]
        frequency = len(recent_ticks) / 5.0 if len(recent_ticks) > 0 else 0
        
        # 2. Instant Velocity
        start_price = self.ticks[-10]['price']
        end_price = self.ticks[-1]['price']
        velocity = end_price - start_price
        
        # 3. Flow Delta (Last 50 ticks)
        # & 4. Order Flow Imbalance
        subset = list(self.ticks)[-50:]
        buy_vol = sum(t['volume'] for t in subset if t['direction'] == 1)
        sell_vol = sum(t['volume'] for t in subset if t['direction'] == -1)
        total_vol = buy_vol + sell_vol
        
        delta = buy_vol - sell_vol
        imbalance = 0
        if total_vol > 0:
            imbalance = (buy_vol - sell_vol) / total_vol # -1 to 1
        
        # 5. Tick Entropy (Machine vs Human)
        # Calculate Shannon Entropy of time intervals between ticks
        # Regular intervals (HFT loops) -> Low Entropy
        # Irregular (Human) -> High Entropy
        if len(recent_ticks) > 5:
            times = [t['time'] for t in recent_ticks]
            intervals = np.diff(times)
            # Histogram
            try:
                hist, _ = np.histogram(intervals, bins=5, density=True)
                hist = hist[hist > 0]
                from scipy.stats import entropy
                tick_entropy = entropy(hist)
            except:
                tick_entropy = 0
        else:
            tick_entropy = 0
            
        # 6. Wick Rejection Detection
        rejection = False
        if len(self.ticks) >= 20:
            prices = [t['price'] for t in list(self.ticks)[-20:]]
            times = [t['time'] for t in list(self.ticks)[-20:]]
            min_p = min(prices)
            max_p = max(prices)
            curr_p = prices[-1]
            time_span = times[-1] - times[0]
            if time_span < 10: 
                range_p = max_p - min_p
                if range_p > 0.5: 
                    if (curr_p - min_p) > (range_p * 0.8): 
                        rejection = "BULLISH_REJECTION"
                    elif (max_p - curr_p) > (range_p * 0.8):
                        rejection = "BEARISH_REJECTION"

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
