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
        self.volume_buckets = deque(maxlen=max_ticks)
        self.current_bucket_vol = 0
        self.current_bucket_buy = 0
        self.current_bucket_sell = 0
        self.vpin_values = deque(maxlen=50)
        self.bucket_size = 10 # Volume units per bucket (calibration needed for XAUUSD)
        
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
        
        # VPIN Logic: Accumulate into volume buckets
        self.current_bucket_vol += volume
        if direction == 1: self.current_bucket_buy += volume
        elif direction == -1: self.current_bucket_sell += volume
        
        # If bucket is "full", calculate imbalance and reset
        if self.current_bucket_vol >= self.bucket_size:
            imbalance = abs(self.current_bucket_buy - self.current_bucket_sell)
            self.volume_buckets.append(imbalance)
            self.current_bucket_vol = 0
            self.current_bucket_buy = 0
            self.current_bucket_sell = 0
            
            # Update VPIN (Probability of Informed Trading)
            # VPIN = sum(|V_buy - V_sell|) / (n * V_total)
            if len(self.volume_buckets) >= 10:
                vpin = sum(self.volume_buckets) / (len(self.volume_buckets) * self.bucket_size)
                self.vpin_values.append(vpin)

        self.last_price = current_price

    def calculate_micro_hurst(self, prices, window=50):
        """Micro-Hurst to detect tick-level exhaustion."""
        if len(prices) < 20: return 0.5
        lags = range(2, 10)
        try:
            log_lags = np.log(lags)
            log_variances = np.log([np.var(prices[lag:] - prices[:-lag]) for lag in lags])
            poly = np.polyfit(log_lags, log_variances, 1)
            return max(0.01, min(0.99, poly[0] / 2.0))
        except:
            return 0.5

    def calculate_ofi_proxy(self):
        """Order Flow Imbalance (OFI) Proxy using tick-level delta."""
        if len(self.ticks) < 10: return 0
        subset = list(self.ticks)[-10:]
        ofi = 0
        for i in range(1, len(subset)):
            prev = subset[i-1]
            curr = subset[i]
            
            # Simplified OFI: (V_buy if P_curr >= P_prev) - (V_sell if P_curr <= P_prev)
            if curr['price'] > prev['price']:
                ofi += curr['volume']
            elif curr['price'] < prev['price']:
                ofi -= curr['volume']
            else: # Price equal
                if curr['direction'] == 1: ofi += curr['volume']
                elif curr['direction'] == -1: ofi -= curr['volume']
        return ofi

    def analyze(self):
        """
        Returns micro-structure metrics:
        - Instant Velocity
        - Tick Frequency
        - Flow Delta
        - Micro Hurst
        - OFI Proxy
        """
        if len(self.ticks) < 10:
            return {'velocity': 0, 'frequency': 0, 'delta': 0, 'rejection': False, 'imbalance': 0, 'entropy': 1.0, 'micro_hurst': 0.5, 'ofi': 0}
            
        # 1. Tick Frequency (Activity Level)
        now = time.time()
        count = 0
        for i in range(len(self.ticks)-1, -1, -1):
            if self.ticks[i]['time'] > now - 5:
                count += 1
            else:
                break
        frequency = count / 5.0
        
        # 2. Instant Velocity
        prices = [t['price'] for t in self.ticks]
        start_price = prices[-10] if len(prices) >= 10 else prices[0]
        end_price = prices[-1]
        velocity = end_price - start_price
        
        # 3. Flow Delta (Last 50 ticks)
        subset_len = min(50, len(self.ticks))
        buy_vol = 0
        sell_vol = 0
        ticks_list = list(self.ticks)
        for i in range(len(ticks_list)-1, len(ticks_list)-1-subset_len, -1):
            t = ticks_list[i]
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
        if len(prices) >= 20:
            subset_prices = prices[-20:]
            min_p = min(subset_prices)
            max_p = max(subset_prices)
            curr_p = subset_prices[-1]
            range_p = max_p - min_p
            if range_p > 0.05:
                if (curr_p - min_p) > (range_p * 0.8): 
                    rejection = "BULLISH_REJECTION"
        
        # 7. VPIN Metric
        vpin = float(self.vpin_values[-1]) if self.vpin_values else 0.5
        toxicity = "HIGH" if vpin > 0.8 else "NORMAL"
        
        # --- NEW HFT METRICS ---
        micro_hurst = self.calculate_micro_hurst(np.array(prices[-50:]))
        ofi = self.calculate_ofi_proxy()
        
        return {
            'velocity': velocity,
            'frequency': frequency,
            'delta': delta,
            'rejection': rejection,
            'imbalance': imbalance,
            'entropy': tick_entropy,
            'vpin': vpin,
            'toxicity': toxicity,
            'micro_hurst': micro_hurst,
            'ofi': ofi
        }

    def reset_candle(self):
        self.current_candle_delta = 0
