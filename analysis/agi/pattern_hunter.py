from analysis.agi.akashic_core import AkashicCore, RealitySnapshot
import pandas as pd
import numpy as np
from datetime import datetime

class PatternHunter:
    """
    The Entity that hunts for patterns in the Akashic Records.
    """
    def __init__(self):
        self.memory = AkashicCore()
        print("🦅 [PATTERN_HUNTER] Awakened. Connecting to Akashic Records...")

    def analyze_live_market(self, candle_open, candle_high, candle_low, candle_close, volume, rsi, atr, timestamp: datetime):
        """
        Takes live market data, constructs a DNA snapshot, and queries history.
        """
        
        # 1. Morphological Analysis
        body_size = abs(candle_close - candle_open)
        upper_wick = candle_high - max(candle_open, candle_close)
        lower_wick = min(candle_open, candle_close) - candle_low
        total_range = candle_high - candle_low
        
        # Estimate SMA 200 Dist (Mocked here, normally passed in)
        # In full integration, we'd pass the full DF or indicators object
        sma_200_dist = 0.0 # Placeholder
        
        # 2. Chronometric Analysis
        is_8min = (timestamp.minute % 8 == 0)
        
        # 3. Create the Snapshot (The 'Question' to the Oracle)
        current_snapshot = RealitySnapshot(
            timestamp=timestamp.timestamp(),
            price_open=candle_open,
            price_high=candle_high,
            price_low=candle_low,
            price_close=candle_close,
            volume=volume,
            body_size=body_size,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            total_range=total_range,
            rsi_14=rsi,
            atr_14=atr,
            sma_200_dist=sma_200_dist,
            hour=timestamp.hour,
            minute=timestamp.minute,
            day_of_week=timestamp.weekday(),
            is_8min_cycle=is_8min
        )
        
        # 4. Consult the Oracle
        prediction = self.memory.consult_the_oracle(current_snapshot)
        
        return prediction

    def learn_from_session(self, df_history: pd.DataFrame):
        """
        Optional: Live learning mode.
        """
        pass
