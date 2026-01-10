
import pandas as pd
import numpy as np

class LiquidityEngineer:
    """
    Tracks liquidity pools (stop-loss clusters) and detects 
    when institutions 'sweep' them before reversing.
    
    Concepts:
    - Liquidity Pool: Cluster of stops above Swing Highs (Buy Stops) or below Swing Lows (Sell Stops).
    - Sweep: Rapid price move into a pool to trigger stops, followed by reversal.
    - SFP (Swing Failure Pattern): Break of level + Close back inside range.
    """

    def __init__(self):
        self.pools = []

    def detect_liquidity_sweep(self, df: pd.DataFrame, direction: str) -> dict:
        """
        Sweep-to-Reversal Pattern Detection.
        
        Logic:
        1. Identify recent structural high/low (Swing Point).
        2. Check if current/recent candle wicked through it (taking liquidity).
        3. Check if candle CLOSED back inside the range (rejection/trap).
        """
        if df is None or len(df) < 20:
            return {'detected': False}
            
        last_candle = df.iloc[-1]
        # Look at last 15 candles for the structure (excluding current)
        recent = df.iloc[-21:-1] 
        
        result = {
            'detected': False,
            'type': 'NONE',
            'level': 0.0,
            'strength': 0.0,
            'age_candles': 0
        }
        
        if direction == "BUY":
            # BULLISH SWEEP:
            # Market was going down. Stops are below recent LOWS.
            # We want to see price DIP below a recent low and REVERSE up.
            recent_low = recent['low'].min()
            low_idx = recent['low'].idxmin()
            
            # Condition 1: Current Low went LOWER than recent low (Sweep)
            if last_candle['low'] < recent_low:
                # Condition 2: Current Close is HIGHER than recent low (Rejection/SFP)
                if last_candle['close'] > recent_low:
                    # Condition 3: Wick validation
                    # The move below the level should be quick (wick), not a massive breakdown yet.
                    # Or it could be a 'Turtle Soup' (ICT term).
                    
                    result['detected'] = True
                    result['type'] = 'BULLISH_SWEEP'
                    result['level'] = recent_low
                    result['strength'] = abs(last_candle['close'] - recent_low) / abs(recent_low - last_candle['low'] + 1e-9)
                    result['age_candles'] = len(df) - df.index.get_loc(low_idx) if low_idx in df.index else 0
                    
        elif direction == "SELL":
            # BEARISH SWEEP:
            # Market was going up. Stops are above recent HIGHS.
            # We want to see price SPIKE above a recent high and REVERSE down.
            recent_high = recent['high'].max()
            high_idx = recent['high'].idxmax()
            
            if last_candle['high'] > recent_high:
                if last_candle['close'] < recent_high:
                    result['detected'] = True
                    result['type'] = 'BEARISH_SWEEP'
                    result['level'] = recent_high
                    result['strength'] = abs(recent_high - last_candle['close']) / abs(last_candle['high'] - recent_high + 1e-9)
                    result['age_candles'] = len(df) - df.index.get_loc(high_idx) if high_idx in df.index else 0
                    
        return result
    
    def map_liquidity_pools(self, df: pd.DataFrame) -> list:
        """
        Maps where stop-losses likely cluster:
        - Below recent swing lows (Sell Side Liquidity - SSL)
        - Above recent swing highs (Buy Side Liquidity - BSL)
        - At psychological levels (.00, .50)
        """
        self.pools = []
        
        # Simple Fractal/Swing Detection (Bill Williams or simple neighbor check)
        # 5-candle High/Low
        window = 5
        
        for i in range(window, len(df) - window):
            # Swing Low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                self.pools.append({
                    'price': df['low'].iloc[i],
                    'type': 'SSL', # Sell Side Liquidity (Stops for Longs)
                    'time': df.index[i],
                    'tested': False
                })
                
            # Swing High
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                self.pools.append({
                    'price': df['high'].iloc[i],
                    'type': 'BSL', # Buy Side Liquidity (Stops for Shorts)
                    'time': df.index[i],
                    'tested': False
                })
                
        # Filter pools that have been taken out by subsequent price action
        active_pools = []
        current_price = df['close'].iloc[-1]
        
        for pool in self.pools:
            # Check if price has crossed this level after the pool creation
            # Optimization: In backtest loop, this is heavy. 
            # Simplified: We just return the list, caller manages state if needed.
            # For this prototype, we just return recent pools (last 50 candles).
            time_diff = (df.index[-1] - pool['time']).total_seconds() / 60
            if time_diff < 120: # 2 hours
                active_pools.append(pool)
                
        return active_pools
