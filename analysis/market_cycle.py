import pandas as pd
import logging

logger = logging.getLogger("Atl4s-Cycle")

class MarketCycle:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Analyzes the market cycle phase: Accumulation, Distribution, Manipulation, Expansion.
        Returns:
            phase (str): "ACCUMULATION", "DISTRIBUTION", "MANIPULATION_BUY", "MANIPULATION_SELL", "EXPANSION", "NEUTRAL"
            score (int): Confidence score (0-100)
        """
        if df is None or len(df) < 20:
            return "NEUTRAL", 0
            
        df = df.copy()
        
        # Indicators
        # ADX for Range detection
        if 'ADX_14' not in df.columns:
            # Assuming ADX is calculated elsewhere or we need to calc it.
            # For now, let's assume it's passed or we check volatility via BB Width
            pass
            
        # Bollinger Band Width
        if 'BBU_20_2.0' in df.columns:
            bb_width = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
            is_tight = bb_width.iloc[-1] < bb_width.rolling(20).mean().iloc[-1] * 0.8
        else:
            is_tight = False # Fallback
            
        # Recent Price Action (Last 10 candles)
        recent = df.iloc[-10:]
        highest = recent['high'].max()
        lowest = recent['low'].min()
        current_close = df.iloc[-1]['close']
        
        # 1. Detect Range (Accumulation/Distribution)
        # Tight BB + Low Volatility
        if is_tight:
            # Check if at Swing Low (Accumulation) or High (Distribution)
            # Macro context needed ideally, but locally:
            # If we are in lower 30% of last 50 candles -> Accumulation
            last_50 = df.iloc[-50:]
            range_high = last_50['high'].max()
            range_low = last_50['low'].min()
            
            pos_in_range = (current_close - range_low) / (range_high - range_low) if range_high != range_low else 0.5
            
            if pos_in_range < 0.3:
                return "ACCUMULATION", 50
            elif pos_in_range > 0.7:
                return "DISTRIBUTION", 50
            else:
                return "NEUTRAL", 0

        # 2. Detect Manipulation (Judas Swing)
        # Look at last 3 candles. Did we break a recent low and close back up?
        
        # Swing Low (Support) from candles -5 to -2
        support_candles = df.iloc[-10:-2]
        recent_support = support_candles['low'].min()
        
        # Check if any of last 2 candles broke support but current close is above
        broke_support = df.iloc[-2]['low'] < recent_support or df.iloc[-1]['low'] < recent_support
        back_above = current_close > recent_support
        
        if broke_support and back_above:
            # Bullish Manipulation (Liquidity Grab)
            logger.info(f"Cycle: Bullish Manipulation Detected (Support {recent_support:.2f} grabbed)")
            return "MANIPULATION_BUY", 90
            
        # Swing High (Resistance)
        resistance_candles = df.iloc[-10:-2]
        recent_resistance = resistance_candles['high'].max()
        
        broke_res = df.iloc[-2]['high'] > recent_resistance or df.iloc[-1]['high'] > recent_resistance
        back_below = current_close < recent_resistance
        
        if broke_res and back_below:
            # Bearish Manipulation
            logger.info(f"Cycle: Bearish Manipulation Detected (Resistance {recent_resistance:.2f} grabbed)")
            return "MANIPULATION_SELL", 90
            
        # 3. Detect Expansion (Trending)
        # If not ranging and not manipulating, maybe we are trending?
        # Simple check: Price > SMA20 > SMA50
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        
        if current_close > sma20 > sma50:
             return "EXPANSION_BUY", 30
        elif current_close < sma20 < sma50:
             return "EXPANSION_SELL", 30

        return "NEUTRAL", 0
