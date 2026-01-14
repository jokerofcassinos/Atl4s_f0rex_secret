import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Atl4s-Fractal")

class FractalVision:
    def __init__(self):
        pass

    def calculate_heikin_ashi(self, df):
        """
        Calculates Heikin Ashi candles.
        HA_Close = (Open + High + Low + Close) / 4
        HA_Open = (Prev_HA_Open + Prev_HA_Close) / 2
        HA_High = Max(High, HA_Open, HA_Close)
        HA_Low = Min(Low, HA_Open, HA_Close)
        """
        if df is None or df.empty:
            return None

        ha_df = df.copy()
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # HA Open requires iteration (or numba/cython for speed, but loop is fine for M5/H1 size)
        # We can use a vectorized approximation or just loop. Loop is safer for correctness.
        ha_open = [df['open'].iloc[0]]
        ha_close = ha_df['ha_close'].values
        
        for i in range(1, len(df)):
            ha_open.append((ha_open[-1] + ha_close[i-1]) / 2)
            
        ha_df['ha_open'] = ha_open
        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        return ha_df

    def identify_swings(self, df, length=2):
        """
        Identifies Fractal Highs and Lows (Bill Williams style or Pivot).
        A fractal high is a high surrounded by 'length' lower highs on both sides.
        """
        # FIX: Check for None/empty BEFORE accessing columns
        if df is None or df.empty or len(df) < length * 2 + 1:
            # Return empty DataFrame with required columns
            if df is None:
                df = pd.DataFrame()
            df['is_swing_high'] = False
            df['is_swing_low'] = False
            return df
        
        try:
            df['is_swing_high'] = False
            df['is_swing_low'] = False
            
            # Let's use a simple pivot detection
            # Pivot High
            is_high = pd.Series(True, index=df.index)
            is_low = pd.Series(True, index=df.index)
            
            for i in range(1, length + 1):
                is_high &= (df['high'] > df['high'].shift(i)) & (df['high'] > df['high'].shift(-i))
                is_low &= (df['low'] < df['low'].shift(i)) & (df['low'] < df['low'].shift(-i))
                
            df['is_swing_high'] = is_high
            df['is_swing_low'] = is_low
        except Exception as e:
            logger.warning(f"Error in identify_swings: {e}")
            df['is_swing_high'] = False
            df['is_swing_low'] = False
        
        return df


    def analyze_structure(self, df):
        """
        Analyzes Market Structure (BOS, CHoCH).
        Returns: structure_status (BULLISH/BEARISH), last_bos_level
        """
        if df is None or df.empty:
            return "NEUTRAL", 0
            
        # Get recent swings
        swings = self.identify_swings(df.copy(), length=3) # Use length 3 for significance
        
        last_highs = swings[swings['is_swing_high']]
        last_lows = swings[swings['is_swing_low']]
        
        if last_highs.empty or last_lows.empty:
            return "NEUTRAL", 0
            
        # Most recent significant structure points
        last_swing_high = last_highs.iloc[-1]['high']
        last_swing_low = last_lows.iloc[-1]['low']
        
        current_close = df.iloc[-1]['close']
        
        # Simple BOS Logic
        # If we broke the last swing high -> Bullish BOS
        # If we broke the last swing low -> Bearish BOS
        
        structure = "RANGING"
        level = 0
        
        # Check Break of Structure
        # We need to see if the break happened recently
        
        if current_close > last_swing_high:
            structure = "BULLISH_BOS"
            level = last_swing_high
        elif current_close < last_swing_low:
            structure = "BEARISH_BOS"
            level = last_swing_low
        else:
            # Check previous trend context
            # If last swing high > prev swing high -> Bullish Structure
            if len(last_highs) > 1 and last_highs.iloc[-1]['high'] > last_highs.iloc[-2]['high']:
                 structure = "BULLISH_STRUCTURE"
            elif len(last_lows) > 1 and last_lows.iloc[-1]['low'] < last_lows.iloc[-2]['low']:
                 structure = "BEARISH_STRUCTURE"
                 
        return structure, level

    def analyze(self, df_h1, df_h4):
        """
        Main analysis function.
        """
        results = {
            'h1_structure': 'NEUTRAL',
            'h4_structure': 'NEUTRAL',
            'ha_trend': 'NEUTRAL',
            'score': 0
        }

        # Null checks for robust operation
        if df_h1 is None or df_h1.empty:
            return results
        if df_h4 is None:
            df_h4 = pd.DataFrame() # Fallback to empty instead of None
        
        # 1. Heikin Ashi Trend (H1)
        ha_df = self.calculate_heikin_ashi(df_h1)
        if ha_df is not None:
            last_ha = ha_df.iloc[-1]
            prev_ha = ha_df.iloc[-2]
            
            # Green HA
            if last_ha['ha_close'] > last_ha['ha_open']:
                results['ha_trend'] = "BULLISH"
                # Strong Bullish if no lower wick
                if last_ha['ha_low'] == last_ha['ha_open']:
                     results['ha_trend'] = "STRONG_BULLISH"
            else:
                results['ha_trend'] = "BEARISH"
                # Strong Bearish if no upper wick
                if last_ha['ha_high'] == last_ha['ha_open']:
                     results['ha_trend'] = "STRONG_BEARISH"
                     
        # 2. Market Structure (H1 & H4)
        h1_struct, h1_lvl = self.analyze_structure(df_h1)
        h4_struct, h4_lvl = self.analyze_structure(df_h4)
        
        results['h1_structure'] = h1_struct
        results['h4_structure'] = h4_struct
        
        # Scoring
        score = 0
        
        # H4 Structure is King
        if "BULLISH" in h4_struct: score += 40
        elif "BEARISH" in h4_struct: score -= 40
        else:
             # Fallback: EMA Trend
             if len(df_h4) > 50:
                 ema20 = df_h4['close'].ewm(span=20).mean().iloc[-1]
                 ema50 = df_h4['close'].ewm(span=50).mean().iloc[-1]
                 if ema20 > ema50: score += 20
                 elif ema20 < ema50: score -= 20
        
        # H1 Structure Confirmation
        if "BULLISH" in h1_struct: score += 20
        elif "BEARISH" in h1_struct: score -= 20
        else:
             # Fallback: EMA Trend
             if len(df_h1) > 50:
                 ema20 = df_h1['close'].ewm(span=20).mean().iloc[-1]
                 ema50 = df_h1['close'].ewm(span=50).mean().iloc[-1]
                 if ema20 > ema50: score += 10
                 elif ema20 < ema50: score -= 10
        
        # HA Trend Confirmation
        if "BULLISH" in results['ha_trend']: score += 10
        elif "BEARISH" in results['ha_trend']: score -= 10
        
        results['score'] = score
        
        return results
