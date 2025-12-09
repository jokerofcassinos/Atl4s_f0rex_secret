import pandas as pd
import ta
import logging

logger = logging.getLogger("Atl4s-Trend")

class TrendArchitect:
    def __init__(self):
        pass

    def analyze(self, df):
        """
        Analyzes the trend structure.
        Returns:
            score (int): Confidence score (0-100)
            direction (int): 1 (Buy), -1 (Sell), 0 (Neutral)
        """
        if df is None or len(df) < 200:
            logger.warning("Insufficient data for Trend Analysis")
            return 0, 0

        df = df.copy()

        # Calculate Indicators using 'ta' library
        
        # Ensure inputs are 1D Series
        close_series = df['close'].squeeze()
        high_series = df['high'].squeeze()
        low_series = df['low'].squeeze()
        
        if isinstance(close_series, pd.DataFrame): close_series = close_series.iloc[:, 0]
        if isinstance(high_series, pd.DataFrame): high_series = high_series.iloc[:, 0]
        if isinstance(low_series, pd.DataFrame): low_series = low_series.iloc[:, 0]

        # EMA
        df['EMA_50'] = ta.trend.EMAIndicator(close=close_series, window=50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(close=close_series, window=200).ema_indicator()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(high=high_series, low=low_series, window1=9, window2=26, window3=52)
import pandas as pd
import ta
import logging

logger = logging.getLogger("Atl4s-Trend")

class TrendArchitect:
    def __init__(self):
        pass

    def analyze(self, df_m5, df_h1=None):
        """
        Analyzes Trend using Multi-Timeframe Context (The River) and ADX Regime.
        """
        df = df_m5.copy()
        score = 0
        direction = 0
        
        # --- 1. Determine Regime (ADX) ---
        # ADX > 25 = Trending, ADX < 25 = Ranging
        adx_len = 14
        try:
            adx_df = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=adx_len)
            adx = adx_df.adx().iloc[-1]
        except:
            adx = 20.0 # Default to weak trend if error
            
        regime = "TRENDING" if adx > 25 else "RANGING"
        logger.info(f"Market Regime: {regime} (ADX: {adx:.2f})")

        # --- 2. Determine The River (H1 Context) ---
        river_dir = 0 # 0: Neutral, 1: Bullish, -1: Bearish
        if df_h1 is not None and not df_h1.empty:
            try:
                # Calculate H1 EMA 200
                ema200_h1 = ta.trend.EMAIndicator(df_h1['close'], window=200).ema_indicator().iloc[-1]
                current_price_h1 = df_h1['close'].iloc[-1]
                
                if current_price_h1 > ema200_h1:
                    river_dir = 1
                else:
                    river_dir = -1
                logger.info(f"The River (H1 Trend): {'BULLISH' if river_dir == 1 else 'BEARISH'}")
            except Exception as e:
                logger.error(f"Error calculating H1 River: {e}")

        # --- 3. M5 Trend Analysis (Micro Structure) ---
        # EMA 20 vs 50
        ema20 = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1]
        ema50 = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        span_a = ichimoku.ichimoku_a().iloc[-1]
        span_b = ichimoku.ichimoku_b().iloc[-1]
        close = df['close'].iloc[-1]

        # Scoring Logic
        if ema20 > ema50:
            score += 30
            direction = 1
        elif ema20 < ema50:
            score += 30
            direction = -1
            
        if close > span_a and close > span_b:
            score += 20
            if direction == 1: score += 10 # Confluence
        elif close < span_a and close < span_b:
            score += 20
            if direction == -1: score += 10 # Confluence

        # --- 4. Final Synthesis ---
        # If Regime is TRENDING, we heavily penalize fighting the River
        if regime == "TRENDING" and river_dir != 0:
            if direction != river_dir:
                logger.warning("M5 Trend opposes H1 River. Applying Penalty (Counter-Trend).")
                score -= 20 # Penalty instead of Veto
                # direction remains as is (Counter-Trend Trade possible)
            else:
                score += 20 # Boost for aligning with River
        
        # If Regime is RANGING, we ignore the River and trust the M5 Reversals (handled by Quant mostly)
        # But Trend Architect just reports what it sees on M5.
        
        return {
            "score": min(score, 100), 
            "direction": direction, 
            "regime": regime,
            "river": river_dir
        }
