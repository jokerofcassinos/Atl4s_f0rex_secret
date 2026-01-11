import logging
from typing import Dict, Any, Optional

import pandas as pd
import ta

from core.agi.module_thought_adapter import AGIModuleAdapter, ModuleThoughtResult

logger = logging.getLogger("Atl4s-Trend")
logger.setLevel(logging.WARNING)


class TrendArchitect:
    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "M5"):
        self.symbol = symbol
        self.timeframe = timeframe
        # Camada AGI: conecta este módulo ao InfiniteWhyEngine
        self.agi_adapter = AGIModuleAdapter(module_name="TrendArchitect")

    def analyze(self, df_m5, df_h1: Optional[pd.DataFrame] = None, df_h4: Optional[pd.DataFrame] = None, df_d1: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyzes Trend using Multi-Timeframe Context (River=H1, Ocean=H4, Galaxy=D1).
        OPTIMIZED: Slices data to last 500 candles to prevent slow TA calc.
        """
        # OPTIMIZATION: Slice input dataframes to constant size (O(1))
        limit = 500
        df = df_m5.iloc[-limit:].copy()
        
        # Overwrite arguments with slices
        if df_h1 is not None: df_h1 = df_h1.iloc[-limit:]
        if df_h4 is not None: df_h4 = df_h4.iloc[-limit:]
        if df_d1 is not None: df_d1 = df_d1.iloc[-limit:]
        
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
        from datetime import datetime
        now_str = datetime.now().strftime("%H:%M:%S")
        logger.info(f"Market Regime: {regime} (ADX: {adx:.2f}) [{now_str}]")

        # --- 2. Determine The River (H1 Context) ---
        river_dir = 0 # 0: Neutral, 1: Bullish, -1: Bearish
        if df_h1 is not None and not df_h1.empty:
            try:
                # Calculate H1 EMA 50 (Faster) and EMA 20 (Fastest) for River Flow
                ema50_h1 = ta.trend.EMAIndicator(df_h1['close'], window=50).ema_indicator().iloc[-1]
                ema20_h1 = ta.trend.EMAIndicator(df_h1['close'], window=20).ema_indicator().iloc[-1]
                current_price_h1 = df_h1['close'].iloc[-1]
                
                # Logic: River is primarily defined by EMA50, but qualified by EMA20 alignment
                # REFINED LOGIC (User Feedback: "Trend is Up" but bot saw Bearish)
                # If Price is above EMA20 (Short Term H1 Trend), we treat it as BULLISH even if below EMA50 (Recovery/Pullback)
                
                if current_price_h1 > ema20_h1:
                    if current_price_h1 > ema50_h1:
                        river_dir = 1 # Strong Bullish (Above Both)
                    else:
                        river_dir = 1 # Weak Bullish (Recovery Phase - Price above fast EMA)
                
                elif current_price_h1 < ema20_h1:
                    if current_price_h1 < ema50_h1:
                        river_dir = -1 # Strong Bearish (Below Both)
                    else:
                        river_dir = -1 # Weak Bearish (Correction Phase - Price below fast EMA)
                
                # EMA Cross Confirmation (Gold & Death Cross)
                if ema20_h1 > ema50_h1 and river_dir == 1:
                     river_dir = 1 # Confirmed Bullish
                elif ema20_h1 < ema50_h1 and river_dir == -1:
                     river_dir = -1 # Confirmed Bearish

                logger.info(f"The River (H1 Trend): {'BULLISH' if river_dir == 1 else 'BEARISH' if river_dir == -1 else 'NEUTRAL'} | P: {current_price_h1:.2f} | EMA20: {ema20_h1:.2f} | EMA50: {ema50_h1:.2f}")
            except Exception as e:
                logger.error(f"Error calculating H1 River: {e}")

        # --- 2.5. Determine The Ocean (H4 Context) - SNIPER FILTER ---
        # PROTOCOL LION UPDATE: Never return 0 if Price Action is clear.
        ocean_dir = 0
        if df_h4 is not None and len(df_h4) > 20:
            try:
                # Prioritize PRICE ACTION over EMA
                current_price = df_h4['close'].iloc[-1]
                ema50 = ta.trend.EMAIndicator(df_h4['close'], window=50).ema_indicator().iloc[-1]
                
                # Check Last 3 Candles for Direction
                c1 = df_h4.iloc[-1]['close']
                c3 = df_h4.iloc[-3]['close']
                
                if current_price > ema50:
                    ocean_dir = 1
                elif current_price < ema50:
                    ocean_dir = -1
                
                # Fallback: If EMA is flat, look at displacement
                if abs(current_price - ema50) / current_price < 0.0005: # Very close
                    if c1 > c3: ocean_dir = 1
                    else: ocean_dir = -1
                    
                logger.info(f"The Ocean (H4 Trend): {'BULLISH' if ocean_dir == 1 else 'BEARISH' if ocean_dir == -1 else 'NEUTRAL'}")
            except Exception as e:
                logger.error(f"Error calculating H4 Ocean: {e}")

        # --- 2.8. Determine The Galaxy (D1 Context) - FORTRESS FILTER ---
        galaxy_dir = 0
        if df_d1 is not None and len(df_d1) > 20:
            try:
                # D1 is slow, so we check EMA 20 for mid-term bias
                ema20_d1 = ta.trend.EMAIndicator(df_d1['close'], window=20).ema_indicator().iloc[-1]
                current_price_d1 = df_d1['close'].iloc[-1]
                
                if current_price_d1 > ema20_d1:
                    galaxy_dir = 1
                elif current_price_d1 < ema20_d1:
                    galaxy_dir = -1
                    
                logger.info(f"The Galaxy (D1 Trend): {'BULLISH' if galaxy_dir == 1 else 'BEARISH' if galaxy_dir == -1 else 'NEUTRAL'}")
            except Exception as e:
                logger.error(f"Error calculating D1 Galaxy: {e}")

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
                # logger.debug("M5 Trend opposes H1 River. Applying Penalty (Counter-Trend).")
                score -= 20 # Penalty instead of Veto
                # direction remains as is (Counter-Trend Trade possible)
            else:
                score += 20 # Boost for aligning with River
        
        # If Regime is RANGING, we ignore the River and trust the M5 Reversals ( handled by Quant mostly)
        # But Trend Architect just reports what it sees on M5.
 
        raw_output: Dict[str, Any] = {
            "score": min(score, 100),
            "direction": direction,
            "regime": regime,
            "river": river_dir,
            "ocean": ocean_dir,
            "galaxy": galaxy_dir
        }

        # --- 5. Camada de Pensamento Recursivo (Fase 8) ---
        market_state: Dict[str, Any] = {
            "price": float(close),
            "ema20_m5": float(ema20),
            "ema50_m5": float(ema50),
            "adx": float(adx),
            "river_dir": int(river_dir),
        }

        # AGI Integration - DISABLED FOR BACKTEST PERFORMANCE
        # TODO Phase 2: Re-enable for live trading
        # thought: ModuleThoughtResult = self.agi_adapter.think_on_analysis(...)
        # enriched = dict(raw_output)
        # enriched["agi_decision"] = thought.decision
        # ...

        return raw_output
