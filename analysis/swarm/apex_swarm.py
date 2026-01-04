
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("ApexSwarm")

class ApexSwarm(SubconsciousUnit):
    """
    The Apex Predator (Asset Selector).
    Phase 30 Innovation.
    Logic:
    1. Scans a basket of assets (provided in context via DataLoader).
    2. Scores each asset:
       - Trend Quality (Hurst > 0.6 is good).
       - Volatility (ATR% needs to be sufficient for profit).
       - Momentum (RSI not neutral).
    3. Identifies the 'King Asset' (Best opportunity).
    """
    def __init__(self):
        super().__init__("ApexSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        data_map = context.get('data_map', {})
        # Apex looks at 'global_basket' which should contain candidates
        basket = data_map.get('global_basket', {})
        
        # Add the current primary symbol to the comparison if data exists
        current_symbol = context.get('tick', {}).get('symbol')
        df_m5_current = data_map.get('M5')
        
        if current_symbol:
             # Ensure current is in basket
             if current_symbol not in basket and df_m5_current is not None:
                 basket[current_symbol] = df_m5_current
             
             # If basket is basically empty or garbage, focus entirely on current
             if len(basket) == 0 and df_m5_current is not None:
                 basket = {current_symbol: df_m5_current}
        
        if not basket: return None
        
        scores = {}
        
        for symbol, df in basket.items():
            if df is None or len(df) < 50: continue
            
            # --- SCORING ENGINE ---
            score = 0.0
            
            # 1. Trend Quality (Hurst) - Using H1 data
            hurst = self._calculate_hurst(df['close'].tail(50).values)
            if hurst > 0.6: score += 2.0
            if hurst > 0.7: score += 1.0 # Bonus
            if hurst < 0.4: score += 1.0 # High Mean Reversion Potential
            
            # 2. Volatility (Opportunity Size)
            # ATR relative to Price
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            price = df['close'].iloc[-1]
            atr_pct = (atr / price) * 100
            
            # We like volatility. 
            score += atr_pct * 2.0 # Volatility drives score up
            
            # 3. Momentum Clarity (RSI)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # Extreme RSI is actionable -> Score up
            # Mid RSI (50) is noise -> Score down
            dist_from_50 = abs(rsi - 50)
            score += (dist_from_50 / 50.0) # 0 to 1 point
            
            scores[symbol] = score
            
        if not scores: return None
        
        # Identify Winner
        king_symbol = max(scores, key=scores.get)
        king_score = scores[king_symbol]
        
        # Current Symbol Score
        current_score = scores.get(current_symbol, 0.0)
        
        # Threshold for Substitution
        # Lowered to 5% to be more agile between BTC/ETH
        significant_gap = (king_score > current_score * 1.05)
        current_is_trash = (current_score < 1.0)
        
        logger.info(f"APEX SCORES: {scores} | Current: {current_symbol} ({current_score:.2f}) | King: {king_symbol} ({king_score:.2f})")
        
        if (significant_gap or current_is_trash) and king_symbol != current_symbol:
            reason = f"Routing to {king_symbol} (Score {king_score:.2f} vs {current_score:.2f})"
            
            return SwarmSignal(
                source="ApexSwarm",
                signal_type="ROUTING", # Special Type
                confidence=100.0,
                timestamp=0,
                meta_data={
                    "reason": reason, 
                    "best_asset": king_symbol, 
                    "scores": scores,
                    "action": "SWITCH_FOCUS"
                }
            )
            
        return None

    def _calculate_hurst(self, series):
        try:
            if len(series) < 20: return 0.5
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            if not tau or any(t == 0 for t in tau): return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
