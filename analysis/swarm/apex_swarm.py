
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from core.interfaces import SwarmSignal, SubconsciousUnit
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

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
        self.agi_adapter = AGISwarmAdapter("ApexSwarm")

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
        
        if not basket:
            return None

        scores: Dict[str, float] = {}
        
        for symbol, df in basket.items():
            if df is None or len(df) < 50: continue
            
            # --- AGGRESSIVE SCORING ENGINE v2 ---
            score = 0.0
            score_breakdown = {}
            
            # 1. Trend Quality (Hurst) - Always contribute points
            hurst = self._calculate_hurst(df['close'].tail(50).values)
            hurst = max(0.0, min(1.0, hurst))
            
            # Base 2 points + bonus for any extreme
            hurst_score = 2.0
            if hurst > 0.55:  # Trending
                hurst_score += (hurst - 0.55) * 10.0  # Up to +4.5 pts
            elif hurst < 0.45:  # Mean reverting
                hurst_score += (0.45 - hurst) * 8.0   # Up to +3.6 pts
            score += hurst_score
            score_breakdown['hurst'] = f"{hurst:.2f}(+{hurst_score:.1f})"
            
            # 2. Trend Strength - 50 bar lookback, aggressive scaling
            price_now = df['close'].iloc[-1]
            price_50_ago = df['close'].iloc[-50] if len(df) >= 50 else df['close'].iloc[0]
            trend_pct = abs((price_now - price_50_ago) / price_50_ago) * 100
            
            # Scale: 0.2% = 2 pts, 0.5% = 5 pts, 1% = 10 pts (capped at 10)
            trend_score = min(10.0, trend_pct * 10.0)
            score += trend_score
            score_breakdown['trend'] = f"{trend_pct:.2f}%(+{trend_score:.1f})"
            
            # 3. Volatility - 10x multiplier for Forex
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            price = df['close'].iloc[-1]
            atr_pct = (atr / price) * 100
            
            # 0.1% ATR = 1 pt, 0.5% = 5 pts, 1% = 10 pts
            vol_score = min(10.0, atr_pct * 10.0)
            score += vol_score
            score_breakdown['vol'] = f"{atr_pct:.2f}%(+{vol_score:.1f})"
            
            # 4. RSI Extremes - Super aggressive
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rs = gain / loss if loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # Cubic scaling for extremes: RSI 30 = 2.7 pts, RSI 20 = 5.8 pts, RSI 10 = 10 pts
            dist_from_50 = abs(rsi - 50)
            rsi_score = min(10.0, (dist_from_50 / 50.0) ** 2 * 10.0)
            score += rsi_score
            score_breakdown['rsi'] = f"{rsi:.0f}(+{rsi_score:.1f})"
            
            # 5. Momentum Consistency (NEW) - Count directional bars
            closes = df['close'].tail(20).values
            up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
            down_bars = 20 - up_bars
            consistency = abs(up_bars - down_bars) / 20.0  # 0 to 1
            momentum_score = consistency * 5.0  # Up to 5 pts
            score += momentum_score
            score_breakdown['momentum'] = f"{consistency:.0%}(+{momentum_score:.1f})"
            
            # logger.debug(f"APEX DETAIL: {symbol} -> {' '.join([f'{k}={v}' for k,v in score_breakdown.items()])} = {score:.1f}")  # Silenced - too spammy
            
            scores[symbol] = score
            
        if not scores:
            return None
        
        # Identify Winner
        king_symbol = max(scores, key=scores.get)
        king_score = scores[king_symbol]
        
        # Current Symbol Score
        current_score = scores.get(current_symbol, 0.0)
        
        # Threshold for Substitution
        # Lowered to 5% to be more agile between BTC/ETH
        significant_gap = (king_score > current_score * 1.05)
        current_is_trash = (current_score < 1.0)
        
        if significant_gap:
             # logger.debug(f"APEX SCORES: {scores} | Current: {current_symbol} ({current_score:.2f}) | King: {king_symbol} ({king_score:.2f})")
             pass
        
        if (significant_gap or current_is_trash) and king_symbol != current_symbol:
            reason = f"Routing to {king_symbol} (Score {king_score:.2f} vs {current_score:.2f})"

            # Phase 9: Think as a swarm about asset selection decision
            symbol = current_symbol or king_symbol
            timeframe = context.get("timeframe", "M5")
            market_state = {
                "king_symbol": king_symbol,
                "current_symbol": current_symbol,
                "king_score": float(king_score),
                "current_score": float(current_score),
            }
            swarm_output = {
                "decision": "ROUTING",
                "score": king_score,
                "scores": scores,
                "reason": reason,
                "aggregated_signal": king_symbol,
            }
            swarm_thought: SwarmThoughtResult = self.agi_adapter.think_on_swarm_output(
                symbol=symbol,
                timeframe=timeframe,
                market_state=market_state,
                swarm_output=swarm_output,
            )

            return SwarmSignal(
                source="ApexSwarm",
                signal_type="ROUTING", # Special Type
                confidence=100.0,
                timestamp=0,
                meta_data={
                    "reason": reason, 
                    "best_asset": king_symbol, 
                    "scores": scores,
                    "action": "SWITCH_FOCUS",
                    "agi_thought_root_id": swarm_thought.thought_root_id,
                    "agi_scenarios": swarm_thought.meta.get("scenario_count", 0),
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
