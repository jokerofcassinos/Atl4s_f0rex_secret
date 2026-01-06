import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from core.agi.module_thought_adapter import AGIModuleAdapter, ModuleThoughtResult

logger = logging.getLogger("Atl4s-Sniper")

class LevelMemory:
    """
    Holographic Memory for Support/Resistance.
    Decays levels over time or touches.
    """
    def __init__(self):
        self.levels = [] # List of dicts: {'price': float, 'type': int, 'strength': float, 'created_at': int}
        self.max_levels = 20
        
    def add_level(self, price, l_type, strength, timestamp):
        # Check for duplicates (nearby levels)
        for lvl in self.levels:
            if abs(lvl['price'] - price) < 0.5: # 50 cents tolerance (Gold)
                lvl['strength'] = max(lvl['strength'], strength) # Reinforce
                lvl['created_at'] = timestamp # Refresh
                return
        
        self.levels.append({
            'price': price, 
            'type': l_type, # 1=Support, -1=Resistance
            'strength': strength, 
            'created_at': timestamp,
            'touches': 0
        })
        
        # Keep memory clean
        if len(self.levels) > self.max_levels:
            # Remove weakest/oldest
            self.levels.sort(key=lambda x: x['strength'], reverse=True)
            self.levels.pop()
            
    def decay(self, current_price, current_time):
        # Decay logic: 
        # 1. Time Decay (Linear)
        # 2. Touch Decay (Each touch reduces strength)
        active_levels = []
        for lvl in self.levels:
            # Check Touch (Price wicks into level)
            dist = abs(current_price - lvl['price'])
            if dist < 1.0: # Close proximity
                 lvl['touches'] += 1
                 lvl['strength'] -= 5.0 # Reduce strength per touch
                 
            # Time Decay (e.g. -1 strength per minute approx, assuming logic runs per candle)
            lvl['strength'] -= 0.1 
            
            if lvl['strength'] > 10.0:
                active_levels.append(lvl)
                
        self.levels = active_levels
        return self.levels

class Sniper:
    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "M5"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.memory = LevelMemory()
        self.agi_adapter = AGIModuleAdapter(module_name="Sniper")

    def analyze(self, df):
        """
        Legacy interface: returns (score, direction) for compatibility.
        """
        result = self.analyze_with_thought(df)
        return result["score"], result["direction"]

    def analyze_with_thought(self, df) -> Dict[str, Any]:
        """
        Fase 8: Analyzes Market Structure (FVG, Liquidity) + camada AGI.
        """
        if df is None or len(df) < 5:
            return {"score": 0, "direction": 0}

        score = 0
        direction = 0
        current_time = df.index[-1].timestamp() if hasattr(df.index[-1], "timestamp") else 0
        current_price = float(df.iloc[-1]["close"])

        # 1. Update Holographic Memory
        self.memory.decay(current_price, current_time)
        
        # 2. Identify New FVGs & Breakaway Gaps
        # Bullish FVG: Low[i] > High[i-2]
        lookback = 10
        start_idx = max(0, len(df) - lookback)
        
        for i in range(start_idx + 2, len(df)):
             # Calculate Candle Velocity (Kinematics approximation)
             # Body size / time? Just use Body Size as proxy for now.
             body = abs(df.iloc[i-1]['close'] - df.iloc[i-1]['open'])
             avg_body = (df.iloc[start_idx:i]['high'] - df.iloc[start_idx:i]['low']).mean()
             is_high_velocity = body > (avg_body * 1.5)
             
             # Bullish FVG
             if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                 gap_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
                 if gap_size > 0.05:
                     strength = 50 if is_high_velocity else 30 # Breakaway Gaps are stronger
                     # Add Support Level (Bottom of Gap and Top of Gap are both levels, mostly Top)
                     self.memory.add_level(df.iloc[i-2]['high'], 1, strength, current_time)
                     self.memory.add_level(df.iloc[i]['low'], 1, strength, current_time) # Top of support zone

             # Bearish FVG
             elif df.iloc[i]['high'] < df.iloc[i-2]['low']:
                 gap_size = df.iloc[i-2]['low'] - df.iloc[i]['high']
                 if gap_size > 0.05:
                     strength = 50 if is_high_velocity else 30
                     self.memory.add_level(df.iloc[i-2]['low'], -1, strength, current_time)
                     self.memory.add_level(df.iloc[i]['high'], -1, strength, current_time)

        # 3. Evaluate Price vs Memory
        # Check if we are bouncing off a memory level
        for lvl in self.memory.levels:
            dist = abs(current_price - lvl['price'])
            
            if dist < 1.0: # Interaction Zone
                if lvl['type'] == 1: # Support
                    # Price at Support -> Buy Signal
                    # Only if we are NOT smashing through it (Velocity check needed externally or via micro)
                    score += lvl['strength']
                    direction = 1
                    logger.info(f"Sniper: HOLOGRAPHIC SUPPORT at {lvl['price']:.2f} (Str {lvl['strength']:.1f})")
                    
                elif lvl['type'] == -1: # Resistance
                    # Price at Resistance -> Sell Signal
                    score += lvl['strength']
                    direction = -1
                    logger.info(f"Sniper: HOLOGRAPHIC RESISTANCE at {lvl['price']:.2f} (Str {lvl['strength']:.1f})")

        # Cap score
        score = min(score, 100)

        # Fallback: Liquidity Sweep (Turtle Soup)
        if score < 20:
            subset = df.iloc[-15:-2]
            recent_low = subset["low"].min()
            recent_high = subset["high"].max()
            last_low = df.iloc[-1]["low"]
            last_high = df.iloc[-1]["high"]
            last_close = df.iloc[-1]["close"]

            if last_low < recent_low and last_close > recent_low:
                raw_output = {"score": 30, "direction": 1, "pattern": "BullishSweep"}
                return self._wrap_with_thought(df, current_price, raw_output)
            if last_high > recent_high and last_close < recent_high:
                raw_output = {"score": 30, "direction": -1, "pattern": "BearishSweep"}
                return self._wrap_with_thought(df, current_price, raw_output)

        raw_output: Dict[str, Any] = {
            "score": score,
            "direction": direction,
            "levels": list(self.memory.levels),
        }
        return self._wrap_with_thought(df, current_price, raw_output)

    def _wrap_with_thought(self, df, current_price: float, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        market_state: Dict[str, Any] = {
            "price": float(current_price),
            "bar_high": float(df.iloc[-1]["high"]),
            "bar_low": float(df.iloc[-1]["low"]),
            "num_levels": len(self.memory.levels),
        }

        thought: ModuleThoughtResult = self.agi_adapter.think_on_analysis(
            symbol=self.symbol,
            timeframe=self.timeframe,
            market_state=market_state,
            raw_module_output=raw_output,
        )

        enriched = dict(raw_output)
        enriched["agi_decision"] = thought.decision
        enriched["agi_score"] = thought.score
        enriched["thought_root_id"] = thought.thought_root_id
        enriched["agi_meta"] = thought.meta

        return enriched
