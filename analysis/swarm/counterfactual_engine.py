
import logging

import numpy as np

from core.interfaces import SubconsciousUnit, SwarmSignal
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

logger = logging.getLogger("CounterfactualEngine")

class CounterfactualEngine(SubconsciousUnit):
    """
    The Imagination.
    "Would I have taken this trade if X was different?"
    Acts as a Meta-Validator.
    """
    def __init__(self):
        super().__init__("Counterfactual_Engine")
        self.agi_adapter = AGISwarmAdapter("Counterfactual_Engine")

    async def process(self, context) -> SwarmSignal:
        # Counterfactuals usually run *after* a draft decision is made, 
        # but as a Swarm Unit, we can analyze the *robustness* of the current setup.
        
        df_m5 = context.get('df_m5')
        if df_m5 is None:
            return None

        # 1. Base Scenario
        last_close = df_m5['close'].iloc[-1]
        rsi_proxy = self._calc_rsi_proxy(df_m5)
        
        # 2. Counterfactual: do(Price = Price - 0.1%)
        # If a small drop kills the signal (e.g. RSI goes from 71 to 69), the signal is fragile.
        
        robustness = 0
        
        # Test 1: Noise Sensitivity
        # If we add noise to the last 5 candles, does the trend direction change?
        # (Simplified heuristic)
        
        trend_score = (last_close - df_m5['close'].iloc[-5])
        
        # Sensitivity
        noise = (df_m5['high'].iloc[-1] - df_m5['low'].iloc[-1]) * 0.5
        
        conf_boost = 0
        if abs(trend_score) > noise:
            # The trend is stronger than the noise.
            robustness += 1
            conf_boost = 10
        else:
            # Fragile trend.
            # "If spread was slightly wider, this trade is a loss"
            robustness -= 1
            
        signal = "WAIT"
        # We act as a filter/booster
        if robustness > 0:
            # We confirm the dominant direction
            direction = "BUY" if trend_score > 0 else "SELL"
            signal = direction
            reason = "Robust to Noise Intervention"

        if signal != "WAIT":
            # Phase 9: Counterfactual imagination as explicit AGI thought node
            symbol = context.get("symbol", "UNKNOWN")
            timeframe = context.get("timeframe", "M5")
            market_state = {
                "price": float(last_close),
                "trend_score": float(trend_score),
                "noise": float(noise),
                "robustness": int(robustness),
            }
            swarm_output = {
                "decision": signal,
                "score": 50 + conf_boost,
                "reason": reason,
                "robustness": robustness,
                "aggregated_signal": signal,
            }
            swarm_thought: SwarmThoughtResult = self.agi_adapter.think_on_swarm_output(
                symbol=symbol,
                timeframe=timeframe,
                market_state=market_state,
                swarm_output=swarm_output,
            )

            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=50 + conf_boost, # Low base confidence, mostly a booster
                timestamp=0,
                meta_data={
                    'robustness': robustness,
                    'agi_thought_root_id': swarm_thought.thought_root_id,
                    'agi_scenarios': swarm_thought.meta.get("scenario_count", 0),
                }
            )
            
        return None

    def _calc_rsi_proxy(self, df):
        # Quick calc
        return 50 # Placeholder
