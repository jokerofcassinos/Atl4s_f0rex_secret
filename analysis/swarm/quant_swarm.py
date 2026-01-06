
import logging
from typing import Dict, Any

import numpy as np

from core.interfaces import SubconsciousUnit, SwarmSignal
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

logger = logging.getLogger("QuantSwarm")

class QuantSwarm(SubconsciousUnit):
    """
    Mathematical Probability Engine.
    Role: Apply Physics/Stats to validate moves.
    """
    def __init__(self):
        super().__init__("Quant_Swarm")
        self.agi_adapter = AGISwarmAdapter("Quant_Swarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 100:
            return None

        # 1. Z-Score (Mean Reversion)
        # Is price statistically overextended?
        close = df_m5['close']
        window = 50
        roll_mean = close.rolling(window).mean().iloc[-1]
        roll_std = close.rolling(window).std().iloc[-1]
        price = close.iloc[-1]
        
        z_score = (price - roll_mean) / (roll_std + 1e-9)
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Extreme Extension -> Reversion Likely
        if z_score > 3.0:
            signal = "SELL"
            confidence = 85.0
            reason = f"Z-Score Extreme (+{z_score:.2f}) - Statistical Reversion"
        elif z_score < -3.0:
            signal = "BUY"
            confidence = 85.0
            reason = f"Z-Score Extreme ({z_score:.2f}) - Statistical Reversion"
            
        # 2. Volatility Breakout (Expansion)
        # If Z-Score is just starting to expand (e.g. 1.5) AND volume is high -> Continuation
        
        if signal != "WAIT":
            # Phase 9: Quantitative swarm thought about statistical extremes
            symbol = context.get("symbol", "UNKNOWN")
            timeframe = context.get("timeframe", "M5")
            market_state = {
                "price": float(price),
                "roll_mean": float(roll_mean),
                "roll_std": float(roll_std),
                "z_score": float(z_score),
            }
            swarm_output = {
                "decision": signal,
                "score": confidence,
                "reason": reason,
                "z_score": float(z_score),
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
                confidence=confidence,
                timestamp=0,
                meta_data={
                    "reason": reason,
                    "z_score": z_score,
                    "agi_thought_root_id": swarm_thought.thought_root_id,
                    "agi_scenarios": swarm_thought.meta.get("scenario_count", 0),
                },
            )

        return None
