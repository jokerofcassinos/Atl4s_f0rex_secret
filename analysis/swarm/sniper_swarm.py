
import logging
from typing import Dict, Any, List

from core.interfaces import SubconsciousUnit, SwarmSignal
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

logger = logging.getLogger("SniperSwarm")

class SniperSwarm(SubconsciousUnit):
    """
    Precision Entry Swarm.
    Role: Identify micro-structure setups (Wicks, Order Blocks, Liquidity Voids).
    """
    def __init__(self):
        super().__init__("Sniper_Swarm")
        self.sub_agents = [
            self._agent_wick_hunter,
            self._agent_imbalance_filler
        ]
        self.agi_adapter = AGISwarmAdapter("Sniper_Swarm")
        
    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        tick = context.get('tick')
        
        if df_m1 is None or tick is None:
            return None

        # Run sub-agents
        votes: List[Dict[str, Any]] = []
        for agent in self.sub_agents:
            vote = agent(df_m1, tick)
            if vote: votes.append(vote)
            
        if not votes:
            # logger.debug("Sniper Scan: No patterns found.")
            return None

        # Consensus
        total_conf = sum(v["confidence"] for v in votes)
        avg_conf = total_conf / len(votes)

        logger.info("Sniper Scan: Found %d signals. Avg Conf: %.1f", len(votes), avg_conf)

        if avg_conf > 70:
            primary_vote = votes[0]

            # Phase 9: Swarm thought about micro-structure pattern
            symbol = tick.get("symbol", "UNKNOWN")
            timeframe = "M1"
            market_state = {
                "price": float(df_m1["close"].iloc[-1]),
                "wick_count": len(votes),
            }
            swarm_output = {
                "decision": primary_vote["type"],
                "score": avg_conf,
                "votes": votes,
                "reason": primary_vote.get("reason", ""),
                "aggregated_signal": primary_vote["type"],
            }
            swarm_thought: SwarmThoughtResult = self.agi_adapter.think_on_swarm_output(
                symbol=symbol,
                timeframe=timeframe,
                market_state=market_state,
                swarm_output=swarm_output,
            )

            return SwarmSignal(
                source=self.name,
                signal_type=primary_vote["type"],  # Assuming alignment
                confidence=avg_conf,
                timestamp=0,  # Filled by Bus
                meta_data={
                    "reason": primary_vote.get("reason", ""),
                    "agi_thought_root_id": swarm_thought.thought_root_id,
                    "agi_scenarios": swarm_thought.meta.get("scenario_count", 0),
                },
            )
        return None

    def _agent_wick_hunter(self, df, tick):
        """Detects long wicks rejection."""
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # Rejection Logic
        if lower_wick > (body * 3): # Hammer
            return {'type': 'BUY', 'confidence': 85.0, 'reason': 'Hammer Wick Rejection'}
        if upper_wick > (body * 3): # Shooting Star
            return {'type': 'SELL', 'confidence': 85.0, 'reason': 'Star Wick Rejection'}
            
        return None

    def _agent_imbalance_filler(self, df, tick):
        """Detects FVG fills."""
        # Simplified FVG logic
        return None
