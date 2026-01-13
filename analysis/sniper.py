import logging
import pandas as pd
from typing import Dict, Any, Tuple
from signals.structure import SMCAnalyzer
from core.agi.module_thought_adapter import AGIModuleAdapter, ModuleThoughtResult

logger = logging.getLogger("Atl4s-Sniper")

class Sniper:
    """
    Sniper Module (Phase 8 Refactor)
    Wraps the unified SMCAnalyzer (from signals.structure) to provide 
    institutional order block and FVG analysis to the Consensus Engine.
    
    Replaces legacy hardcoded thresholds with dynamic SMC logic.
    """
    def __init__(self, symbol: str = "UNKNOWN", timeframe: str = "M5"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.smc_analyzer = SMCAnalyzer()
        self.agi_adapter = AGIModuleAdapter(module_name="Sniper")
        
        # Legacy compatibility - kept for interface, but logic delegated to SMCAnalyzer
        self.memory = None 

    def analyze(self, df: pd.DataFrame) -> Tuple[float, int]:
        """
        Legacy interface: returns (score, direction).
        """
        result = self.analyze_with_thought(df)
        return result["score"], result["direction"]

    def analyze_with_thought(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes SMC Analysis and wraps it with AGI Thought.
        """
        if df is None or len(df) < 20:
            return {"score": 0, "direction": 0}

        current_price = float(df['close'].iloc[-1])
        
        # 1. Execute Unified SMC Analysis
        # This detects FVG, Order Blocks, Liquidity Pools, and Entries
        smc_result = self.smc_analyzer.analyze(df, current_price)
        
        # 2. Extract Signal
        entry_signal = smc_result.get('entry_signal', {})
        direction_str = entry_signal.get('direction')
        confidence = entry_signal.get('confidence', 0.0)
        
        direction = 0
        if direction_str == "BUY":
            direction = 1
        elif direction_str == "SELL":
            direction = -1
            
        score = float(confidence)
        
        # 3. Enhance Score with Structural Context if no direct signal
        # If no entry signal (score 0), we can still provide directional bias
        if score < 20:
             trend = smc_result.get('trend', 'RANGING')
             if trend == "BULLISH":
                 score = 30.0
                 direction = 1
             elif trend == "BEARISH":
                 score = 30.0
                 direction = -1

        # 4. Filter Invalid FVG thresholds (Safety sanity check)
        # SMCAnalyzer handles pips dynamically, so this is just a passthrough.

        raw_output = {
            "score": score,
            "direction": direction,
            "smc_summary": {
                "active_obs": len(smc_result.get('active_order_blocks', [])),
                "active_fvgs": len(smc_result.get('active_fvgs', [])),
                "pools": len(smc_result.get('liquidity_pools', [])),
                "trend": smc_result.get('trend', 'RANGING')
            }
        }
        
        # Log Bridge
        if score > 50:
            logger.info(f"Sniper Signal: {direction_str} (Score {score:.1f}) | {entry_signal.get('reason')}")

        return self._wrap_with_thought(df, current_price, raw_output)

    def _wrap_with_thought(self, df, current_price: float, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        market_state: Dict[str, Any] = {
            "price": float(current_price),
            "smc_structure": raw_output.get("smc_summary", {})
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
