

import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.order_flow import OrderFlowEngine

logger = logging.getLogger("CausalGraphSwarm")

class CausalGraphSwarm(SubconsciousUnit):
    """
    Phase 120: The Architect (Causal Inference).
    
    Constructs a Causal DAG (Directed Acyclic Graph) to validate market moves.
    Causal Chain: News (Info) -> OrderFlow (Aggression) -> Price (Result).
    
    If 'Result' happens without 'Cause', it is an Anomaly (Trap).
    """
    def __init__(self):
        super().__init__("Causal_Graph_Swarm")
        self.order_flow = OrderFlowEngine()
        # Causal Edges Probabilities
        self.edges = {
            ('delta', 'price'): 0.5,
            ('volume', 'volatility'): 0.5
        }

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Analyze Cause (Order Flow)
        flow_state = self.order_flow.analyze_flow(df_m5)
        if not flow_state: return None
        
        delta = flow_state['delta'] # Net Aggression
        is_whale = flow_state['is_whale']
        is_absorption = flow_state['is_absorption']
        
        # 2. Analyze Effect (Price)
        price_change = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-1]
        is_bullish_candle = price_change > 0
        
        # 3. Causal Validation (The Why)
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Case A: Valid Move (Cause == Effect)
        # Price UP AND Delta POSITIVE (Aggressive Buying)
        if is_bullish_candle and delta > 0:
            signal = "BUY"
            confidence = 75.0
            if is_whale: 
                confidence += 15.0 # Whale support
                reason = "VALID: Whale Buying supporting Price."
            else:
                reason = "VALID: Delta supports Price."

        # Case B: Valid Move (Cause == Effect)
        # Price DOWN AND Delta NEGATIVE (Aggressive Selling)    
        elif not is_bullish_candle and delta < 0:
            signal = "SELL"
            confidence = 75.0
            if is_whale:
                confidence += 15.0
                reason = "VALID: Whale Selling supporting Price."
            else:
                reason = "VALID: Delta supports Price."
                
        # Case C: Divergence / Absorption (Trap)
        # Price UP but Delta NEGATIVE (Limit Sellers absorbing Market Buys? No, wait)
        # Price UP but Delta NEGATIVE -> Means more SELLING aggression, but Price went UP?
        # This implies Limit BUY orders absorbed the selling and pushed price up (Passive Buyers).
        # OR Stop Hunt (triggered stops = market buys).
        # Commonly: "Effort vs Result" anomaly.
        
        elif is_bullish_candle and delta < 0:
            # Trap? Or Reversal?
            # Rising Price on Selling Volume is suspicious.
            signal = "SELL" # Fade the move?
            confidence = 60.0
            reason = "ANOMALY: Price Rising on Selling Delta (Divergence)"
            
        elif not is_bullish_candle and delta > 0:
            # Price Dropping on Buying Volume.
            signal = "BUY" # Fade the drop
            confidence = 60.0
            reason = "ANOMALY: Price Dropping on Buying Delta (Divergence)"
            
        # Case D: Absorption (Doji High Vol)
        if is_absorption:
            # High Vol, No Move. Reversal likely.
            # If recent trend was UP, Sell.
            # Simple heuristic:
            signal = "WAIT" # Volatile, let's wait for the break.
            confidence = 0.0
            reason = "ABSORPTION: High Effort, No Result."

        if signal != "WAIT":
             return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'reason': reason, 'delta': delta, 'whale': is_whale}
            )
            
        return None

