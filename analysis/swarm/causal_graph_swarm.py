

import logging
import numpy as np
import pandas as pd
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.order_flow import OrderFlowEngine
from core.agi.big_beluga.snr_matrix import SNRMatrix

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
        self.snr_matrix = SNRMatrix()
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
        
        # 2. Analyze Structure (The "Ontological Context")
        # Is there a Wall in front of us?
        current_price = df_m5['close'].iloc[-1]
        self.snr_matrix.scan_structure(df_m5)
        # Assuming M5 ATR is approx 0.0010 (10 points) or calculated:
        high_low = df_m5['high'] - df_m5['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr == 0: atr = 0.0010
        
        levels = self.snr_matrix.get_nearest_levels(current_price)
        dist_res = levels.get('res_dist', 99.0)
        dist_sup = levels.get('sup_dist', 99.0)
        
        # STRUCTURAL CAUSALITY (The Wall Principle)
        # Cause: Momentum -> Effect: Price Move?
        # Check: Does Obstacle block Effect?
        # If Dist to Wall < 0.5 * ATR (Very close), we need MASSIVE energy to break it.
        wall_proximity = 0.5 * atr
        hit_resistance = dist_res < wall_proximity
        hit_support = dist_sup < wall_proximity
        
        # 3. Analyze Effect (Price)
        price_change = df_m5['close'].iloc[-1] - df_m5['open'].iloc[-1]
        is_bullish_candle = price_change > 0
        
        # 3. Causal Validation (The Why)
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Case A: Valid Move (Cause == Effect)
        # Price UP AND Delta POSITIVE (Aggressive Buying)
        if is_bullish_candle and delta > 0:
            if hit_resistance:
                 # We are hitting a wall.
                 if is_whale:
                     # Whales Break Walls.
                     signal = "BUY"
                     confidence = 85.0
                     reason = "BREAKOUT: Whale Smashing Resistance."
                 else:
                     # "Structural Inversion": The Wall will hold. Retail is trapped.
                     # Instead of waiting, we FADE the move.
                     signal = "SELL" 
                     confidence = 70.0 # Good probability rejection
                     reason = "REJECTION: Buying into Wall without Whale Support (Fade)."
            else:
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
            if hit_support:
                if is_whale:
                    signal = "SELL"
                    confidence = 85.0
                    reason = "BREAKOUT: Whale Smashing Support."
                else:
                    # Structural Inversion: Support will hold.
                    signal = "BUY"
                    confidence = 70.0
                    reason = "BOUNCE: Selling into Support without Whale Support (Fade)."
            else:
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
            if hit_resistance:
                signal = "SELL"
                confidence = 80.0
                reason = "TRAP: Rising into Resistance on Selling Delta (Fakeout)."
            else:
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

