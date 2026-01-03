
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("HarvesterSwarm")

class HarvesterSwarm(SubconsciousUnit):
    """
    The Harvester (Dynamic Exit Logic).
    Logic:
    - Analyzes 'Ease of Movement' (Efficiency Ratio).
    - If Market is struggling (High Volatility / Low Displacement) -> Signal EXIT.
    - If Market is flowing (Low Volatility / High Displacement) -> STAY.
    - Only affects existing positions (Managed by Orchestrator context).
    """
    def __init__(self):
        super().__init__("HarvesterSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        tick = context.get('tick')
        
        if df_m1 is None or len(df_m1) < 10: return None
        
        # 1. Calculate Efficiency Ratio (ER)
        # ER = Change / Volatility
        # Change = Abs(Close - Open)
        # Volatility = Sum(Abs(H-L))
        
        period = 5
        # Debug Data Quality
        logger.info(f"HARVESTER DATA CHECK: \n{df_m1.tail(3)[['open','high','low','close']]}")

        # Use -2 (Last Closed Candle) to avoid forming candle noise
        idx = -2
        if len(df_m1) < 10: return None
        
        last = df_m1.iloc[idx]
        prev_n = df_m1.iloc[idx - period]
        
        # Log Timestamp to check freshness
        logger.info(f"HARVESTER Analyzing Candle Time: {last.name} (Now: {pd.Timestamp.now()})")
        
        change = abs(last['close'] - prev_n['close'])
        
        # Recalculate Volatility manually for the window
        # We need the sum of ranges for the window ending at idx
        # Slice from [idx - period + 1] to [idx] (inclusive)
        window = df_m1.iloc[idx - period + 1 : idx + 1]
        volatility = (window['high'] - window['low']).sum()
        
        if volatility == 0:
            # Flat Candle / Bad Data
            # logger.warning("Harvester skipping flat candle (Vol=0)")
            return None
        
        er = change / volatility # Volatility is > 0
        
        body = abs(last['close'] - last['open'])
        rng = last['high'] - last['low']
        
        wick_ratio = (rng - body) / rng if rng > 0 else 0
        
        # logger.info(f"HARVESTER CALC: Change={change:.2f} Vol={volatility:.2f} ER={er:.2f} Wick={wick_ratio:.2f}")
        
        # 3. Decision Logic
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # If Price Action is "Difficult" (Low ER, High Wicks) -> SUGGEST EXIT
        # "Quanto mais dificil mais rapido fecha"
        
        if er < 0.3 or wick_ratio > 0.6:
            # Market is Churning/Struggling.
            # Only signal EXIT if we actually have positions to protect.
            # Otherwise, just VETO new entries.
            
            positions = tick.get('positions', 0)
            current_profit = tick.get('profit', 0.0) # New Field from Bridge
            
            if positions > 0:
                # PROFIT SNATCHER LOGIC
                if current_profit > 0 and er < 0.4:
                     signal = "EXIT_ALL"
                     confidence = 90.0
                     reason = f"Profit Snatcher: Green (${current_profit:.2f}) but Stalling (ER {er:.2f})"
                
                elif current_profit > 15.0: # Hard Target (Scalp Bag check)
                     signal = "EXIT_ALL"
                     confidence = 95.0
                     reason = f"Hard Target Reached: ${current_profit:.2f}"
                     
                else:
                     signal = "EXIT_ALL" # Acts as 'Close if Open' (Panic defense)
                     confidence = 85.0 # High urgency
                     reason = f"Struggle Detected (Protecting Capital): ER {er:.2f} | Wick {wick_ratio:.2f}"
            else:
                signal = "VETO" # Prevent entering this mess
                confidence = 60.0
                reason = f"Chop Detected (Vetoing Entry): ER {er:.2f} | Wick {wick_ratio:.2f}"
            
        elif er > 0.7:
             # Market is Flying.
             signal = "HOLD" # Encourage holding
             confidence = 90.0
             reason = f"Flow State: ER {er:.2f}"
             
        return SwarmSignal(
            source="HarvesterSwarm",
            signal_type=signal,
            confidence=confidence,
            timestamp=0,
            meta_data={"er": er, "wick_ratio": wick_ratio, "reason": reason}
        )
