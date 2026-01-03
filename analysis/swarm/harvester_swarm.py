
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
        
        
        # --- 3. TACHYON SHIELD (Phase 78: Priority Profit Taking) ---
        # Logic: If we hit critical targets, EXIT IMMEDIATELY.
        # Do not wait for "Market Struggle". Do not analyze ER. Secure the bag.
        
        positions = tick.get('positions', 0)
        current_profit = tick.get('profit', 0.0) 
        best_profit = tick.get('best_profit', -999.0)
        best_ticket = tick.get('best_ticket', 0)
        
        if positions > 0:
            # A. EMERGENCY CIRCUIT BREAKER (Stop Loss)
            if current_profit < -15.0: # Tightened from -20 to -15
                 signal = "EXIT_ALL"
                 confidence = 100.0 # MAX PRIORITY
                 reason = f"🚨 EMERGENCY STOP: Loss ${current_profit:.2f} > Limit $15.00"
                 return SwarmSignal(self.name, signal, confidence, pd.Timestamp.now(), {'reason': reason})

            # B. SURGICAL SNIPER EXIT (Single Trade TP)
            # If any single trade is > $3.00 (Lowered for Scalping), take it.
            if best_profit > 3.0:
                 signal = "EXIT_SPECIFIC"
                 confidence = 100.0
                 reason = f"🎯 SNIPER EXIT: Trade {best_ticket} Green (${best_profit:.2f}) > $3.00 Threshold."
                 return SwarmSignal(self.name, signal, confidence, pd.Timestamp.now(), {'ticket': best_ticket, 'reason': reason})
                 
            # C. VIRTUAL TP (Global TP)
            # If Net PnL > $5.00, take it.
            if current_profit > 5.0:
                 signal = "EXIT_ALL"
                 confidence = 95.0
                 reason = f"💎 VIRTUAL TP HIT: Profit ${current_profit:.2f} > $5.00 Threshold."
                 return SwarmSignal(self.name, signal, confidence, pd.Timestamp.now(), {'reason': reason})

        # --- 4. Efficiency Analysis (Market Quality) ---
        # Used for "Soft Exits" (Stalling) or Vetoing new trades.
        
        if er < 0.3 or wick_ratio > 0.6:
            # Market is Churning/Struggling.
            if positions > 0:
                # Soft Exit: We are Green but stalling.
                if current_profit > 0.5: # Breakeven+
                     signal = "EXIT_ALL"
                     confidence = 90.0
                     reason = f"Profit Snatcher: Green (${current_profit:.2f}) but Stalling (ER {er:.2f})"
                else:
                     # We are Red and Stalling. HEDGE? Or just WAIT?
                     # Panic Close if very choppy?
                     signal = "WAIT" # Let logic handle it, or panic close if risky
            else:
                signal = "VETO"
                confidence = 60.0
                reason = f"Chop Detected (Vetoing Entry): ER {er:.2f} | Wick {wick_ratio:.2f}"
                
        elif er > 0.7:
             # Market is Flying.
             signal = "HOLD"
             confidence = 90.0
             reason = f"Flow State: ER {er:.2f}"
             
        # --- 5. Phase 94: The Pre-Cog Safety (Virtual SL) ---
        # Logic: If Drawdown > Threshold AND Physics Invalidation -> KILL.
        
        # Updated: Trigger at -$10 (Early Warning) and -$20 (Kill Zone) for $40 Max Risk.
        if positions > 0 and current_profit < -10.0: 
            
            # A. Riemann Curvature Check (Simplified)
            closes = df_m1['close']
            if len(closes) > 20:
                sma = closes.rolling(20).mean().iloc[-1]
                ema = closes.ewm(span=20).mean().iloc[-1]
                sep = ema - sma
                
                # If Profit < -20.0 (50% of Virtual SL) AND Market is Efficiently Moving Against Us (ER > 0.6)
                if current_profit < -20.0:
                    if er > 0.6: 
                        # High Efficiency AGAINST us.
                        signal = "EXIT_ALL"
                        confidence = 95.0
                        reason = f"PRE-COG SAFETY: Efficient Trend Interception. Loss ${current_profit:.2f} with ER {er:.2f}."
                        
                    # B. Nash Invalidation (Volume Pressure)
                    last_vol = df_m1['volume'].iloc[-1]
                    avg_vol = df_m1['volume'].mean()
                    
                    if last_vol > avg_vol * 1.5:
                         signal = "EXIT_ALL"
                         confidence = 85.0
                         reason = f"PRE-COG SAFETY: Volume Spike Rejection. Loss ${current_profit:.2f}."

        return SwarmSignal(
            source="HarvesterSwarm",
            signal_type=signal,
            confidence=confidence,
            timestamp=0,
            meta_data={"er": er, "wick_ratio": wick_ratio, "reason": reason}
        )
