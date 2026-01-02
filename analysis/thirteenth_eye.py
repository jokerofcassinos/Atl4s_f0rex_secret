import logging
import pandas as pd
import numpy as np
import config
from src.macro_math import MacroMath # Re-using helpful math tools if needed

logger = logging.getLogger("Atl4s-ThirteenthEye")

class ThirteenthEye:
    """
    The Time-Knife (Quantum Grid).
    Analyzes M1 Micro-Structure for violent reversals.
    Executes a 'Cluster' of trades (Grid) within the same candle to capture volatility.
    """
    def __init__(self):
        self.cooldown_tracker = 0
        self.last_signal_time = 0

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return None
        
        close = df['close']
        
        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (2.5 SD for Extreme Reversal)
        df['ma20'] = close.rolling(20).mean()
        df['std20'] = close.rolling(20).std()
        df['upper'] = df['ma20'] + (df['std20'] * 2.5)
        df['lower'] = df['ma20'] - (df['std20'] * 2.5)
        
        return df

    def determine_grid_size(self, capital, atr=1.0):
        """
        Dynamic Slot Allocation & Elastic Spacing.
        Returns: num_slots, volume, spacing
        """
        # Spacing scales with Volatility (ATR)
        # 1.0 ATR (M1) is usually ~10-20 points ($0.10-$0.20) on Gold? No, points are $0.01 ticks. 
        # On Gold, 1 pip = $0.10. M1 ATR might be $0.50 - $1.00.
        # Let's say ATR is raw price diff.
        
        base_spacing = 15 # default points
        elastic_spacing = max(10, atr * 10.0) # Scale: ATR * 10 (e.g. ATR $0.50 -> 50 points -> $0.50)
        # Adjust spacing logic: points = price * 100? No, standard points.
        # Let's assume ATR is passed in Price Units (e.g. 1.20). 
        # 1.20 move is substantial. 
        
        grid_spacing = int(elastic_spacing)
        
        if capital < 50: return 2, 0.01, grid_spacing
        if capital < 200: return 3, 0.01, grid_spacing
        if capital < 1000: return 5, 0.02, grid_spacing
        return 8, 0.05, grid_spacing

    def scan_for_reversal(self, df_m1, current_capital, current_time):
        """
        Scans M1 for:
        1. Exhaustion (Reversal)
        2. Boom (Momentum/Expansion)
        Returns dict with action/slots.
        """
        # Cooldown (1 Minute per cluster) - Reduced to 30s for Boom
        if current_time - self.last_signal_time < 30:
            return None
            
        df = self.calculate_indicators(df_m1)
        if df is None: return None
        
        # Calculate EMA 9, 20 for Flow
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

        
        # Calculate ATR/Body size
        df['body_size'] = (df['close'] - df['open']).abs()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = None
        reason = ""
        
        avg_body = df['body_size'].rolling(10).mean().iloc[-1]
        current_body = last['body_size']
        
        # --- MACHINE GUN MODE (Hyper-Swarm) ---
        # If in strong M1 Trend, fire rapid scalps
        ema9 = last['ema9']
        ema20 = last['ema20']
        close = last['close']
        
        # Bullish Flow
        if close > ema9 and ema9 > ema20:
             # Check for pullbacks/continuations
             # Fire if we are not overextended (Body not huge yet)
             if current_body < (avg_body * 1.5): # Steady climb
                  signal = "BUY"
                  reason = "HYPER-SWARM: M1 Laminar Flow (Machine Gun) - Bullish"
                  # Special ID for Machine Gun to allow faster cooldown
                  self.last_signal_time = current_time - 20 # Trick cooldown to 10s (30-20)
        
        # Bearish Flow
        elif close < ema9 and ema9 < ema20:
             if current_body < (avg_body * 1.5):
                  signal = "SELL"
                  reason = "HYPER-SWARM: M1 Laminar Flow (Machine Gun) - Bearish"
                  self.last_signal_time = current_time - 20

        is_boom = current_body > (avg_body * 2.0)
        
        # BOOM OVERRIDE (Takes precedence if valid)
        if is_boom:
            if last['close'] > last['open']:
                # Bullish Boom
                signal = "BUY"
                reason = f"M1 BOOM (UP): Body {current_body:.2f} > 2.0x Avg"
            else:
                # Bearish Boom
                signal = "SELL"
                reason = f"M1 BOOM (DOWN): Body {current_body:.2f} > 2.0x Avg"
                
        # --- REVERSAL DETECTION (Mean Reversion) ---
        # Only check if NO Boom (Boom takes priority)
        if not signal:
            # 1. Bearish Reversal (Top)
            if last['close'] > last['upper'] or prev['close'] > prev['upper']:
                if last['rsi'] > 85: # Stricter
                    signal = "SELL"
                    reason = f"M1 Exhaustion (RSI {last['rsi']:.1f} + BB Break)"
                    
            # 2. Bullish Reversal (Bottom)
            elif last['close'] < last['lower'] or prev['close'] < prev['lower']:
                if last['rsi'] < 15: # Stricter
                    signal = "BUY"
                    reason = f"M1 Exhaustion (RSI {last['rsi']:.1f} + BB Break)"
                
        if signal:
            self.last_signal_time = current_time
            
            # Calculate Quick ATR (Last 10 candles range average)
            # Assuming 'range' or 'high'-'low' logic
            high_low = df['high'] - df['low']
            atr_val = high_low.rolling(10).mean().iloc[-1]
            if np.isnan(atr_val): atr_val = 1.0
            
            num_slots, vol_slot, spacing = self.determine_grid_size(current_capital, atr=atr_val)
            
            # Boom Logic: Aggressive entry (Market)
            # Reversal Logic: Grid (Market + Limits)
            # For simplicity, we use the same Grid structure but the Main Loop handles it.
            
            slots = []
            slots.append({"type": "MARKET", "vol": vol_slot, "offset": 0})
            
            # Additional Layers
            for i in range(1, num_slots):
                offset = i * spacing # Elastic Spacing
                slots.append({"type": "LIMIT", "vol": vol_slot, "offset": offset})
                
            logger.info(f"QUANTUM GRID TRIGGER: {signal} | {reason} | Layers: {num_slots} | Spacing: {spacing}")
            
            return {
                "action": signal,
                "slots": slots,
                "reason": reason
            }
            
        return None
