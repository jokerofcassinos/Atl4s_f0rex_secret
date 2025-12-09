import pandas as pd
import logging
import numpy as np

logger = logging.getLogger("StrategyCore")

class StrategyCore:
    def __init__(self):
        self.risk_per_trade = 0.02 # 2% of capital (Standard) but for $30 we might need fixed lots.
        self.min_lot = 0.01

    def decide(self, analysis_result_tuple, account_info, open_positions):
        """
        Decides on the next action.
        analysis_result_tuple: (df, prediction_probs)
        """
        df, predictions = analysis_result_tuple
        if df.empty:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = None
        
        # --- Logic Definitions ---
        
        # 1. Trend Filter (EMA / Kalman)
        trend_bullish = current['close'] > current['EMA_50']
        trend_bearish = current['close'] < current['EMA_50']

        # 2. Reversal / Overextension (Z-Score & Bollinger)
        overbought = current['Z_Score'] > 2.0 or current['close'] > current['BB_Upper']
        oversold = current['Z_Score'] < -2.0 or current['close'] < current['BB_Lower']

        # 3. Entry Triggers (FVG / RSI)
        # Pullback into FVG in direction of trend?
        # Detecting if we are inside an FVG zone is complex with just the current row "FVG" tag. 
        # The tag I created marks the formation of FVG.
        # Simple Logic: RSI Cross or Reversal.
        
        rsi_buy = current['RSI'] < 30
        rsi_sell = current['RSI'] > 70
        
        # 4. Probabilistic Confirmation
        prob_bullish = 0.5
        if predictions:
            # predictions is dict like {'WeakBull': 0.6, 'StrongBear': 0.1, ...}
            # Sum up Bullish probs
            prob_bullish = sum([p for k, p in predictions.items() if 'Bull' in k])
        
        # --- Decision Matrix ---
        
        # STRATEGY A: Mean Reversion (Z-Score Extremes)
        if undersold and prob_bullish > 0.6:
            # Strong signal to Buy
            signal = "BUY"
            reason = "MeanReversion: Oversold Z-Score + High Prob"
            
        elif overbought and prob_bullish < 0.4:
            signal = "SELL"
            reason = "MeanReversion: Overbought Z-Score + Low Bull Prob"
            
        # STRATEGY B: Trend Following (Dip Buy)
        elif trend_bullish and rsi_buy: # Simple dip
             signal = "BUY"
             reason = "TrendFollow: Dip in Uptrend"
             
        elif trend_bearish and rsi_sell:
             signal = "SELL"
             reason = "TrendFollow: Rally in Downtrend"

        # Check if we already have position
        if open_positions:
            # Manage existing (Close logic handled by SL/TP usually, but we can dynamic exit)
            pass

        if signal:
            # Risk Calc
            sl, tp = self.calculate_sl_tp(current, signal)
            lot_size = self.calculate_lot_size(account_info['equity'], sl_pips=abs(current['close']-sl))
            
            return {
                "action": signal,
                "lot": lot_size,
                "sl": sl,
                "tp": tp,
                "reason": reason
            }
            
        return {"action": "HOLD"}

    def calculate_sl_tp(self, row, signal):
        """Calculates SL/TP based on ATR."""
        atr = row['ATR'] if not pd.isna(row['ATR']) else 1.0
        # XAUUSD 5m: ATR might be around 0.5 - 2.0 dollars.
        
        # 1.5 ATR Stop, 3 ATR Target (1:2 R:R)
        sl_dist = 1.5 * atr
        tp_dist = 3.0 * atr
        
        price = row['close']
        
        if signal == 'BUY':
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
            
        return sl, tp

    def calculate_lot_size(self, equity, sl_pips):
        # Risk 2%
        risk_amt = equity * self.risk_per_trade
        # Value per pip for 1 lot XAUUSD is usually $1 (Contract size 100? No, standard is $1 per 0.1 move? Need verify)
        # MT5 XAUUSD: Contract size 100 oz. 1 lot = 100 oz. $1 move = $100 P/L.
        # $30 capital -> 2% is $0.60. 
        # If SL is $2 (price move), then 1 lot loses $200. 
        # 0.01 lot loses $2. 
        # So sticking to 0.01 is already highly leveraged (>5% risk).
        # We must use 0.01 fixed for $30 account.
        
        return 0.01
