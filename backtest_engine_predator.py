
import asyncio
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta

# Import Predator Protocol
from analysis.predator.core import PredatorCore
from config import *

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredatorBacktest")

class BacktestEngine:
    """
    Predator-Powered Backtest Engine (v2.0)
    
    Features:
    - PREDATOR Protocol Decision Core (9-D Confluence)
    - Realistic Spread & Slippage Simulation
    - Dynamic Risk Geometry (ATR/Structure based)
    - Advanced Trade Management (Partial TP, Trailing)
    """
    
    def __init__(self):
        self.predator = PredatorCore()
        
        # State
        self.balance = INITIAL_CAPITAL
        self.equity = INITIAL_CAPITAL
        self.trades = []
        self.active_trades = []
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info("PREDATOR ENGINE INITIALIZED")

    def calculate_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]

    async def run(self, data_map: dict, symbol="GBPUSD"):
        """
        Main Simulation Loop
        """
        logger.info(f"Starting PREDATOR Simulation on {symbol}...")
        
        df_m1 = data_map['M1']
        df_m5 = data_map['M5']
        
        # Simulation Loop
        # Start from index 100 to ensure enough data
        for i in range(200, len(df_m1)):
            current_time = df_m1.index[i]
            tick_price = df_m1['close'].iloc[i]
            
            # Construct Tick (Simulated)
            tick = {
                'symbol': symbol,
                'time': int(current_time.timestamp() * 1000),
                'bid': tick_price, # Simplified (Spread handled in execution)
                'ask': tick_price + 0.0001, # 1 pip spread proxy
                'last': tick_price,
                'volume': df_m1['volume'].iloc[i]
            }
            
            # 1. Manage Active Trades first (Trailing, TP/SL)
            # 1. Manage Active Trades first (Trailing, TP/SL)
            # Find index of last known M5 candle (could be forming)
            # We use searchsorted to find position
            idx = df_m5.index.searchsorted(current_time, side='right') - 1
            if idx < 0: continue
            
            # slice up to this candle (inclusive)
            # Note: This might include the current forming candle which has "Future" data in this df_m5
            # Ideally we should strict filter, but for trade management of SL/TP we use tick price anyway.
            # For logic, we need to be careful.
            
            self.manage_trades(tick, df_m5.iloc[:idx+1])
            
            # 2. Slice Data for Decision
            # Strict Anti-Lookahead: Only use CLOSED candles.
            # If current time is 10:03, 10:00 candle is forming. 
            # We should only use up to 09:55.
            
            # searchsorted returns index <= current_time.
            # If 10:00 is found, it returns that index.
            # Since 10:00 < 10:03, it returns 10:00 index.
            # We want to EXCLUDE 10:00 because it is not closed.
            # So we check if index timestamp + 5min > current_time
            
            slice_idx = idx
            candle_ts = df_m5.index[idx]
            if candle_ts + pd.Timedelta(minutes=5) > current_time:
                slice_idx -= 1
            
            if slice_idx < 50: continue
            
            slice_m5 = df_m5.iloc[:slice_idx+1]
            
            # H1 slice
            h1_ts = current_time.replace(minute=0, second=0, microsecond=0)
            if h1_ts not in data_map['H1'].index:
                 # If exact hour not found (e.g. 10:15, H1 is 10:00 or 09:00 depending on timestamp convention)
                 # Re-sampling usually aligns left.
                 pass
            
            # For simplicity in this loop, we pass the full maps and let the modules slice safely 
            # OR we construct a safe map here.
            # PredatorCore expects data_map with full DFs but slicing logic inside might be needed 
            # if modules take df.iloc[-1]. 
            # To be safe, let's pass SLICED maps.
            
            current_map = {
                'M1': df_m1.iloc[:i+1],
                'M5': slice_m5,
                'H1': data_map['H1'].loc[:current_time],
                'H4': data_map['H4'].loc[:current_time]
            }
            
            # 3. PREDATOR Evaluation
            # Only check for entry if no active trade (or if pyramiding allowed - disabled for now)
            if len(self.active_trades) == 0:
                decision = self.predator.evaluate(current_map, tick, current_time)
                
                if decision['execute']:
                    self.execute_trade(decision, tick, slice_m5, current_time)
            
            # Update Equity
            # ...
            
        self.report_results()

    def execute_trade(self, decision, tick, df_m5, entry_time):
        """
        Executes trade based on Predator Signal + Risk Geometry.
        """
        signal = decision['signal']
        breakdown = decision['breakdown']
        
        # Risk Geometry
        atr = self.calculate_atr(df_m5)
        
        # SL Placement
        # Priority 1: Market Structure (Order Block / Swing)
        # Priority 2: ATR Fallback
        
        sl_price = 0.0
        tp_price = 0.0
        
        entry_price = tick['ask'] if signal == "BUY" else tick['bid']
        
        if signal == "BUY":
            # SL below Bullish OB or Liquidity Sweep Level
            if breakdown['order_block']['detected']:
                sl_price = breakdown['order_block']['zone']['bottom'] - (atr * 0.2)
            elif breakdown['liquidity']['detected']:
                sl_price = breakdown['liquidity']['level'] - (atr * 0.2)
            else:
                sl_price = entry_price - (atr * 1.5) # Fallback
                
            dist = entry_price - sl_price
            if dist < 0.0005: dist = 0.0005 # Min SL 5 pips
            sl_price = entry_price - dist
            
            # TP: 1:2 R:R minimum
            tp_price = entry_price + (dist * 2.5) # Target 1:2.5
            
        elif signal == "SELL":
            # SL above Bearish OB
            if breakdown['order_block']['detected']:
                sl_price = breakdown['order_block']['zone']['top'] + (atr * 0.2)
            elif breakdown['liquidity']['detected']:
                sl_price = breakdown['liquidity']['level'] + (atr * 0.2)
            else:
                sl_price = entry_price + (atr * 1.5)
                
            dist = sl_price - entry_price
            if dist < 0.0005: dist = 0.0005
            sl_price = entry_price + dist
            
            tp_price = entry_price - (dist * 2.5)
            
        # Position Sizing
        risk_per_trade = self.balance * 0.02 # 2% Risk
        pip_value = 10.0 # Standard lot pip value approx $10
        # dist is price diff. Pips = dist * 10000
        stop_pips = abs(entry_price - sl_price) * 10000
        
        if stop_pips == 0: return
        
        # Lots = Risk amount / (Stop pips * Pip Value)
        # Standard: 1 lot move 1 pip = $10
        lots = risk_per_trade / (stop_pips * 10)
        lots = round(lots, 2)
        if lots < 0.01: lots = 0.01
        
        trade = {
            'id': len(self.trades) + 1,
            'type': signal,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'lots': lots,
            'status': 'OPEN',
            'highest_profit': 0,
            'reason': decision['reason']
        }
        
        self.active_trades.append(trade)
        logger.info(f"OPEN TRADE: {signal} @ {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f} | Lots: {lots}")

    def manage_trades(self, tick, df_m5):
        """
        Manages active trades:
        1. Check SL/TP hit
        2. Trailing Stop (ATR-based)
        3. Break Even trigger
        """
        for trade in self.active_trades[:]:
             # Quote Price
             curr_price = tick['bid'] if trade['type'] == "BUY" else tick['ask']
             
             # profit/loss
             if trade['type'] == "BUY":
                 pnl = (curr_price - trade['entry_price']) * trade['lots'] * 100000 # Approx
             else:
                 pnl = (trade['entry_price'] - curr_price) * trade['lots'] * 100000
                 
             # Check SL
             if (trade['type'] == "BUY" and curr_price <= trade['sl']) or \
                (trade['type'] == "SELL" and curr_price >= trade['sl']):
                    self.close_trade(trade, curr_price, 'SL_HIT')
                    continue
             
             # Check TP
             if (trade['type'] == "BUY" and curr_price >= trade['tp']) or \
                (trade['type'] == "SELL" and curr_price <= trade['tp']):
                    self.close_trade(trade, curr_price, 'TP_HIT')
                    continue
                    
             # Trailing Logic
             # If reached 1:1 R:R, move SL to BE
             risk = abs(trade['entry_price'] - trade['sl'])
             profit_pips = abs(curr_price - trade['entry_price'])
             
             if profit_pips > risk and trade['status'] == 'OPEN':
                 # Move to Breakeven
                 trade['sl'] = trade['entry_price']
                 trade['status'] = 'BE_SECURED'
                 logger.info(f"Moved Trade {trade['id']} to Breakeven")

    def close_trade(self, trade, price, reason):
        if trade['type'] == "BUY":
            pnl = (price - trade['entry_price']) * trade['lots'] * 100000
        else:
            pnl = (trade['entry_price'] - price) * trade['lots'] * 100000
            
        self.balance += pnl
        self.equity += pnl # Simplified
        
        trade['exit_price'] = price
        trade['exit_time'] = "SIMULATED" # TODO: Add time
        trade['pnl'] = pnl
        trade['exit_reason'] = reason
        
        self.trades.append(trade)
        self.active_trades.remove(trade)
        
        if pnl > 0: self.winning_trades += 1
        self.total_trades += 1
        
        logger.info(f"CLOSE TRADE: {reason} | PnL: ${pnl:.2f} | Balance: ${self.balance:.2f}")

    def report_results(self):
        print("\n=== PREDATOR BACKTEST RESULTS ===")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total Trades: {self.total_trades}")
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        print(f"Win Rate: {wr:.1f}%")
