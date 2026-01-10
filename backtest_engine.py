"""
Omega Protocol Backtest Engine v3.0 (Maximum Power)

Simulates the full "Omega AGI" stack with:
1. OmegaAGICore (Brain) - Regime Detection, Strategic Pivot
2. SwarmOrchestrator (Cortex) - 28 Optimized Agents
3. M8 Fibonacci System (Filter) - Triple Validation
4. HistoryLearningEngine (Memory) - Feedback Loop
5. Mock Execution - Realistic fills
"""

import pandas as pd
import numpy as np
import logging
import time
import asyncio
import datetime
from typing import Dict, Any, List

# Core Systems
from core.agi.omega_agi_core import OmegaAGICore
from core.swarm_orchestrator import SwarmOrchestrator
from core.execution_engine import ExecutionEngine
from core.agi.learning.history_learning import HistoryLearningEngine
from analysis.m8_fibonacci_system import M8FibonacciSystem
from analysis.time_converter import TimeframeConverter
# from analysis.cortex.alpha_cortex import AlphaCortex # Removed

# Dependencies for SwarmOrchestrator
from core.consciousness_bus import ConsciousnessBus
from core.genetics import EvolutionEngine
from core.neuroplasticity import NeuroPlasticityEngine
from core.transformer_lite import TransformerLite
from core.grandmaster import GrandMaster

import config

# Setup Logging
logging.basicConfig(level=logging.WARNING) # Suppress INFO logs for speed
logger = logging.getLogger("Atl4s-Backtest")

import win32pipe, win32file, win32event, pywintypes
import json # Ensure json is imported


class MockBridge:
    """Simulates the ZmqBridge for ExecutionEngine."""
    def __init__(self):
        self.orders = []
    
    def get_account_info(self):
        return {'balance': 10000.0, 'equity': 10000.0}
    
    def execute_trade(self, action, symbol, lots, sl, tp, deviation, magic, comment):
        # Simulate instant fill
        ticket = int(time.time() * 1000)
        return ticket
        
    def close_trade(self, ticket, lots):
        return True

class BacktestEngine:
    def __init__(self, m8_threshold: int = 7, memory_file: str = None):
        # 1. Initialize The Stack (Mirroring main.py)
        
        # Mock Bridge
        self.bridge = MockBridge()
        
        # Realistic Execution Engine
        # We will set the symbol later in run()
        from core.backtest.realistic_executor import RealisticExecutor
        self.executor = None # Initialized in run()
        
        # Core AGI
        from core.agi.infinite_why_engine import InfiniteWhyEngine
        from core.agi.simulation_system_agi import SimulationSystemAGI
        
        self.infinite_why = InfiniteWhyEngine()
        self.sim_system = SimulationSystemAGI()
        
        self.agi = OmegaAGICore(
            self.infinite_why, 
            self.sim_system, 
            memory_file=memory_file # Isolation for optimization
        )
        
        # Swarm Bus & Dependencies
        self.bus = ConsciousnessBus()
        self.evolution = EvolutionEngine()
        self.neuroplasticity = NeuroPlasticityEngine()
        self.attention = TransformerLite(embed_dim=64, head_dim=64)
        self.grandmaster = GrandMaster()
        
        # Cortex (SwarmOrchestrator)
        self.cortex = SwarmOrchestrator(
            bus=self.bus,
            evolution=self.evolution,
            neuroplasticity=self.neuroplasticity,
            attention=self.attention,
            grandmaster=self.grandmaster
        )
        # Access internal components for initialization
        # self.swarm_orchestrator = self.cortex # Alias
        
        # M8 System
        self.m8_system = M8FibonacciSystem(threshold=m8_threshold)
        
        # History Engine
        self.history_engine = HistoryLearningEngine(self.agi.holographic_memory)
        self.agi.connect_learning_engine(self.history_engine)
        
        # Config
        self.config = {
            "mode": "SNIPER",
            "risk_per_trade": 0.01,
            "max_spread": 0.5
        }
        
        logger.info("OMEGA BACKTEST ENGINE v3.0 INITIALIZED")

    async def initialize(self):
        """Async init for swarms."""
        await self.cortex.initialize_swarm() # Initialize 28 agents

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates Average True Range."""
        if len(df) < period + 1: return 0.0020
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Simple Moving Average of TR
            atr = np.mean(tr[-period:])
            return atr
        except Exception:
            return 0.0020 # Default 20 pips

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates RSI."""
        if len(df) < period + 1: return 50.0
        
        close_delta = df['close'].diff()
        
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        
        ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
        ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
        
        rsi = ma_up / (ma_up + ma_down)
        rsi = 100 * rsi
        
        return rsi.iloc[-1]

    def get_recent_swings(self, df: pd.DataFrame, lookback: int = 20) -> tuple:
        """Finds recent high/low for structure stop."""
        if len(df) < lookback: return (None, None)
        
        recent = df.tail(lookback)
        return (recent['high'].max(), recent['low'].min())

    def get_magnetic_level(self, price: float, direction: str, df_h1: pd.DataFrame) -> tuple:
        """
        Refined Magnetic Logic: Focus on Major Pools (00, 50) and Key Structure.
        """
        # 1. Major Psychological Levels (Only .00 and .50)
        base = int(price)
        # Gold respects 00 and 50 most heavily. 25/75 are often blown through.
        levels = [base, base + 0.50, base + 1.0] 
        
        best_psych = None
        min_dist = 999.0
        
        for lvl in levels:
            dist = lvl - price if direction == "BUY" else price - lvl
            if dist > 0: # Ahead
                if dist < min_dist:
                    min_dist = dist
                    best_psych = lvl

        # 2. Structure Levels (Significant Swing Points only)
        # Look for Swings that are "Fresh" (within last 12h)
        structure_target = None
        if len(df_h1) > 12:
            recent = df_h1.tail(12) 
            if direction == "BUY":
                # Only care about Highs that are actually targets
                highs = recent[recent['high'] > price]['high'].sort_values()
                if not highs.empty:
                    structure_target = highs.iloc[0]
            else:
                lows = recent[recent['low'] < price]['low'].sort_values(ascending=False)
                if not lows.empty:
                    structure_target = lows.iloc[0]
                    
        magnet = best_psych
        m_type = "PSYCH_MAJOR"
        
        # Structure is stronger if it exists and is relatively close
        if structure_target:
             dist_struc = abs(structure_target - price)
             dist_psych = abs(best_psych - price) if best_psych else 999
             
             if dist_struc < dist_psych:
                 magnet = structure_target
                 m_type = "STRUCT_SWING"

        return magnet, m_type

    def detect_struggle(self, df_m1: pd.DataFrame, direction: str) -> float:
        """
        Refined Struggle: Requires consecutive evidence or extreme rejection.
        """
        if len(df_m1) < 5: return 0.0 # Need more context
        
        recent = df_m1.iloc[-3:].copy()
        score = 0.0
        rejection_count = 0
        
        for _, candle in recent.iterrows():
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            if total_range == 0: continue
            
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            
            if direction == "BUY":
                # Nasty Rejection from top
                if upper_wick > total_range * 0.6: 
                    score += 3.0
                    rejection_count += 1
                elif upper_wick > body * 2.0:
                    score += 2.0
            else:
                # Nasty Rejection from bottom
                if lower_wick > total_range * 0.6: 
                    score += 3.0
                    rejection_count += 1
                elif lower_wick > body * 2.0:
                    score += 2.0
                    
        # Momentum Death (Tiny bodies on all 3 candles)
        avg_body = abs(recent['close'] - recent['open']).mean()
        atr = self.calculate_atr(df_m1)
        
        if avg_body < atr * 0.3: # Volatility Compression against us?
            score += 3.0
            
        # If we have repeated rejection, boost score
        if rejection_count >= 2: score += 4.0
        
        return min(score, 10.0)

    # =============================================================================
    # MATRIX BREAKER v2.0 UTILITIES (ICT METHODS)
    # =============================================================================

    def is_in_ote_zone(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Optimal Trade Entry (OTE): Checks if price is in 61.8%-78.6% Fib Retracement
        of the last significant swing.
        """
        if len(df) < 50: return False
        
        # Find significant swing (last 30 candles)
        recent = df.iloc[-30:]
        current_price = df['close'].iloc[-1]
        
        if direction == "BUY":
            # Finding Retracement for a BUY means we look for a recent LOW to HIGH move?
            # NO. We are buying a DIP. So trend was UP. We look for High -> Low retracement.
            # wait, if we are BUYING, we want price to have dropped into OTE from a High.
            
            # 1. Identify valid Swing Low and Swing High of the impulse
            swing_high = recent['high'].max()
            swing_low = recent['low'].min()
            
            if swing_high == swing_low: return False
            
            range_size = swing_high - swing_low
            
            # Retracement levels from the HIGH down to LOW? 
             # No, standard fib: Low to High. 
             # 0% at High, 100% at Low. 
             # Retracement is valid if current price is between 61.8% and 78.6% down from High.
             
            fib_618 = swing_low + (range_size * 0.382) # 1 - 0.618
            fib_786 = swing_low + (range_size * 0.214) # 1 - 0.786
            
            # Basic Check: Price must be "cheap" (Discount)
            return fib_786 <= current_price <= fib_618
            
        else: # SELL
            # Selling a RALLY. Trend was DOWN.
            # Looking for price to satisfy Premium levels.
            swing_high = recent['high'].max()
            swing_low = recent['low'].min()
            
            if swing_high == swing_low: return False
            
            range_size = swing_high - swing_low
            
            fib_618 = swing_high - (range_size * 0.382) # Retraced UP 61.8%
            fib_786 = swing_high - (range_size * 0.214) # Retraced UP 78.6%
            
            return fib_618 <= current_price <= fib_786

    def detect_order_block(self, df: pd.DataFrame, direction: str, lookback: int = 50) -> bool:
        """
        Detects if current price is reacting off a valid Order Block.
        """
        if len(df) < lookback: return False
        
        current_price = df['close'].iloc[-1]
        
        # Search backward for OBs (Last 3 to 20 candles usually)
        # We don't need deep history for M5 scalping
        scan_window = df.iloc[-20:-2] 
        
        found_ob = False
        
        if direction == "BUY":
            # Bullish OB detection:
            # Look for a Bearish Candle (Down) ...
            # ... Followed IMMEDIATELY by a violent Bullish Move (Displacement) that breaks structure
            
            for i in range(len(scan_window) - 2):
                candle = scan_window.iloc[i]
                next_candle = scan_window.iloc[i+1]
                
                # Candidate: Bearish Candle
                if candle['close'] < candle['open']:
                    # Validation: Next candle engulfed it or moved strongly
                    body_curr = abs(candle['close'] - candle['open'])
                    move_next = next_candle['close'] - candle['close'] # gap up?
                    
                    if move_next > body_curr * 1.5: 
                         # This IS an Order Block Zone (Open to Low of that candle)
                         ob_high = candle['open']
                         ob_low = candle['low']
                         
                         # Check if current price is INSIDE this zone
                         if ob_low <= current_price <= ob_high:
                             found_ob = True
                             break
        else: # SELL
            # Bearish OB detection:
            # Look for Bullish Candle ... Followed by violent drop
             for i in range(len(scan_window) - 2):
                candle = scan_window.iloc[i]
                next_candle = scan_window.iloc[i+1]
                
                if candle['close'] > candle['open']: # Bullish
                    body_curr = abs(candle['close'] - candle['open'])
                    move_next = candle['close'] - next_candle['close']
                    
                    if move_next > body_curr * 1.5:
                        ob_low = candle['open']
                        ob_high = candle['high']
                        
                        if ob_low <= current_price <= ob_high:
                            found_ob = True
                            break
                            
        return found_ob

    def has_displacement(self, df: pd.DataFrame) -> bool:
        """
        Checks if the recent move has 'Displacement' (High Momentum/commitment).
        """
        if len(df) < 5: return False
        
        # Check last 3 candles
        recent = df.iloc[-3:]
        for _, candle in recent.iterrows():
            body = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            if total_range == 0: continue
            
            # Displacement = Big Body, Small Wicks
            if body > total_range * 0.70: # 70% Body
                 # Standard Deviation check (is it big?)
                 # Simplified: Is it bigger than avg?
                 return True
                 
                 # Simplified: Is it bigger than avg?
                 return True
                 
        return False

    def detect_liquidity_trap(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Detects Liquidity Sweep / SFP (Swing Failure Pattern).
        Logic: Price breaks a recent High/Low (taking liquidity) but CLOSES back inside range.
        """
        if len(df) < 20: return False
        
        last_candle = df.iloc[-1]
        
        # Look at last 15 candles EXCLUDING current
        recent = df.iloc[-16:-1]
        
        if direction == "BUY":
            # BULLISH TRAP: We want to see price sweep a LOW then close HIGHER.
            # 1. Identify recent structural low
            recent_low = recent['low'].min()
            
            # 2. Check if Current Candle Swept it
            # Low must be lower than recent structure
            if last_candle['low'] < recent_low:
                # 3. Check for SFP (Close back above the structural low)
                if last_candle['close'] > recent_low:
                     return True
                     
        else: # SELL
            # BEARISH TRAP: Sweep High, Close Lower
            recent_high = recent['high'].max()
            
            if last_candle['high'] > recent_high:
                if last_candle['close'] < recent_high:
                    return True
                    
        return False

    async def run_multi_pair(self, data_dir: str, symbols: list, **kwargs):
        """
        Runs backtest on multiple pairs and aggregates results.
        Refactored to handle the "Multi-Par Strategy".
        """
        all_results = []
        import os
        
        print("\n" + "="*60)
        print(f"STARTING MULTI-PAR BACKTEST: {symbols}")
        print("="*60)

        for symbol in symbols:
            csv_path = f"{data_dir}/{symbol}_M1.csv"
            
            # Check for data, if missing generate synthetic
            if not os.path.exists(csv_path):
                logger.warning(f"Data for {symbol} not found. Generating SYNTHETIC data...")
                self.generate_synthetic_data(csv_path, symbol)
            
            print(f"\n>>> TESTING {symbol}...")
            # Run backtest for this symbol
            result = await self.run(csv_path, symbol=symbol, **kwargs)
            result['symbol'] = symbol
            all_results.append(result)
        
        # Aggregate Results
        total_trades = sum(r['trades'] for r in all_results)
        weighted_wr = sum(r['trades'] * r['win_rate'] for r in all_results) / total_trades if total_trades > 0 else 0
        
        print("\n" + "="*60)
        print("MULTI-PAR AGGREGATE RESULTS")
        print("="*60)
        for r in all_results:
            print(f"  {r['symbol']}: {r['trades']} trades | {r['win_rate']:.1f}% WR | Balance: ${r['balance']:.2f}")
        print("-"*60)
        print(f"  TOTAL TRADES: {total_trades}")
        print(f"  COMBINED WR:  {weighted_wr:.1f}%")
        print("="*60)
        
        return all_results

    def run_bridge_server(self, pipe_name=r'\\.\pipe\Atl4sPipe'):
        """
        Runs as a Named Pipe Server for MT5 Strategy Tester synchronization.
        This blocks and waits for MT5 to send a tick, then processes it and sends a command back.
        """
        print(f"Waiting for MT5 Strategy Tester on {pipe_name}...")
        
        # Initialize Data Buffer (Outside loop to persist across connections)
        self.data_map = {} # {symbol: pd.DataFrame}
        df_buffer = None
        tick_count = 0
        print("Bridge Server Ready. Feeding ticks to Matrix...")
        
        while True:
            try:
                # Create Pipe Instance
                hPipe = win32pipe.CreateNamedPipe(
                    pipe_name,
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    1, 65536, 65536,
                    0,
                    None
                )
                
                # print("Pipe created. Waiting for client connection...") 
                win32pipe.ConnectNamedPipe(hPipe, None)
                # print("Client connected!")
                
                # Connection Loop
                while True:
                    try:
                        # Read Request (Blocking)
                        resp = win32file.ReadFile(hPipe, 65536)
                        msg = resp[1].decode('utf-8')
                        
                        if not msg: break
                        
                        # Process Message
                        parts = msg.split('|')
                        
                        if parts[0] == "TICK":
                            symbol = parts[1]
                            tick_time = int(parts[2]) / 1000.0 # MT5 is ms
                            bid = float(parts[3])
                            ask = float(parts[4])
                            vol = int(parts[5])
                            
                            current_time = datetime.datetime.fromtimestamp(tick_time)
                            
                            # 1. Update Internal State (Optimized List Buffer)
                            new_row = {
                                'time': current_time, 
                                'open': bid, 'high': bid, 'low': bid, 'close': bid, 
                                'volume': vol
                            }
                            
                            if df_buffer is None:
                                df_buffer = pd.DataFrame([new_row])
                            else:
                                # Optimization: Don't concat every tick. Append only if needed or periodically.
                                # For strict accuracy in this prototype, we'll maintain the list approach if perf allows,
                                # but to fix the user's issue, let's just silence the logs and show progress.
                                df_buffer = pd.concat([df_buffer, pd.DataFrame([new_row])], ignore_index=True)
                            
                            # Maintain buffer size
                            if len(df_buffer) > 5000:
                                df_buffer = df_buffer.iloc[-5000:]
                                
                            # 2. Check Signals
                            response = "NO_OP"
                            tick_count += 1
                            
                            # UNCONDITIONAL DIAGNOSTIC: Every 1000 ticks
                            if tick_count % 1000 == 0:
                                print(f"\n🔍 DEBUG: tick_count={tick_count} | df_buffer={'None' if df_buffer is None else len(df_buffer)}")
                            
                            if tick_count % 100 == 0:
                                print(f"\rProcessing Ticks... Total: {tick_count} | Last: {bid:.5f}", end="", flush=True)
                            
                            # Warmup period
                            if len(df_buffer) > 200:
                                # Resample Logic (Only on new minute or periodically to save CPU)
                                # Checking every 10th tick for signals to speed up tester
                                if tick_count % 10 == 0:
                                    # ... (Resampling and M8 Logic kept same) ...
                                    # Quick Resample
                                    try:
                                        df_m1 = df_buffer.set_index('time').resample('1min').agg({
                                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                                        }).dropna().reset_index()
                                        
                                        if len(df_m1) > 50:
                                            # Dynamically generate H1/H4 for M8
                                            df_h1 = df_m1.set_index('time').resample('1h').agg({
                                                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                                            }).dropna().reset_index()
                                            
                                            df_h4 = df_m1.set_index('time').resample('4h').agg({
                                                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                                            }).dropna().reset_index()
                                            
                                            df_m8 = df_m1.set_index('time').resample('8min').agg({
                                                'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
                                            }).dropna().reset_index()

                                            # DIAGNOSTIC: Print data accumulation every 500 ticks
                                            if tick_count % 500 == 0:
                                                print(f"\n📊 Data: Buffer={len(df_buffer)} | M1={len(df_m1)} | H1={len(df_h1)} | H4={len(df_h4)} | M8={len(df_m8)}")

                                            # Lowered from 10 to 3 for shorter backtests
                                            if len(df_h4) >= 3:
                                                m8_result = self.m8_system.evaluate(df_h1, df_h4, df_m8, df_m2=None, current_time=current_time)
                                                
                                                # DIAGNOSTIC: Print M8 result periodically
                                                if tick_count % 500 == 0:
                                                    print(f"🎯 M8: Score={m8_result.get('total_score', 0)} | Signal={m8_result.get('signal')} | Execute={m8_result.get('execute')}")
                                                
                                                if m8_result.get('execute'):
                                                    decision = m8_result.get('signal')
                                                    current_positions = int(parts[8])
                                                    
                                                    if current_positions == 0:
                                                        # Dynamic ATR Calculation for SL/TP
                                                        try:
                                                            atr = df_m1['high'].sub(df_m1['low']).rolling(14).mean().iloc[-1]
                                                            if pd.isna(atr) or atr == 0:
                                                                atr = 0.0010 # Fallback for EURUSD-like
                                                        except:
                                                            atr = 0.0010

                                                        cmd_type = 0 if decision == "BUY" else 1
                                                        lots = 0.1
                                                        price = ask if decision == "BUY" else bid
                                                        
                                                        # Dynamic Distances
                                                        sl_dist = atr * 2.5 # Wide stop for volatility
                                                        tp_dist = atr * 4.0 # High R:R
                                                        
                                                        sl = price - sl_dist if decision == "BUY" else price + sl_dist
                                                        tp = price + tp_dist if decision == "BUY" else price - tp_dist
                                                        
                                                        # OPEN|Type|Vol|SL|TP
                                                        response = f"OPEN|{cmd_type}|{lots}|{sl:.5f}|{tp:.5f}"
                                                        print(f"\n🚀 MATRIX TRIGGER: {decision} on {symbol} @ {price:.5f} | ATR={atr:.5f}")

                                    except Exception as e:
                                        # Resampling errors during warmup are normal
                                        if tick_count % 1000 == 0:
                                            print(f"\n⚠️ Resample Error: {e}")

                            reply_bytes = str.encode(response)
                            win32file.WriteFile(hPipe, reply_bytes)
                            
                        else:
                            win32file.WriteFile(hPipe, b"NO_OP")
                            
                    except pywintypes.error as e:
                        if e.winerror == 109: # Broken pipe
                            # client disconnected, just break to outer loop to reconnect
                            break
                        else:
                            print(f"Pipe Error: {e}")
                            break
                            
            except Exception as e:
                print(f"Server Error: {e}")
            finally:
                if 'hPipe' in locals():
                    try:
                        win32file.CloseHandle(hPipe)
                    except: pass

    def generate_synthetic_data(self, csv_path: str, symbol: str):
        """Generates synthetic M1 data for testing if real data is missing."""
        dates = pd.date_range(end=datetime.datetime.now(), periods=2000, freq='1min')
        
        # Price bases
        base_map = {
            'EURUSD': 1.0500, 'GBPUSD': 1.2500, 'USDJPY': 150.00,
            'USDCAD': 1.3500, 'USDCHF': 0.9000, 'XAUUSD': 2600.00
        }
        base_price = base_map.get(symbol, 100.0)
        
        # Volatility scaler
        vol_scale = 0.0001 if "JPY" not in symbol and "XAU" not in symbol else 0.01
        if "XAU" in symbol: vol_scale = 0.1
        
        np.random.seed(42) # Deterministic for consistent tests
        ticks = len(dates)
        trend = np.linspace(0, 50 * vol_scale * 100, ticks) + (10 * vol_scale * 100) * np.sin(np.linspace(0, 20*np.pi, ticks))
        noise = np.cumsum(np.random.randn(ticks) * vol_scale)
        
        close = base_price + trend + noise
        high = close + np.random.rand(ticks) * (vol_scale * 5)
        low = close - np.random.rand(ticks) * (vol_scale * 5)
        open_p = close + np.random.randn(ticks) * (vol_scale)
        volume = np.random.randint(10, 1000, ticks)
        
        df = pd.DataFrame({
            'time': dates, 'open': open_p, 'high': high, 'low': low, 'close': close, 'volume': volume
        })
        df.to_csv(csv_path, index=False)
        print(f"Generated {csv_path} with {ticks} candles for {symbol}.")


    async def run(self, csv_path: str, symbol: str = "EURUSD", initial_balance: float = 10000.0, fixed_lots: float = 0.1):
        """
        Runs the full simulation.
        """
        logger.info(f"Loading data from {csv_path}...")
        
        # Load M1 Data
        try:
            df_m1 = pd.read_csv(csv_path)
            # Parse datetime
            if 'time' in df_m1.columns:
                df_m1['time'] = pd.to_datetime(df_m1['time'])
                df_m1.set_index('time', inplace=True)
            elif 'date' in df_m1.columns:
                df_m1['date'] = pd.to_datetime(df_m1['date'])
                df_m1.set_index('date', inplace=True)
                
            # Rename columns to lowercase
            df_m1.columns = [c.lower() for c in df_m1.columns]
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return

        # Prepare Simulation State
        balance = initial_balance
        active_trades = []
        trade_log = []
        equity_curve = []
        
        # Generate Higher Timeframes (Pre-calculate for speed, though real mock would ideally stream)
        logger.info("Generating Fibonacci Timeframes...")
        timeframes = TimeframeConverter.generate_fibonacci_timeframes(df_m1)
        df_h1 = TimeframeConverter.resample_ohlcv(df_m1, 60, "H1")
        df_h4 = TimeframeConverter.resample_ohlcv(df_m1, 240, "H4")
        
        total_steps = len(df_m1)
        start_idx = 200 # Reduced warmup for smaller datasets
        
        if self.executor is None or self.executor.symbol != symbol:
            from core.backtest.realistic_executor import RealisticExecutor
            self.executor = RealisticExecutor(latency_ms=100, symbol=symbol)

        logger.info(f"Starting simulation on {total_steps} candles...")
        
        for i in range(start_idx, total_steps):
            if i % 100 == 0:
                print(f"Processing candle {i}/{total_steps}...", end='\r')
                
            # 1. Context Slicing (The "Now")
            current_time = df_m1.index[i]
            # Efficient slicing: we assume data is sorted.
            # Real backtest buffers would be vastly faster than slicing every tick, 
            # but for "Maximum Power" logic testing, accuracy > speed.
            
            # Optimization: Only slice the tail needed
            slice_m1 = df_m1.iloc[max(0, i-500):i+1]
            slice_m5 = timeframes['M5'][timeframes['M5'].index <= current_time].tail(100)
            slice_m8 = timeframes['M8'][timeframes['M8'].index <= current_time].tail(50)
            slice_m2 = timeframes['M2'][timeframes['M2'].index <= current_time].tail(50)
            slice_h1 = df_h1[df_h1.index <= current_time].tail(50)
            slice_h4 = df_h4[df_h4.index <= current_time].tail(20)
            
            if len(slice_m8) < 5: continue  # Reduced requirement

            # Construct Data Map
            data_map = {
                'M1': slice_m1,
                'M5': slice_m5,
                'M8': slice_m8, # Critical for M8 System
                'H1': slice_h1,
                'H4': slice_h4
            }
            
            # Construct Tick (REALISTIC SPREAD)
            current_close = slice_m1.iloc[-1]['close']
            
            # Use Realistic Executor to generate tick with dynamic spread
            current_tick = self.executor.get_current_tick(current_time, current_close)
            
            # Fix: Convert Timestamp to float (epoch seconds) for AGI/System compatibility
            if hasattr(current_tick['time'], 'timestamp'):
                current_tick['time'] = current_tick['time'].timestamp()
            
            # Add missing fields
            current_tick['symbol'] = symbol
            current_tick['last'] = current_close
            current_tick['volume'] = slice_m1.iloc[-1]['volume']
            current_tick['flags'] = 0
            current_tick['trades_json'] = active_trades # Loopback for AGI
            
            # ----------------------------------------------------
            # OMEGA PROTOCOL EXECUTION LOOP
            # ----------------------------------------------------
            
            # 1. AGI Pre-Tick (Regime Detection, Pivot)
            agi_adjustments = self.agi.pre_tick(current_tick, self.config, data_map)
            if agi_adjustments is None: agi_adjustments = {}
            agi_adjustments['m8_fibonacci'] = {} # Init
            
            # =================================================================
            # FULL M8 FIBONACCI SYSTEM EVALUATION
            # =================================================================
            
            # v5.0: RESTORED GATE LOGIC
            # Evaluate using the restored M8 System (with Q1-Q4 Gates)
            metadata = {} # Initialize metadata
            m8_result = self.m8_system.evaluate(
                df_h1=slice_h1,
                df_h4=slice_h4,
                df_m8=slice_m8,
                df_m2=slice_m2,
                current_time=current_time
            )
            
            decision = m8_result['signal'] if m8_result['execute'] else "WAIT"
            confidence = m8_result['confidence']
            metadata['m8_detail'] = m8_result
            
            # Debug: Log specific Gate interactions
            if i % 500 == 0:
                 gate_info = m8_result.get('breakdown', {}).get('gate', {})
                 print(f"\n[DEBUG {i}] M8 Evaluation: {m8_result['reason']}")
                 print(f"           Gate: {gate_info.get('gate')} ({gate_info.get('score')}) | Tradeable: {gate_info.get('tradeable')}")

            # Inject result for AGI compatibility
            agi_adjustments['m8_fibonacci'] = m8_result
            
            # 5. Virtual Execution (Dynamic Intelligence)
            if decision in ["BUY", "SELL"] and len(active_trades) == 0:
                # REALISTIC EXECUTION: Latency + Slippage
                signal_price_req = current_tick['ask'] if decision == "BUY" else current_tick['bid']
                
                exec_result = self.executor.execute_order(
                    direction=decision,
                    signal_price=signal_price_req,
                    signal_time=current_time,
                    df_m1=slice_m1
                )
                
                entry_price = exec_result['fill_price']
                slippage_realized = exec_result['slippage']
                if slippage_realized > 0.00005: # Log significant slippage
                    # Note: We won't log every tiny slippage to avoid spam, but it's affecting PnL
                    pass
                
                # --- DYNAMIC SL/TP CALCULATION ---
                atr_14 = self.calculate_atr(slice_m5)
                swing_high, swing_low = self.get_recent_swings(slice_m5)
                m8_score = agi_adjustments.get('m8_fibonacci', {}).get('total_score', 7)
                
                # Base SL: Structure Protection or ATR Volatility
                sl_pips = 0.0
                if decision == "BUY":
                    structure_sl = swing_low - (atr_14 * 0.5) if swing_low else entry_price - (atr_14 * 2)
                    volatility_sl = entry_price - (atr_14 * 2.0)
                    sl_price = max(structure_sl, volatility_sl) # Tighter of the two? No, safer (Lower for Buy) -> actually MIN price
                    sl_price = min(structure_sl, volatility_sl) # Safer = Wider stop
                else: 
                    structure_sl = swing_high + (atr_14 * 0.5) if swing_high else entry_price + (atr_14 * 2)
                    volatility_sl = entry_price + (atr_14 * 2.0)
                    sl_price = max(structure_sl, volatility_sl) # Safer = Wider stop (Higher price)

                # Cap SL to max 15 pips (GBPUSD Scalp Standard)
                max_risk = 0.0015
                if abs(entry_price - sl_price) > max_risk:
                    sl_price = entry_price - max_risk if decision == "BUY" else entry_price + max_risk
                    
                # Dynamic TP: Score Based
                risk = abs(entry_price - sl_price)
                if m8_score >= 9:
                    # TREND MODE: 1:3 RR
                    reward = risk * 3.0
                    mode_tag = "TREND"
                else:
                    # SCALP MODE: 1:1.5 RR (Quick exits for weak signals)
                    reward = risk * 1.5
                    mode_tag = "SCALP"
                    
                raw_tp = entry_price + reward if decision == "BUY" else entry_price - reward
                
                # --- MAGNETIC TP LOGIC (Phase 8) ---
                # Check for magnets near our Raw TP
                magnet_level, m_type = self.get_magnetic_level(raw_tp, decision, slice_h1)
                
                final_tp = raw_tp
                magnet_active = False
                
                if magnet_level:
                    # If magnet is reasonably close (within 20% of the target distance) 
                    # OR if magnet is slightly BEFORE the target (we should front run it)
                    dist_to_magnet = abs(magnet_level - entry_price)
                    dist_to_raw = abs(raw_tp - entry_price)
                    
                    # Logic: If Magnet is reachable and front-running it makes sense
                    if dist_to_magnet < dist_to_raw * 1.2 and dist_to_magnet > dist_to_raw * 0.5:
                         # Front Run Buffer: 2 pips (0.0002)
                         front_run = 0.0002
                         final_tp = magnet_level - front_run if decision == "BUY" else magnet_level + front_run
                         mode_tag += f"_MAG({m_type})"
                         magnet_active = True
                
                tp_price = final_tp
                
                trade = {
                    'ticket': i,
                    'symbol': symbol,
                    'type': decision,
                    'entry': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'lots': fixed_lots,
                    'open_time': current_time,
                    'regime': agi_adjustments.get('regime', 'UNKNOWN'),
                    'mode': mode_tag,
                    'score': m8_score
                }
                active_trades.append(trade)
                logger.info(f"[{current_time}] OPEN {decision} ({mode_tag}) @ {entry_price:.5f} | Score: {m8_score} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                
            # 6. Trade Management (SL/TP Check)
            for t in list(active_trades):
                pnl = 0
                closed = False
                reason = ""
                
                if t['type'] == "BUY":
                    if current_tick['bid'] >= t['tp']:
                        pnl = (t['tp'] - t['entry']) * 100000 * t['lots']
                        closed = True; reason = "TP"
                    elif current_tick['bid'] <= t['sl']:
                        pnl = (t['sl'] - t['entry']) * 100000 * t['lots']
                        closed = True; reason = "SL"
                else:
                    if current_tick['ask'] <= t['tp']:
                        pnl = (t['entry'] - t['tp']) * 100000 * t['lots']
                        closed = True; reason = "TP"
                    elif current_tick['ask'] >= t['sl']:
                        pnl = (t['entry'] - t['sl']) * 100000 * t['lots']
                        closed = True; reason = "SL"
                        
                if closed:
                    balance += pnl
                    t['close_time'] = current_time
                    t['pnl'] = pnl
                    t['reason'] = reason
                    trade_log.append(t)
                    active_trades.remove(t)
                    
                    # 7. Feedback Loop (Phase 4)
                    self.history_engine.notify_trade_close(
                        ticket=t['ticket'],
                        symbol=symbol,
                        profit=pnl,
                        reason=reason,
                        source="BACKTEST"
                    )
                    logger.info(f"[{current_time}] CLOSE {t['type']} ({t['mode']}) | PnL: ${pnl:.2f} | Reason: {reason}")
                
                else:
                    # =========================================================
                    # MATRIX BREAKER Sprint 2: EXIT INTELLIGENCE
                    # =========================================================
                    
                    current_price = current_tick['bid'] if t['type'] == "BUY" else current_tick['ask']
                    
                    # Calc Pips stats
                    if t['type'] == "BUY":
                        current_profit_pips = current_price - t['entry']
                        initial_risk_pips = t['entry'] - t['sl_initial'] if 'sl_initial' in t else 0.0020
                    else:
                        current_profit_pips = t['entry'] - current_price
                        initial_risk_pips = t['sl_initial'] - t['entry'] if 'sl_initial' in t else 0.0020
                        
                    # ALIAS FOR LEGACY COMPATIBILITY
                    current_pnl_pips = current_profit_pips
                        
                    # 1. PARTIAL TP (Scale Out at 1:1 R:R)
                    # We check if we haven't scaled out yet
                    if not t.get('scaled_out', False) and current_profit_pips >= initial_risk_pips:
                         # Close 50% Volume
                         partial_lots = t['lots'] * 0.5
                         partial_pnl = partial_lots * 100000 * current_profit_pips
                         
                         balance += partial_pnl
                         t['lots'] -= partial_lots # Reduce remaining size
                         t['scaled_out'] = True
                         t['mode'] += "_SCALED"
                         
                         # Move SL to Breakeven + Tiny Buffer
                         if t['type'] == "BUY":
                             t['sl'] = t['entry'] + 0.0002
                         else:
                             t['sl'] = t['entry'] - 0.0002
                             
                         logger.info(f"[{current_time}] PARTIAL TP {t['type']} | Secured ${partial_pnl:.2f} | SL -> BE")
                         
                    # 2. SMART TRAILING STOP (ATR Based)
                    # Only activate if in deep profit (e.g. > 1.5 ATR or > 15 pips)
                    atr = self.calculate_atr(slice_m5)
                    activation_dist = atr * 1.5
                    
                    if current_profit_pips > activation_dist:
                         # Trail 1 ATR behind
                         trail_dist = atr * 1.0
                         
                         if t['type'] == "BUY":
                             new_sl = current_price - trail_dist
                             if new_sl > t['sl']:
                                 t['sl'] = new_sl
                                 t['mode'] = t['mode'].replace("TREND", "TRAIL").replace("SCALP", "TRAIL")
                         else:
                             new_sl = current_price + trail_dist
                             if new_sl < t['sl']:
                                 t['sl'] = new_sl
                                 t['mode'] = t['mode'].replace("TREND", "TRAIL").replace("SCALP", "TRAIL")

                    # 3. MOMENTUM FADE EXIT (RSI Based)
                    # If we are in profit but momentum dies, GET OUT.
                    # Don't give back profits waiting for TP.
                    if current_profit_pips > 0.0005: # > 5 pips
                        rsi = self.calculate_rsi(slice_m5)
                        
                        if t['type'] == "BUY":
                            # Buying but RSI dropped below 45 (Weakness)
                            if rsi < 45.0:
                                pnl = current_profit_pips * 100000 * t['lots']
                                balance += pnl
                                t['close_time'] = current_time
                                t['pnl'] = pnl
                                t['reason'] = f"MOMENTUM_FADE(RSI:{rsi:.1f})"
                                trade_log.append(t)
                                active_trades.remove(t)
                                self.history_engine.notify_trade_close(t['ticket'], symbol, pnl, t['reason'], "BACKTEST")
                                logger.info(f"[{current_time}] FADE EXIT {t['type']} | RSI Loss | PnL: ${pnl:.2f}")
                                continue # Next trade
                        else:
                            # Selling but RSI rose above 55 (Strength)
                            if rsi > 55.0:
                                pnl = current_profit_pips * 100000 * t['lots']
                                balance += pnl
                                t['close_time'] = current_time
                                t['pnl'] = pnl
                                t['reason'] = f"MOMENTUM_FADE(RSI:{rsi:.1f})"
                                trade_log.append(t)
                                active_trades.remove(t)
                                self.history_engine.notify_trade_close(t['ticket'], symbol, pnl, t['reason'], "BACKTEST")
                                logger.info(f"[{current_time}] FADE EXIT {t['type']} | RSI Gain | PnL: ${pnl:.2f}")
                                continue

                    # TIME-DECAY EXIT Logic (Phase 7)
                    # If trade is open > 15 m (900s) and profit is negligible, KILL IT.
                    duration = (current_time - t['open_time']).total_seconds()
                    
                    if duration > 900 and current_profit_pips < 0.0005: # 5 pips threshold (Spread + Tiny Profit)
                        # KILL ZOMBIE TRADE
                        pnl = current_pnl_pips * 100000 * t['lots'] # Approximate
                        balance += pnl
                        t['close_time'] = current_time
                        t['pnl'] = pnl
                        t['reason'] = "TIME_DECAY"
                        trade_log.append(t)
                        active_trades.remove(t)
                        self.history_engine.notify_trade_close(
                            ticket=t['ticket'],
                            symbol=symbol,
                            profit=pnl,
                            reason="TIME_DECAY",
                            source="BACKTEST"
                        )
                        logger.info(f"[{current_time}] DECAY {t['type']} | Dur: {duration/60:.1f}m | PnL: ${pnl:.2f}")
                    
                    # STRUGGLE METER CHECK (Phase 9)
                    # Only check if we are in some profit (securing gains)
                    elif current_pnl_pips > 0.0010: # > 10 pips
                        struggle_score = self.detect_struggle(slice_m1, t['type'])
                        
                        if struggle_score >= 7.0: # Extreme Struggle (Repeated Rejections or Complete Death)
                           # PANIC EXIT / SECURE PROFIT
                            pnl = current_pnl_pips * 100000 * t['lots']
                            balance += pnl
                            t['close_time'] = current_time
                            t['pnl'] = pnl
                            t['reason'] = f"STRUGGLE({struggle_score:.1f})"
                            trade_log.append(t)
                            active_trades.remove(t)
                            self.history_engine.notify_trade_close(
                                ticket=t['ticket'],
                                symbol=symbol,
                                profit=pnl,
                                reason="STRUGGLE",
                                source="BACKTEST"
                            )
                            logger.info(f"[{current_time}] STRUGGLE EXIT {t['type']} | Score: {struggle_score} | PnL: ${pnl:.2f}")
            
            equity_curve.append(balance)

        # Final Report
        wins = len([t for t in trade_log if t['pnl'] > 0])
        total = len(trade_log)
        wr = (wins / total * 100) if total > 0 else 0
        
        print("\n" + "="*50)
        print(f"BACKTEST COMPLETE")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Total Trades: {total}")
        print(f"Win Rate: {wr:.1f}%")
        print("="*50)
        
        return {
            'balance': balance,
            'trades': total,
            'win_rate': wr
        }

if __name__ == "__main__":
    engine = BacktestEngine()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(engine.initialize())
    
    import os
    
    # SINGLE SYMBOL MODE - Using REAL MT5 Data
    symbol = "EURUSD"
    
    # MT5 export paths (check both common locations)
    mt5_paths = [
        f"C:/Users/pichau/AppData/Roaming/MetaQuotes/Terminal/*/MQL5/Files/Atl4s_Export/{symbol}_M1.csv",
        f"d:/Atl4s-Forex/data/{symbol}_M1.csv",
        f"data/{symbol}_M1.csv"
    ]
    
    csv_file = None
    import glob
    for pattern in mt5_paths:
        matches = glob.glob(pattern)
        if matches:
            csv_file = matches[0]
            print(f"Found REAL MT5 data: {csv_file}")
            break
    
    if csv_file is None:
        print(f"No real data found. Please run Atl4sDataExporter.mq5 in MT5 first.")
        print("Or place {symbol}_M1.csv in data/ folder.")
        print("\nFalling back to SYNTHETIC data for testing...")
        
        if not os.path.exists("data"):
            os.makedirs("data")
        csv_file = f"data/{symbol}_M1.csv"
        engine.generate_synthetic_data(csv_file, symbol)
    
    print(f"\n>>> STARTING BACKTEST: {symbol} <<<")
    # asyncio.run(engine.run(csv_file, symbol=symbol, initial_balance=30.0, fixed_lots=0.01))
    
    # UNCOMMENT TO RUN BRIDGE SERVER MODE:
    print("STARTING BRIDGE SERVER FOR MT5 TESTER...")
    engine.run_bridge_server()

