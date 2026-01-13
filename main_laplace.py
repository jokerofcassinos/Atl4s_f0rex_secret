"""
LAPLACE DEMON - Live Trading System
════════════════════════════════════

A deterministic trading intelligence for FOREX markets.
Integrates advanced institutional theories into a clean, maintainable system.

Named after Laplace's Demon - the entity that could predict the future
perfectly by knowing all positions and forces in the universe.

Target: 70% Win Rate | $30+ Capital | GBPUSD
"""

import asyncio
import logging
import datetime
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import Optional, Dict, Any

# Core Components
from core.zmq_bridge import ZmqBridge
from core.execution_engine import ExecutionEngine
from data_loader import DataLoader

# Laplace Demon Intelligence
from core.laplace_demon import LaplaceDemonCore, LaplacePrediction

# Configuration
from config import (
    INITIAL_CAPITAL, RISK_PER_TRADE, SYMBOLS, LEVERAGE,
    KILLZONES, SPREAD_LIMITS
)

# Analytics
from analytics.telegram_notifier import get_notifier

# Auto-Training
from train_oracle_v2 import run_auto_training

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Mute Peewee (YFinance internal DB)
logging.getLogger("peewee").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("laplace_demon.log"),
        logging.StreamHandler()
    ]
)

# Silence noisy libraries
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# --- SILENCE AGI NOISE (The "Holographic" Garbage) ---
NOISY_LOGGERS = [
    "RecursiveReflection", "InfiniteWhyEngine", "HolographicMemory", 
    "HolographicPlateUltra", "AGISwarmAdapter", "MCTS_Planner", 
    "HealthMonitor", "UnifiedReasoning", "NeuroPlasticity", 
    "EvolutionEngine", "MemoryIntegration", "CortexMemory",
    "NeuralOracle", "Consensus", "ZmqBridge", "ExecutionEngine",
    "CausalGraph", "SwarmOrchestrator"
]
for log_name in NOISY_LOGGERS:
    logging.getLogger(log_name).setLevel(logging.WARNING)

# Full Silence for Reflection Warnings
logging.getLogger("RecursiveReflection").setLevel(logging.ERROR)

logger = logging.getLogger("LaplaceDemon")

import subprocess
import os
import sys

def run_auto_training():
    """
    Phase 4: Auto-Evolution.
    Checks for new data and retrains the Neural Oracle before trading starts.
    """
    try:
        if os.path.exists("data/training/live_trades.csv"):
            logger.info("🧠 EVOLUTION: Checking for new training data...")
            # Run detached to not block startup significantly, but give it a moment
            subprocess.Popen([sys.executable, "train_oracle_v2.py"])
            time.sleep(2) 
    except Exception as e:
        logger.error(f"Auto-Training Trigger Failed: {e}")


# ============================================================================
# LAPLACE DEMON TRADING SYSTEM
# ============================================================================

class LaplaceTradingSystem:
    """
    Main trading system using the Laplace Demon for signal generation.
    
    Responsibilities:
    1. Connect to MT5 via ZMQ bridge
    2. Fetch and cache market data
    3. Generate signals via Laplace Demon
    4. Execute trades via ExecutionEngine
    5. Monitor and manage open positions
    """
    
    def __init__(self, zmq_port: int = 5558, symbol: str = "GBPUSD"):
        """Initialize the trading system."""
        
        # Symbol and configuration
        self.symbol = symbol
        self.config = {
            'initial_capital': INITIAL_CAPITAL,
            'risk_per_trade': RISK_PER_TRADE,
            'leverage': LEVERAGE,
            'max_concurrent_trades': 30, # Match Backtest (Aggressive Scaling)
            'virtual_sl': INITIAL_CAPITAL * 0.60,  # Catastrophe Stop (60% of Equity - Backtest Parity)
            'virtual_tp': INITIAL_CAPITAL * 1.50,  # Moonshot Take Profit (150% - Not a hard limit)
            'spread_limit': SPREAD_LIMITS.get(symbol, 0.00030),
            'mode': 'LAPLACE'
        }
        
        # Analytics
        self.telegram = get_notifier()
        
        # Core components
        self.bridge = ZmqBridge(port=zmq_port)
        self.data_loader = DataLoader()
        self.executor = ExecutionEngine(self.bridge, notifier=self.telegram)
        self.executor.set_config(self.config)
        
        # Laplace Demon - The Brain
        self.laplace = LaplaceDemonCore(symbol)
        self.laplace.set_bridge(self.bridge) # Phase 1 Integration: Inject Execution Authority
        
        # Phase 4 Integration: Connect Learning Feedback Loop
        self.executor.register_learning_engine(self.laplace)
        
        # State
        self.cached_data: Dict[str, pd.DataFrame] = {}
        self.last_data_fetch: float = 0
        self.data_refresh_interval: int = 60  # seconds
        self.last_signal_time: float = 0
        self.min_signal_interval: int = 60  # seconds between signals
        self.system_start_time: float = 0
        self.warm_up_period: int = 30  # seconds
        
        # Statistics
        self.tick_count: int = 0
        self.signal_count: int = 0
        self.trade_count: int = 0
        
        logger.info(f"LAPLACE DEMON INITIALIZED | Symbol: {symbol}")
    
    async def start(self):
        """Start the trading system."""
        print("\n" + "═" * 60)
        print("  🔮 LAPLACE DEMON - DETERMINISTIC TRADING INTELLIGENCE 🔮")
        print("═" * 60)
        print(f"\n  Symbol: {self.symbol}")
        print(f"  Capital: ${self.config['initial_capital']}")
        print(f"  Risk/Trade: {self.config['risk_per_trade']}%")
        print(f"  Mode: {self.config['mode']}")
        print("\n" + "═" * 60 + "\n")
        
        # Wait for bridge connection
        await self._wait_for_connection()
        
        # Initial data load
        await self._refresh_data()
        
        # Record start time
        self.system_start_time = time.time()
        
        # Run main loop
        await self._main_loop()
    
    async def _wait_for_connection(self, timeout: int = 60):
        """Wait for ZMQ bridge to receive first tick."""
        print("═" * 60 + "\n")
        
        # Awakening
        await self.laplace.initialize()
        
        # Connect to MT5
        logger.info("Waiting for MT5 connection...")
        
        for i in range(timeout):
            tick = self.bridge.get_tick()
            if tick:
                logger.info(f"MT5 Connected! First tick: {tick.get('symbol')}")
                
                # Auto-Detect Capital
                if 'equity' in tick:
                    detected_capital = float(tick['equity'])
                    if detected_capital > 0:
                        self.config['initial_capital'] = detected_capital
                        # Recalculate Catastrophe Stops (Backtest Parity: 60%)
                        self.config['virtual_sl'] = detected_capital * 0.60
                        self.config['virtual_tp'] = detected_capital * 1.50
                        logger.info(f"Account Capital Detected: ${detected_capital:.2f}")
                        
                        # Dynamic Leverage (FTMO Compliance)
                        if detected_capital > 1000:
                             self.config['leverage'] = 50.0
                        else:
                             self.config['leverage'] = 100.0
                             
                        logger.info(f"FTMO LEVERAGE ADJUSTED: 1:{int(self.config['leverage'])}")
                        
                        self.telegram.notify_system_status("ONLINE", f"Symbol: {self.symbol} | Capital: ${detected_capital:.2f} | Lev: 1:{int(self.config['leverage'])}")
                
                return True
            
            if i % 10 == 0:
                print(f"[LAPLACE]: Waiting for MT5 tick... ({i}/{timeout}s)")
            
            await asyncio.sleep(1)
        
        logger.warning("⚠️ No MT5 connection. Running in simulation mode.")
        return False
    
    async def _refresh_data(self):
        """Refresh OHLCV data from data loader."""
        try:
            # Get historical data
            await self.data_loader.get_data(self.symbol)
            
            # Create timeframe map
            if hasattr(self.data_loader, 'cache_data'):
                m1_data = self.data_loader.cache_data.get('M1')
                
                if m1_data is not None and len(m1_data) > 100:
                    # Fix: Robust Deduplication (Groupby is safer than duplicated())
                    m1_data = m1_data.groupby(m1_data.index).last()
                    m1_data = m1_data.sort_index()
                    
                    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    
                    self.cached_data['M1'] = m1_data
                    self.cached_data['M5'] = m1_data.resample('5min').agg(agg).dropna()
                    self.cached_data['H1'] = m1_data.resample('1h').agg(agg).dropna()
                    self.cached_data['H4'] = m1_data.resample('4h').agg(agg).dropna()
                    self.cached_data['D1'] = m1_data.resample('1D').agg(agg).dropna() # Added D1
                    
                    logger.info(f"Data refreshed: M1={len(m1_data)}, M5={len(self.cached_data['M5'])}, H1={len(self.cached_data['H1'])}, H4={len(self.cached_data['H4'])}, D1={len(self.cached_data['D1'])}")
            
            self.last_data_fetch = time.time()
            
        except Exception as e:
            logger.warning(f"Data refresh failed: {e}")
    
    async def _main_loop(self):
        """Main trading loop."""
        logger.info("Starting main trading loop...")
        
        last_heartbeat = datetime.datetime.now()
        
        while True:
            try:
                # Get tick
                tick = self._get_tick()
                
                if not tick:
                    await asyncio.sleep(0.5)
                    continue
                
                self.tick_count += 1
                
                # Heartbeat logging & Telegram Report
                now = datetime.datetime.now()
                if (now - last_heartbeat).total_seconds() >= 3600: # Hourly Report
                     self.telegram.send_sync(f"⏱ *HOURLY REPORT*\nTicks: {self.tick_count}\nSignals: {self.signal_count}\nTrades: {self.trade_count}\nCapital: ${self.config['initial_capital']:.2f}")
                     last_heartbeat = now
                     
                if (now - last_heartbeat).total_seconds() >= 30:
                    logger.info(f"[HEARTBEAT] Ticks: {self.tick_count} | Signals: {self.signal_count} | Trades: {self.trade_count}")
                    last_heartbeat = now
                    self.tick_count = 0
                
                # Update symbol from tick
                self.symbol = tick.get('symbol', self.symbol)
                current_price = tick.get('bid', 0)
                current_time = datetime.datetime.now()
                
                # Sync Capital (Auto-Compounding)
                if 'equity' in tick:
                     equity = float(tick['equity'])
                     equity = float(tick['equity'])
                     self.config['initial_capital'] = equity
                     self.config['virtual_sl'] = equity * 0.60 # 60% Safety Net
                     self.config['virtual_tp'] = equity * 1.50
                     
                     # Sync Leverage
                     if equity > 1000: self.config['leverage'] = 50.0
                     else: self.config['leverage'] = 100.0
                
                # Manage existing positions first
                await self._manage_positions(tick)
                
                # Check if we should attempt new signals
                if not self._should_check_signals():
                    await asyncio.sleep(0.1)
                    continue
                
                # Refresh data if needed
                if time.time() - self.last_data_fetch > self.data_refresh_interval:
                    await self._refresh_data()
                
                # Check if we have data
                if self.cached_data.get('M5') is None or self.cached_data.get('M5').empty:
                    logger.warning("Waiting for data... (M5 DataFrame empty)")
                    await asyncio.sleep(1)
                    continue
                
                # Generate Laplace Demon prediction
                prediction = await self._get_prediction(current_time, current_price)
                
                if prediction and prediction.execute:
                    await self._execute_signal(prediction, tick)
                
                await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Trading loop stopped.")
    
    def _get_tick(self) -> Optional[Dict]:
        """Get current tick from bridge or simulate."""
        tick = self.bridge.get_tick()
        
        if tick:
            # Check for staleness
            tick_time = tick.get('time', 0)
            if time.time() - tick_time > 5:
                # return None # Allow some staleness for now if needed
                pass 
            return tick
        
        # Simulation mode fallback
        return {
            'symbol': self.symbol,
            'time': int(time.time()),
            'time_msc': int(time.time() * 1000),
            'bid': 1.2500,
            'ask': 1.2502,
            'volume': 100
        }
    
    def _should_check_signals(self) -> bool:
        """Check if we should look for new signals."""
        now = time.time()
        
        # Warm-up period
        if now - self.system_start_time < self.warm_up_period:
            return False
        
        # Minimum interval between signals
        if now - self.last_signal_time < self.min_signal_interval:
            return False
        
        # Check if in killzone
        hour = datetime.datetime.now().hour
        in_killzone = False
        
        for zone, params in KILLZONES.items():
            if params['start'] <= hour < params['end']:
                in_killzone = True
                break
        
        if not in_killzone:
            # We CONTINUE to analyze for logs, but we will block execution later.
            pass
        
        # Check max concurrent trades
        open_trades = self.bridge.get_open_trades()
        if len(open_trades) >= self.config['max_concurrent_trades']:
            return False
        
        return True
    
    def _is_in_killzone(self) -> bool:
        """Helper to check strict trading hours."""
        hour = datetime.datetime.now().hour
        for zone, params in KILLZONES.items():
            if params['start'] <= hour < params['end']:
                return True
        return False

    async def _get_prediction(self, current_time: datetime.datetime, current_price: float) -> Optional[LaplacePrediction]:
        """Get prediction from Laplace Demon."""
        try:
            prediction = await self.laplace.analyze(
                df_m1=self.cached_data.get('M1'),
                df_m5=self.cached_data.get('M5'),
                df_h1=self.cached_data.get('H1'),
                df_h4=self.cached_data.get('H4'),
                df_d1=self.cached_data.get('D1'), 
                current_time=current_time,
                current_price=current_price
            )
            
            # KILLZONE FILTER (Execution Gate)
            if prediction.execute and not self._is_in_killzone():
                logger.info(f"🛑 SIGNAL IGNORED: Outside Killzone ({current_time.hour}h)")
                prediction.execute = False
                return prediction

            if prediction.execute:
                self.signal_count += 1
                logger.info(
                    f"🎯 LAPLACE SIGNAL: {prediction.direction} | "
                    f"Conf: {prediction.confidence:.0f}% | "
                    f"Confluence: {prediction.confluence_count} | "
                    f"SL: {prediction.sl_pips}p | TP: {prediction.tp_pips}p"
                )
                
                for reason in prediction.reasons[:3]:
                    logger.info(f"   → {reason}")
                
                if prediction.warnings:
                    for warning in prediction.warnings[:2]:
                        logger.warning(f"   ⚠ {warning}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    async def _execute_signal(self, prediction: LaplacePrediction, tick: Dict):
        """Execute a trade based on Laplace prediction."""
        try:
            symbol = tick.get('symbol', self.symbol)
            current_price = tick.get('ask' if prediction.direction == 'BUY' else 'bid', 0)
            
            # Calculate lot size based on risk
            risk_amount = self.config['initial_capital'] * (self.config['risk_per_trade'] / 100)
            pip_value = 0.0001
            sl_pips = prediction.sl_pips
            
            # Lot size = Risk Amount / (SL pips * pip value * lot multiplier)
            lot_multiplier = 10  # For standard forex
            lots = risk_amount / (sl_pips * pip_value * lot_multiplier)
            lots = max(0.01, min(lots, 1.0))  # Clamp between 0.01 and 1.0
            
            # Apply position multiplier from volatility
            lots *= prediction.lot_multiplier
            lots = round(lots, 2)
            
            # Execute via bridge
            order_type = prediction.direction
            
            logger.info(f"⚡ EXECUTING: {order_type} {lots} lots @ {current_price:.5f}")
            logger.info(f"   ∟ SL: {prediction.sl_price:.5f} ({prediction.sl_pips}p) | TP: {prediction.tp_price:.5f} ({prediction.tp_pips}p)")
            
            result = self.executor.execute_trade(
                symbol=symbol,
                direction=prediction.direction,
                lots=lots,
                sl_pips=prediction.sl_pips,
                tp_pips=prediction.tp_pips,
                comment=f"LAPLACE_{prediction.strength.name}"
            )
            
            if result:
                self.trade_count += 1
                self.last_signal_time = time.time()
                logger.info(f"✅ Trade executed: {result}")
                
                # Telegram Notification
                await self.telegram.notify_trade_entry(
                    direction=prediction.direction,
                    symbol=symbol,
                    entry=current_price,
                    sl=current_price - (prediction.sl_pips * 0.0001) if prediction.direction == "BUY" else current_price + (prediction.sl_pips * 0.0001),
                    tp=current_price + (prediction.tp_pips * 0.0001) if prediction.direction == "BUY" else current_price - (prediction.tp_pips * 0.0001),
                    confidence=prediction.confidence,
                    setup=prediction.primary_signal
                )
            else:
                logger.warning("❌ Trade execution failed")
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
    
    async def _manage_positions(self, tick: Dict):
        """
        Delegate position management to the ExecutionEngine (The Hand of God).
        This includes VSL, VTP, Trailing Stops, and Predictive Exits.
        """
        try:
            # We pass the tick to the specialized engine
            await self.executor.monitor_positions(tick)
            
            # Also run Event Horizon (Dynamic Stops)
            await self.executor.manage_dynamic_stops(tick)
                    
        except Exception as e:
            logger.error(f"Position management error: {e}")
                    
        except Exception as e:
            logger.error(f"Position management error: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point."""
    import sys
    
    # Parse arguments
    symbol = "GBPUSD"
    port = 5558
    
    for i, arg in enumerate(sys.argv):
        if arg == "--symbol" and i + 1 < len(sys.argv):
            symbol = sys.argv[i + 1]
        elif arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    # Create and start system
    system = LaplaceTradingSystem(zmq_port=port, symbol=symbol)
    await system.start()


if __name__ == "__main__":
    try:
        run_auto_training()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🔮 Laplace Demon shutting down gracefully...")
