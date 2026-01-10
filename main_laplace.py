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


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
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

logger = logging.getLogger("LaplaceDemon")


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
            'max_concurrent_trades': 3,
            'virtual_sl': 20.0,  # Virtual stop in $
            'virtual_tp': 40.0,  # Virtual TP in $
            'spread_limit': SPREAD_LIMITS.get(symbol, 0.00030),
            'mode': 'LAPLACE'
        }
        
        # Core components
        self.bridge = ZmqBridge(port=zmq_port)
        self.data_loader = DataLoader()
        self.executor = ExecutionEngine(self.bridge)
        self.executor.set_config(self.config)
        
        # Laplace Demon - The Brain
        self.laplace = LaplaceDemonCore(symbol)
        
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
        
        logger.info(f"🔮 LAPLACE DEMON INITIALIZED | Symbol: {symbol}")
    
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
        logger.info("Waiting for MT5 connection...")
        
        for i in range(timeout):
            tick = self.bridge.get_tick()
            if tick:
                logger.info(f"✅ MT5 Connected! First tick: {tick.get('symbol')}")
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
                    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                    
                    self.cached_data['M1'] = m1_data
                    self.cached_data['M5'] = m1_data.resample('5min').agg(agg).dropna()
                    self.cached_data['H1'] = m1_data.resample('1h').agg(agg).dropna()
                    self.cached_data['H4'] = m1_data.resample('4h').agg(agg).dropna()
                    self.cached_data['D1'] = m1_data.resample('1D').agg(agg).dropna() # Added D1
                    
                    logger.info(f"Data refreshed: M1={len(m1_data)}, M5={len(self.cached_data['M5'])}, D1={len(self.cached_data['D1'])}")
            
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
                
                # Heartbeat logging
                now = datetime.datetime.now()
                if (now - last_heartbeat).total_seconds() >= 30:
                    logger.info(f"[HEARTBEAT] Ticks: {self.tick_count} | Signals: {self.signal_count} | Trades: {self.trade_count}")
                    last_heartbeat = now
                    self.tick_count = 0
                
                # Update symbol from tick
                self.symbol = tick.get('symbol', self.symbol)
                current_price = tick.get('bid', 0)
                current_time = datetime.datetime.now()
                
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
                    # logger.warning("Waiting for data...")
                    await asyncio.sleep(1)
                    continue
                
                # Generate Laplace Demon prediction
                prediction = self._get_prediction(current_time, current_price)
                
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
            # logger.debug("Outside Killzone")
            return False
        
        # Check max concurrent trades
        open_trades = self.bridge.get_open_trades()
        if len(open_trades) >= self.config['max_concurrent_trades']:
            return False
        
        return True
    
    def _get_prediction(self, current_time: datetime.datetime, current_price: float) -> Optional[LaplacePrediction]:
        """Get prediction from Laplace Demon."""
        try:
            prediction = self.laplace.analyze(
                df_m1=self.cached_data.get('M1'),
                df_m5=self.cached_data.get('M5'),
                df_h1=self.cached_data.get('H1'),
                df_h4=self.cached_data.get('H4'),
                df_d1=self.cached_data.get('D1'), # Added for Phase 4
                current_time=current_time,
                current_price=current_price
            )
            
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
            lots *= prediction.position_multiplier
            lots = round(lots, 2)
            
            # Execute via bridge
            order_type = prediction.direction
            
            logger.info(f"⚡ EXECUTING: {order_type} {lots} lots @ {current_price:.5f}")
            
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
            else:
                logger.warning("❌ Trade execution failed")
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
    
    async def _manage_positions(self, tick: Dict):
        """Manage open positions with virtual SL/TP."""
        try:
            # Get open trades from tick
            open_trades = tick.get('trades_json', [])
            
            if not open_trades:
                return
            
            for trade in open_trades:
                profit = trade.get('profit', 0)
                
                # Virtual SL check
                if profit < -self.config['virtual_sl']:
                    logger.warning(f"🛑 VIRTUAL SL HIT: Closing trade {trade.get('ticket')} at ${profit:.2f}")
                    self.executor.close_trade(trade.get('ticket'))
                
                # Virtual TP check
                elif profit > self.config['virtual_tp']:
                    logger.info(f"🎯 VIRTUAL TP HIT: Closing trade {trade.get('ticket')} at ${profit:.2f}")
                    self.executor.close_trade(trade.get('ticket'))
                    
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
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🔮 Laplace Demon shutting down gracefully...")
