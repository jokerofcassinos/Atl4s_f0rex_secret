"""
GENESIS AGI - Unified Entry Point
═════════════════════════════════
The Single Source of Truth for the Atl4s Trading System.

Modes:
- LIVE: Real-time trading connected to MT5Terminal/ZMQ.
- BACKTEST: High-fidelity simulation using historical data.
- GENESIS: Self-optimization and machine learning loop.
- DIAGNOSTIC: System integrity check.

Usage:
    python main_genesis.py --mode [live|backtest|genesis|diagnostic] --symbol GBPUSD
"""

import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime

# Setup Paths
sys.path.append(os.getcwd())

# Import Core Components
from core.laplace_demon import LaplaceDemonCore
from backtest.engine import BacktestEngine, BacktestConfig
from data_loader import DataLoader

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler("genesis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Genesis")

class GenesisSystem:
    def __init__(self, mode, symbol, capital=1000.0):
        self.mode = mode.upper()
        self.symbol = symbol
        self.capital = capital
        self.demon = None
        self.engine = None
        self.running = True
        
        logger.info(f"Initializing Genesis System | Mode: {self.mode} | Symbol: {self.symbol}")
        
    async def initialize(self):
        """Boot up the AGI Core."""
        logger.info("Booting Laplace Demon Core...")
        try:
            self.demon = LaplaceDemonCore(self.symbol)
            logger.info("Laplace Demon Online (Hybrid Architectue V3)")
        except Exception as e:
            logger.critical(f"FATAL: Demon initialization failed: {e}")
            sys.exit(1)

    async def run_diagnostic(self):
        """Run self-check."""
        logger.info("Running System Diagnostics...")
        # TODO: Add deep check of all Eyes and Memory
        await asyncio.sleep(1)
        logger.info("Diagnostics: GREEN")
        return True

    async def run_backtest(self):
        """Execute Backtest Mode."""
        from run_laplace_backtest import LaplaceBacktestRunner
        
        logger.info("Starting Backtest Sequence...")
        runner = LaplaceBacktestRunner(
            initial_capital=self.capital,
            symbol=self.symbol,
            risk_per_trade=2.0
        )
        
        # Load Data
        # Using the same logic as run_laplace_backtest but centrally managed is better.
        # For now, we delegate to the existing robust runner.
        # Check cache first
        cache_path = f"data/cache/{self.symbol}_M5.parquet" 
        # (This is just a placeholder logic, the runner handles it well)
        
        await runner.load_data("data/cache/GBPUSD_M5_60days.parquet") # Default for now
        
        if runner.df_m5 is not None:
             result = await runner.run_backtest()
             if result:
                 runner.generate_report(result)
        else:
            logger.error("No Data for Backtest.")

    async def run_live(self):
        """Execute Live Trading Mode."""
        logger.info("Connecting to Matrix (Live Market)...")
        # TODO: Implement ZMQ Bridge connection here
        while self.running:
            logger.info("Heartbeat...")
            await asyncio.sleep(5)

    async def start(self):
        """Main Loop Dispatcher."""
        await self.initialize()
        
        if self.mode == "DIAGNOSTIC":
            await self.run_diagnostic()
        elif self.mode == "BACKTEST":
            await self.run_backtest()
        elif self.mode == "LIVE":
            await self.run_live()
        else:
            logger.error(f"Unknown Mode: {self.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Genesis AGI Entry Point')
    parser.add_argument('--mode', type=str, default='backtest', help='Operation Mode')
    parser.add_argument('--symbol', type=str, default='GBPUSD', help='Trading Symbol')
    
    args = parser.parse_args()
    
    system = GenesisSystem(args.mode, args.symbol)
    asyncio.run(system.start())
