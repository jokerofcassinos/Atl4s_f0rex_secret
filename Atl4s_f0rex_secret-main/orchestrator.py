import time
import datetime
import sys
import pytz
import logging
from pathlib import Path

# Add src to path if needed (though we will likely run from root)
sys.path.append(str(Path(__file__).parent / 'src'))

# Import modules (placeholders for now until we create them)
from src.data_loader import DataLoader
from src.analysis_engine import AnalysisEngine
from src.strategy_core import StrategyCore
from src.executor import Executor
from src.dashboard_generator import DashboardGenerator

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/atl4s.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Orchestrator")

class Atl4sOrchestrator:
    def __init__(self):
        logger.info("Initializing Atl4s-Forex Orchestrator...")
        self.running = False
        
        # Configuration
        self.timezone = pytz.timezone('America/Sao_Paulo') # Brasilia Time
        self.start_hour = 10
        self.end_hour = 12
        self.symbol = "XAUUSD"
        self.timeframe = "M5"
        
        # Initialize Sub-systems
        self.data_loader = DataLoader(self.symbol)
        self.analysis_engine = AnalysisEngine()
        self.strategy_core = StrategyCore()
        self.executor = Executor(self.symbol)
        self.dashboard = DashboardGenerator()
        
    def check_market_hours(self):
        """Checks if current time is within the operating window (10:00 - 12:00 BRT)."""
        now = datetime.datetime.now(self.timezone)
        
        # Weekend check
        if now.weekday() >= 5:
            logger.info("Weekend - Market Closed.")
            # return False # Commented out for Verification/Demo purposes if user runs on weekend

        # Hour check
        start_time = now.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
        end_time = now.replace(hour=self.end_hour, minute=0, second=0, microsecond=0)
        
        # For DEMO PURPOSE: if outside hours, warn but maybe allow?
        # Strict rule: return False
        
        if start_time <= now <= end_time:
            return True
        else:
            if now < start_time:
                logger.info(f"Waiting for market open at {self.start_hour}:00...")
            else:
                logger.info(f"Market closed for the day (Ended at {self.end_hour}:00).")
            return False # Strict adherence

    def run_daily_cycle(self):
        """Main execution loop for the trading session."""
        logger.info("Starting Daily Cycle...")
        self.running = True
        
        while self.running:
            try:
                # if not self.check_market_hours():
                #    time.sleep(60)
                #    continue
                # DISABLED TIME CHECK FOR FIRST RUN VERIFICATION
                
                logger.info(f"--- M5 Period Start: {datetime.datetime.now(self.timezone)} ---")
                
                # 1. Fetch Data
                logger.info("Step 1: Syncing Data...")
                df = self.data_loader.sync_data()
                
                if df.empty:
                    logger.warning("No data retrieved. Waiting...")
                    time.sleep(10)
                    continue

                # 2. Analyze
                logger.info("Step 2: Running Quantum Analysis...")
                df_enriched, prediction = self.analysis_engine.analyze(df)
                
                # 3. Strategy Decision
                logger.info("Step 3: Calculating Strategy...")
                account_info = self.data_loader.get_account_info()
                # Mock account info if None (e.g. MT5 fail)
                if not account_info:
                    account_info = {'equity': 30.0, 'balance': 30.0, 'profit': 0.0}
                    
                open_positions = self.executor.get_open_positions()
                decision = self.strategy_core.decide((df_enriched, prediction), account_info, open_positions)
                
                # 4. Execute
                if decision and decision['action'] != 'HOLD':
                    logger.info(f"Step 4: Executing {decision['action']} ({decision.get('reason')})")
                    self.executor.execute(decision)
                else:
                    logger.info("Step 4: No Trade (HOLD).")
                
                # 5. Dashboard
                self.dashboard.generate_report(df_enriched, account_info)
                
                # 6. Wait
                logger.info("Cycle Complete. Sleeping...")
                time.sleep(60) # Fast loop for demo
                
            except KeyboardInterrupt:
                logger.info("Bot manually stopped.")
                self.running = False
            except Exception as e:
                logger.error(f"Critical Error: {e}", exc_info=True)
                time.sleep(10)


if __name__ == "__main__":
    bot = Atl4sOrchestrator()
    bot.run_daily_cycle()
