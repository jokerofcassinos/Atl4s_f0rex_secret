import logging
import json
import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from data_loader import DataLoader
from backtest_engine import BacktestEngine
import config

# Configure logging for the simulation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Atl4s-Sim")

class SimulationSystem:
    def __init__(self):
        self.loader = DataLoader()
        self.engine = BacktestEngine()

    def run(self, days=30, params_file=None):
        logger.info(f"--- ATl4S SIMULATION SYSTEM START ---")
        logger.info(f"Period: Last {days} days")
        
        # Load Data
        data_map = self.loader.get_data()
        
        # Filter data for requested period
        if data_map['M5'] is not None:
            # yfinance returns UTC data by default
            utc = pytz.UTC
            cutoff = datetime.now(utc) - timedelta(days=days)
            data_map['M5'] = data_map['M5'][data_map['M5'].index >= cutoff]
            if data_map['H1'] is not None:
                data_map['H1'] = data_map['H1'][data_map['H1'].index >= cutoff]
        
        if data_map['M5'] is None or len(data_map['M5']) < 100:
            logger.error("Insufficient data for the selected period.")
            return

        # Load Custom Params if any
        params = None
        if params_file and os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded parameters from {params_file}")

        # Run Simulation
        results = self.engine.run_simulation(data_map, params=params, verbose=True)
        
        if "error" in results:
            logger.error(f"Simulation failed: {results['error']}")
            return

        self._print_summary(results)
        self._save_results(results)

    def _print_summary(self, res):
        print("\n" + "="*50)
        print("         SIMULATION RESULTS SUMMARY")
        print("="*50)
        print(f"Total Trades:     {res['total_trades']}")
        print(f"Win Rate:         {res['wr']*100:.1f}%")
        print(f"Profit Factor:    {res['pf']:.2f}")
        print(f"Final Balance:    ${res['final_balance']:.2f}")
        print(f"Safety Score:     {res['safety_score']}/100")
        print(f"Risk of Ruin:     {res['risk_of_ruin']:.1f}%")
        print("="*50)
        
        if res['trades']:
            print("\nLAST 5 TRADES:")
            for t in res['trades'][-5:]:
                color = "\033[92m" if t['pnl'] > 0 else "\033[91m"
                reset = "\033[0m"
                print(f"[{t['entry_time'].strftime('%Y-%m-%d %H:%M')}] {t['decision']} | PnL: {color}${t['pnl']:.2f}{reset} | Reason: {t['reason']}")
        print("="*50 + "\n")

    def _save_results(self, res):
        # Convert timestamps to string for JSON serialization
        results_copy = res.copy()
        for trade in results_copy['trades']:
            trade['entry_time'] = trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
            trade['exit_time'] = trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
        
        output_file = "simulation_report.json"
        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=4)
        logger.info(f"Detailed report saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Atl4s Simulation System")
    parser.add_argument("--days", type=int, default=15, help="Number of days to simulate")
    parser.add_argument("--params", type=str, default="optimal_params.json", help="Path to params JSON file")
    
    args = parser.parse_args()
    
    sim = SimulationSystem()
    sim.run(days=args.days, params_file=args.params)
