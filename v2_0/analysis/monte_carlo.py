import numpy as np
import pandas as pd
import logging
import random

logger = logging.getLogger("Atl4s-MonteCarlo")

class MonteCarloSimulator:
    def __init__(self, num_simulations=1000, initial_capital=100.0):
        self.num_simulations = num_simulations
        self.initial_capital = initial_capital

    def run(self, trades_pnl_percent):
        """
        Runs Monte Carlo simulation on a list of trade PnL percentages.
        trades_pnl_percent: List of floats (e.g., [0.01, -0.005, 0.02]) representing 1% gain, 0.5% loss, etc.
        Returns:
            metrics (dict): Risk of Ruin, Median Drawdown, etc.
        """
        if not trades_pnl_percent or len(trades_pnl_percent) < 10:
            logger.warning("Insufficient trades for Monte Carlo simulation.")
            return {
                "risk_of_ruin": 0.0,
                "median_drawdown": 0.0,
                "safety_score": 50 # Neutral
            }

        ruin_count = 0
        max_drawdowns = []
        final_balances = []

        # Convert to numpy array for speed
        trades = np.array(trades_pnl_percent)

        for _ in range(self.num_simulations):
            # Shuffle trades
            np.random.shuffle(trades)
            
            # Simulate Equity Curve
            equity = [self.initial_capital]
            peak = self.initial_capital
            max_dd = 0.0
            ruined = False
            
            for pnl in trades:
                # Simple compounding: New Balance = Old Balance * (1 + PnL)
                # Or Fixed Risk? Let's assume the PnL% is realized on the current balance.
                current_balance = equity[-1] * (1 + pnl)
                
                if current_balance < (self.initial_capital * 0.5): # Ruin threshold (50% drawdown)
                    ruined = True
                
                equity.append(current_balance)
                
                # Drawdown Calc
                if current_balance > peak:
                    peak = current_balance
                dd = (peak - current_balance) / peak
                if dd > max_dd:
                    max_dd = dd
            
            if ruined:
                ruin_count += 1
            
            max_drawdowns.append(max_dd)
            final_balances.append(equity[-1])

        # Calculate Metrics
        risk_of_ruin = (ruin_count / self.num_simulations) * 100
        median_dd = np.median(max_drawdowns) * 100
        worst_case_dd = np.percentile(max_drawdowns, 95) * 100 # 95th percentile
        
        logger.info(f"Monte Carlo Results: RoR={risk_of_ruin:.1f}%, Median DD={median_dd:.1f}%, Worst DD={worst_case_dd:.1f}%")
        
        # Safety Score (0-100)
        # 100 = RoR 0% and Low DD
        # 0 = RoR > 5%
        score = 100 - (risk_of_ruin * 10) - (median_dd * 2)
        score = max(0, min(100, score))
        
        return {
            "risk_of_ruin": risk_of_ruin,
            "median_drawdown": median_dd,
            "worst_case_drawdown": worst_case_dd,
            "safety_score": score
        }
