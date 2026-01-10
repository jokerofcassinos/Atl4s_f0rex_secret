"""
Atl4s Metrics Calculator

Calculates all backtest metrics including:
- Walk-forward analysis
- Monte Carlo simulation
- Statistical significance tests
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("Atl4s-Metrics")


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    fold_results: List[Dict]
    in_sample_metrics: Dict
    out_sample_metrics: Dict
    stability_ratio: float  # OOS performance / IS performance
    is_robust: bool


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    simulations: int
    median_final_equity: float
    percentile_5: float
    percentile_95: float
    probability_profit: float
    probability_ruin: float
    max_drawdown_median: float


class MetricsCalculator:
    """
    Advanced metrics calculation for backtesting.
    """
    
    @staticmethod
    def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        excess_returns = returns - risk_free_rate / 252
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return (excess_returns.mean() * 252) / (downside.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_calmar(total_return_pct: float, max_drawdown_pct: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if max_drawdown_pct == 0:
            return 0.0
        return total_return_pct / max_drawdown_pct
    
    @staticmethod
    def calculate_profit_factor(gross_profit: float, gross_loss: float) -> float:
        """Calculate profit factor."""
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / abs(gross_loss)
    
    @staticmethod
    def calculate_expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate mathematical expectancy per trade."""
        if avg_loss == 0:
            return avg_win * (win_rate / 100) if win_rate > 0 else 0
        return (win_rate / 100) * avg_win - (1 - win_rate / 100) * abs(avg_loss)
    
    @staticmethod
    def monte_carlo_simulation(trades: List, 
                                initial_capital: float = 30.0,
                                simulations: int = 1000,
                                ruin_threshold_pct: float = 50.0) -> MonteCarloResult:
        """
        Run Monte Carlo simulation to estimate distribution of outcomes.
        
        Randomly reorders trades to see range of possible equity curves.
        """
        if not trades:
            return MonteCarloResult(
                simulations=0, median_final_equity=initial_capital,
                percentile_5=initial_capital, percentile_95=initial_capital,
                probability_profit=0.0, probability_ruin=0.0, max_drawdown_median=0.0
            )
        
        pnls = [t.pnl_dollars for t in trades]
        final_equities = []
        max_drawdowns = []
        ruin_count = 0
        profit_count = 0
        
        ruin_level = initial_capital * (1 - ruin_threshold_pct / 100)
        
        for _ in range(simulations):
            # Shuffle trade order
            shuffled_pnls = np.random.permutation(pnls)
            
            # Simulate equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0.0
            ruined = False
            
            for pnl in shuffled_pnls:
                equity += pnl
                
                if equity <= ruin_level:
                    ruined = True
                    
                if equity > peak:
                    peak = equity
                
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            
            if ruined:
                ruin_count += 1
            if equity > initial_capital:
                profit_count += 1
        
        return MonteCarloResult(
            simulations=simulations,
            median_final_equity=np.median(final_equities),
            percentile_5=np.percentile(final_equities, 5),
            percentile_95=np.percentile(final_equities, 95),
            probability_profit=(profit_count / simulations) * 100,
            probability_ruin=(ruin_count / simulations) * 100,
            max_drawdown_median=np.median(max_drawdowns)
        )
    
    @staticmethod
    def walk_forward_split(df: pd.DataFrame, 
                           n_folds: int = 5, 
                           train_pct: float = 0.7) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data for walk-forward analysis.
        
        Returns list of (train_df, test_df) tuples.
        """
        folds = []
        fold_size = len(df) // n_folds
        
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            
            fold_data = df.iloc[start_idx:end_idx]
            split_point = int(len(fold_data) * train_pct)
            
            train = fold_data.iloc[:split_point]
            test = fold_data.iloc[split_point:]
            
            folds.append((train, test))
        
        return folds
    
    @staticmethod
    def calculate_statistical_edge(trades: List, confidence_level: float = 0.95) -> Dict:
        """
        Calculate if the trading system has a statistically significant edge.
        
        Uses t-test to determine if mean return is significantly different from zero.
        """
        if len(trades) < 30:
            return {
                'has_edge': False,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
                'reason': 'Insufficient trades for statistical analysis (need 30+)'
            }
        
        pnls = [t.pnl_dollars for t in trades]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        n = len(pnls)
        
        # Standard error
        se = std_pnl / np.sqrt(n)
        
        # T-statistic
        t_stat = mean_pnl / se if se > 0 else 0
        
        # Approximate p-value (two-tailed)
        # For large n, t-distribution ≈ normal
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        
        # Confidence interval
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        ci_lower = mean_pnl - t_critical * se
        ci_upper = mean_pnl + t_critical * se
        
        has_edge = p_value < (1 - confidence_level) and mean_pnl > 0
        
        return {
            'has_edge': has_edge,
            'p_value': p_value,
            't_statistic': t_stat,
            'confidence_interval': (ci_lower, ci_upper),
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'reason': 'Statistically significant positive edge' if has_edge else 'No significant edge detected'
        }
    
    @staticmethod
    def analyze_consecutive_patterns(trades: List) -> Dict:
        """Analyze winning/losing streaks."""
        if not trades:
            return {}
        
        max_wins = max_losses = 0
        current_wins = current_losses = 0
        win_streaks = []
        loss_streaks = []
        
        for trade in trades:
            if trade.is_winner():
                current_wins += 1
                if current_losses > 0:
                    loss_streaks.append(current_losses)
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                if current_wins > 0:
                    win_streaks.append(current_wins)
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Don't forget the last streak
        if current_wins > 0:
            win_streaks.append(current_wins)
        if current_losses > 0:
            loss_streaks.append(current_losses)
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'streak_distribution': {
                'wins': win_streaks,
                'losses': loss_streaks
            }
        }
    
    @staticmethod
    def calculate_risk_of_ruin(win_rate: float, 
                                risk_per_trade: float,
                                avg_win_r: float = 1.0,
                                avg_loss_r: float = 1.0) -> float:
        """
        Estimates probability of account ruin using simplified formula.
        
        Based on: RoR = ((1 - (W-L)) / (1 + (W-L)))^U
        Where W = win_rate, L = lose_rate, U = units to go broke
        """
        if win_rate >= 100:
            return 0.0
        if win_rate <= 0:
            return 100.0
        
        w = win_rate / 100
        l = 1 - w
        
        # Adjust for R-multiples
        edge = (w * avg_win_r) - (l * avg_loss_r)
        
        if edge <= 0:
            return 100.0  # Negative expectancy = eventual ruin
        
        # Units to ruin at given risk %
        units = 100 / risk_per_trade if risk_per_trade > 0 else 1000
        
        # Simplified RoR formula
        ratio = (1 - edge) / (1 + edge)
        
        if ratio >= 1:
            return 100.0
        
        ror = (ratio ** units) * 100
        
        return min(ror, 100.0)
