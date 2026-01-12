"""
Genesis Advanced Backtesting Suite

Complete backtesting toolkit with:
- Multi-period testing
- Walk-forward analysis
- Monte Carlo simulation
- Parameter optimization
- Comprehensive reporting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import random
import logging

logger = logging.getLogger("BacktestSuite")


@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    simulations: int
    mean_profit: float
    std_profit: float
    median_profit: float
    worst_case: float  # 5th percentile
    best_case: float   # 95th percentile
    probability_profit: float  # % of profitable outcomes
    var_95: float  # Value at Risk (95%)
    expected_max_drawdown: float
    confidence_interval: Tuple[float, float]  # 95% CI


@dataclass  
class WalkForwardResult:
    """Results from walk-forward analysis"""
    in_sample_periods: int
    out_sample_periods: int
    in_sample_wr: float
    out_sample_wr: float
    degradation: float  # % performance loss OOS
    robustness_score: float  # 0-100
    optimized_params: Dict
    period_results: List[BacktestResult]


class AdvancedBacktester:
    """
    Advanced Backtesting Suite
    
    Features:
    - Multi-period analysis
    - Walk-forward optimization
    - Monte Carlo simulation
    - Parameter sensitivity
    - Comprehensive reporting
    """
    
    def __init__(self, trade_data: List[Dict] = None):
        self.trades = trade_data or []
        self.results_dir = Path("reports/backtest_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Advanced Backtester initialized")
    
    def load_trades_from_analyzer(self):
        """Load trades from trade analyzer"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from analytics.trade_analyzer import TradeAnalyzer
            
            analyzer = TradeAnalyzer()
            self.trades = []
            
            for t in analyzer.trades:
                self.trades.append({
                    'timestamp': t.timestamp,
                    'direction': t.direction,
                    'profit_loss': t.profit_loss or 0,
                    'profit_pips': t.profit_pips or 0,
                    'win': t.win if t.win is not None else (t.profit_loss > 0 if t.profit_loss else False),
                    'setup_type': t.setup_type,
                    'confidence': t.confidence
                })
            
            logger.info(f"Loaded {len(self.trades)} trades from analyzer")
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
    
    def run_basic_backtest(self, trades: List[Dict] = None) -> BacktestResult:
        """Run basic backtest analysis"""
        trades = trades or self.trades
        
        if not trades:
            logger.warning("No trades for backtest")
            return None
        
        # Calculate metrics
        wins = [t for t in trades if t.get('win', False)]
        losses = [t for t in trades if not t.get('win', False)]
        
        total = len(trades)
        winning = len(wins)
        losing = len(losses)
        
        profits = [t.get('profit_loss', 0) for t in trades]
        win_profits = [t.get('profit_loss', 0) for t in wins]
        loss_profits = [t.get('profit_loss', 0) for t in losses]
        
        # Calculate drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / np.maximum(running_max, 1) * 100
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate streaks
        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0
        
        for t in trades:
            if t.get('win', False):
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)
        
        # Sharpe ratio (simplified)
        returns = np.array(profits)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return BacktestResult(
            period_start=trades[0].get('timestamp', datetime.now()),
            period_end=trades[-1].get('timestamp', datetime.now()),
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=winning / total * 100 if total > 0 else 0,
            total_profit=sum(profits),
            max_drawdown=max_dd,
            profit_factor=abs(sum(win_profits)) / abs(sum(loss_profits)) if sum(loss_profits) != 0 else 0,
            sharpe_ratio=sharpe,
            avg_trade=np.mean(profits) if profits else 0,
            avg_win=np.mean(win_profits) if win_profits else 0,
            avg_loss=np.mean(loss_profits) if loss_profits else 0,
            best_trade=max(profits) if profits else 0,
            worst_trade=min(profits) if profits else 0,
            consecutive_wins=max_win_streak,
            consecutive_losses=max_loss_streak
        )
    
    def monte_carlo_simulation(self, trades: List[Dict] = None, 
                                simulations: int = 1000) -> MonteCarloResult:
        """
        Run Monte Carlo simulation
        
        Shuffles trade order to estimate range of possible outcomes
        """
        trades = trades or self.trades
        
        if len(trades) < 10:
            logger.warning("Need at least 10 trades for Monte Carlo")
            return None
        
        profits = [t.get('profit_loss', 0) for t in trades]
        
        results = []
        max_drawdowns = []
        
        for _ in range(simulations):
            # Shuffle trades
            shuffled = random.sample(profits, len(profits))
            
            # Calculate metrics
            total_profit = sum(shuffled)
            results.append(total_profit)
            
            # Calculate max drawdown for this simulation
            cumulative = np.cumsum(shuffled)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative)
            max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
            max_drawdowns.append(max_dd)
        
        results = np.array(results)
        max_drawdowns = np.array(max_drawdowns)
        
        return MonteCarloResult(
            simulations=simulations,
            mean_profit=np.mean(results),
            std_profit=np.std(results),
            median_profit=np.median(results),
            worst_case=np.percentile(results, 5),
            best_case=np.percentile(results, 95),
            probability_profit=np.sum(results > 0) / len(results) * 100,
            var_95=np.percentile(results, 5),  # 5th percentile loss
            expected_max_drawdown=np.mean(max_drawdowns),
            confidence_interval=(np.percentile(results, 2.5), np.percentile(results, 97.5))
        )
    
    def walk_forward_analysis(self, trades: List[Dict] = None,
                               in_sample_ratio: float = 0.7) -> WalkForwardResult:
        """
        Walk-forward optimization analysis
        
        Tests how in-sample optimization performs out-of-sample
        """
        trades = trades or self.trades
        
        if len(trades) < 20:
            logger.warning("Need at least 20 trades for walk-forward")
            return None
        
        # Split data
        split_idx = int(len(trades) * in_sample_ratio)
        in_sample = trades[:split_idx]
        out_sample = trades[split_idx:]
        
        # Analyze both periods
        is_result = self.run_basic_backtest(in_sample)
        os_result = self.run_basic_backtest(out_sample)
        
        if not is_result or not os_result:
            return None
        
        # Calculate degradation
        degradation = (is_result.win_rate - os_result.win_rate) / is_result.win_rate * 100 if is_result.win_rate > 0 else 0
        
        # Robustness score (100 = no degradation, 0 = complete failure)
        robustness = max(0, min(100, 100 - degradation))
        
        return WalkForwardResult(
            in_sample_periods=len(in_sample),
            out_sample_periods=len(out_sample),
            in_sample_wr=is_result.win_rate,
            out_sample_wr=os_result.win_rate,
            degradation=degradation,
            robustness_score=robustness,
            optimized_params={},  # Placeholder for actual optimization
            period_results=[is_result, os_result]
        )
    
    def generate_report(self, save: bool = True) -> str:
        """Generate comprehensive backtest report"""
        
        if not self.trades:
            return "No trades available for analysis."
        
        # Run analyses
        basic = self.run_basic_backtest()
        monte_carlo = self.monte_carlo_simulation()
        walk_forward = self.walk_forward_analysis()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           GENESIS ADVANCED BACKTEST REPORT                   ║
║           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                       ║
╚══════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════
📊 BASIC BACKTEST RESULTS
═══════════════════════════════════════════════════════════════
"""
        
        if basic:
            report += f"""
Period:             {basic.period_start.strftime('%Y-%m-%d')} to {basic.period_end.strftime('%Y-%m-%d')}
Total Trades:       {basic.total_trades}
Win Rate:           {basic.win_rate:.1f}%
Total Profit:       ${basic.total_profit:.2f}
Profit Factor:      {basic.profit_factor:.2f}
Sharpe Ratio:       {basic.sharpe_ratio:.2f}
Max Drawdown:       {basic.max_drawdown:.1f}%

Average Trade:      ${basic.avg_trade:.2f}
Average Win:        ${basic.avg_win:.2f}
Average Loss:       ${basic.avg_loss:.2f}
Best Trade:         ${basic.best_trade:.2f}
Worst Trade:        ${basic.worst_trade:.2f}

Max Win Streak:     {basic.consecutive_wins}
Max Loss Streak:    {basic.consecutive_losses}
"""
        
        report += """
═══════════════════════════════════════════════════════════════
🎲 MONTE CARLO SIMULATION (1000 runs)
═══════════════════════════════════════════════════════════════
"""
        
        if monte_carlo:
            report += f"""
Mean Profit:        ${monte_carlo.mean_profit:.2f}
Std Deviation:      ${monte_carlo.std_profit:.2f}
Median Profit:      ${monte_carlo.median_profit:.2f}

Best Case (95%):    ${monte_carlo.best_case:.2f}
Worst Case (5%):    ${monte_carlo.worst_case:.2f}
Value at Risk:      ${monte_carlo.var_95:.2f}

Probability Profit: {monte_carlo.probability_profit:.1f}%
Expected Max DD:    ${monte_carlo.expected_max_drawdown:.2f}

95% Confidence:     ${monte_carlo.confidence_interval[0]:.2f} to ${monte_carlo.confidence_interval[1]:.2f}
"""
        else:
            report += "\nInsufficient data for Monte Carlo simulation.\n"
        
        report += """
═══════════════════════════════════════════════════════════════
📈 WALK-FORWARD ANALYSIS
═══════════════════════════════════════════════════════════════
"""
        
        if walk_forward:
            report += f"""
In-Sample Trades:   {walk_forward.in_sample_periods}
Out-Sample Trades:  {walk_forward.out_sample_periods}

In-Sample WR:       {walk_forward.in_sample_wr:.1f}%
Out-Sample WR:      {walk_forward.out_sample_wr:.1f}%
Degradation:        {walk_forward.degradation:.1f}%

Robustness Score:   {walk_forward.robustness_score:.0f}/100
"""
            
            if walk_forward.robustness_score >= 80:
                report += "Status:             ✅ ROBUST - Strategy generalizes well\n"
            elif walk_forward.robustness_score >= 60:
                report += "Status:             ⚠️ MODERATE - Some overfitting present\n"
            else:
                report += "Status:             ❌ POOR - Significant overfitting detected\n"
        else:
            report += "\nInsufficient data for walk-forward analysis.\n"
        
        report += """
═══════════════════════════════════════════════════════════════
💡 RECOMMENDATIONS
═══════════════════════════════════════════════════════════════
"""
        
        recommendations = []
        
        if basic:
            if basic.win_rate >= 70:
                recommendations.append("✅ Win rate meets 70% target - ready for deployment")
            elif basic.win_rate >= 60:
                recommendations.append("⚠️ Win rate below target - consider parameter tuning")
            else:
                recommendations.append("❌ Win rate too low - strategy needs major revision")
            
            if basic.profit_factor >= 2:
                recommendations.append("✅ Excellent profit factor - strong edge")
            elif basic.profit_factor >= 1.5:
                recommendations.append("✅ Good profit factor - positive expectancy")
            else:
                recommendations.append("⚠️ Low profit factor - consider improving risk/reward")
            
            if basic.max_drawdown <= 10:
                recommendations.append("✅ Drawdown under control")
            else:
                recommendations.append("⚠️ High drawdown - reduce position sizing")
        
        if walk_forward and walk_forward.robustness_score < 70:
            recommendations.append("⚠️ Strategy may be overfit - use more conservative parameters")
        
        for rec in recommendations:
            report += f"{rec}\n"
        
        report += """
═══════════════════════════════════════════════════════════════
Generated by Genesis Advanced Backtesting Suite
"""
        
        if save:
            filepath = self.results_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {filepath}")
        
        return report


if __name__ == "__main__":
    # Test with sample data
    backtester = AdvancedBacktester()
    
    # Load from analyzer if available
    backtester.load_trades_from_analyzer()
    
    if len(backtester.trades) < 10:
        # Generate sample data
        print("Generating sample trades for demo...")
        import random
        
        for i in range(50):
            win = random.random() < 0.65  # 65% WR
            profit = random.uniform(20, 50) if win else -random.uniform(15, 35)
            
            backtester.trades.append({
                'timestamp': datetime.now() - timedelta(days=50-i),
                'direction': random.choice(['BUY', 'SELL']),
                'profit_loss': profit,
                'win': win,
                'setup_type': random.choice(['PULLBACK', 'BREAKOUT', 'REVERSAL']),
                'confidence': random.uniform(60, 90)
            })
    
    # Generate report
    print(backtester.generate_report())
