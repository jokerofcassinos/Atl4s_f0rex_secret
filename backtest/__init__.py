"""
LAPLACE DEMON - Backtest Module
════════════════════════════════

Professional-grade backtesting with:
- Realistic spread/slippage simulation
- Comprehensive metrics
- Visual analysis
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult, Trade, TradeDirection
from .charts import ChartGenerator
from .metrics import MetricsCalculator, MonteCarloResult, WalkForwardResult

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'Trade',
    'TradeDirection',
    'ChartGenerator',
    'MetricsCalculator',
    'MonteCarloResult',
    'WalkForwardResult'
]
