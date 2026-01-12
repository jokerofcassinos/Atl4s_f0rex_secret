"""
Genesis Analytics Suite

Complete analytics package including:
- Trade Analyzer: Deep performance analysis
- ML Optimizer: Parameter optimization
- Trade Journal: Auto documentation
- Dashboard Generator: Visual reporting
- Risk Monitor: Real-time risk management
- Advanced Backtester: Monte Carlo & walk-forward
- Telegram Notifier: Real-time alerts
"""

from analytics.trade_analyzer import TradeAnalyzer, TradeRecord
from analytics.ml_optimizer import MLOptimizer
from analytics.genesis_analytics import GenesisAnalyticsIntegration, get_analytics
from analytics.trade_journal import TradeJournal, JournalEntry
from analytics.dashboard_generator import DashboardGenerator
from analytics.risk_monitor import RiskMonitor, RiskAlert, RiskMetrics
from analytics.advanced_backtester import AdvancedBacktester, BacktestResult, MonteCarloResult
from analytics.telegram_notifier import TelegramNotifier, get_notifier

__all__ = [
    # Core Analytics
    'TradeAnalyzer',
    'TradeRecord',
    'GenesisAnalyticsIntegration',
    'get_analytics',
    
    # ML Optimization
    'MLOptimizer',
    
    # Trade Journal
    'TradeJournal',
    'JournalEntry',
    
    # Dashboard
    'DashboardGenerator',
    
    # Risk Management
    'RiskMonitor',
    'RiskAlert',
    'RiskMetrics',
    
    # Advanced Backtesting
    'AdvancedBacktester',
    'BacktestResult',
    'MonteCarloResult',
    
    # Notifications
    'TelegramNotifier',
    'get_notifier',
]

__version__ = "2.0.0"
__author__ = "Genesis Trading System"
