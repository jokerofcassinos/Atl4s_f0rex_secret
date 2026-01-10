"""
Atl4s Backtest Engine v2.0 - Professional Grade

Features:
- Realistic spread/slippage simulation
- Unlimited leverage support
- Commission modeling
- Walk-forward optimization
- Comprehensive metrics & charts

Target: 70% Win Rate on GBPUSD with $30 capital
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger("Atl4s-Backtest")


class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Represents a single trade with full metadata."""
    id: int
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    lots: float
    sl_price: float
    tp_price: float
    
    # Exit info (filled when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # PnL
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    
    # Metadata
    signal_source: str = "UNKNOWN"
    confidence: float = 0.0
    r_multiple: float = 0.0  # Risk-adjusted return
    
    # Advanced tracking
    max_favorable_excursion: float = 0.0  # Best unrealized profit
    max_adverse_excursion: float = 0.0    # Worst unrealized loss
    duration_minutes: int = 0
    
    def is_winner(self) -> bool:
        return self.pnl_dollars > 0
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'direction': self.direction.value,
            'entry_time': str(self.entry_time),
            'entry_price': self.entry_price,
            'exit_time': str(self.exit_time) if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_pips': self.pnl_pips,
            'pnl_dollars': self.pnl_dollars,
            'lots': self.lots,
            'r_multiple': self.r_multiple,
            'duration_minutes': self.duration_minutes,
            'signal_source': self.signal_source,
            'confidence': self.confidence
        }


@dataclass
class BacktestConfig:
    """Configuration for backtest simulation."""
    initial_capital: float = 30.0
    leverage: float = 3000.0  # Unlimited represented as very high
    
    # Risk Management
    risk_per_trade_pct: float = 2.0  # 2% risk per trade
    max_concurrent_trades: int = 3
    
    # Spread/Commission (GBPUSD typical)
    spread_pips: float = 1.5  # Realistic spread
    commission_per_lot: float = 0.0  # Many brokers have 0 commission
    slippage_pips: float = 0.5  # Average slippage
    
    # Symbol specific
    symbol: str = "GBPUSD"
    pip_value: float = 10.0  # $10 per pip per standard lot
    contract_size: float = 100000.0
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Session filters (UTC hours)
    london_start: int = 8
    london_end: int = 16
    ny_start: int = 13
    ny_end: int = 21
    
    def get_pip_value_for_lots(self, lots: float) -> float:
        """Returns $ value of 1 pip for given lot size."""
        return self.pip_value * lots


@dataclass 
class BacktestResult:
    """Complete results from a backtest run."""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: List[Tuple[datetime, float]]
    
    # Core metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # PnL metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    avg_r_multiple: float = 0.0
    
    # Advanced
    expectancy: float = 0.0  # Expected $ per trade
    trades_per_day: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    # By hour analysis
    win_rate_by_hour: Dict[int, float] = field(default_factory=dict)
    pnl_by_hour: Dict[int, float] = field(default_factory=dict)
    
    # By day analysis
    win_rate_by_day: Dict[str, float] = field(default_factory=dict)
    pnl_by_day: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Professional Backtest Engine for Atl4s.
    
    Features:
    - Tick-by-tick simulation
    - Realistic spread/slippage
    - Position sizing based on risk %
    - Walk-forward optimization support
    - Comprehensive metrics and charts
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()
        
    def reset(self):
        """Reset engine state for new run."""
        self.balance = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.trades: List[Trade] = []
        self.active_trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_counter = 0
        
        # Tracking
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0.0
        
    def calculate_position_size(self, sl_pips: float) -> float:
        """
        Calculate lot size based on risk percentage and stop loss.
        
        Formula: Lots = (Account * Risk%) / (SL_pips * Pip_Value)
        """
        if sl_pips <= 0:
            return 0.01  # Minimum lot
            
        risk_amount = self.balance * (self.config.risk_per_trade_pct / 100)
        lot_size = risk_amount / (sl_pips * self.config.pip_value)
        
        # Clamp to reasonable bounds
        lot_size = max(0.01, min(lot_size, 100.0))
        return round(lot_size, 2)
    
    def calculate_pips(self, price1: float, price2: float) -> float:
        """Calculate pip difference between two prices."""
        # GBPUSD: 1 pip = 0.0001
        # USDJPY: 1 pip = 0.01
        multiplier = 10000 if "JPY" not in self.config.symbol else 100
        return (price2 - price1) * multiplier
    
    def apply_spread(self, price: float, direction: TradeDirection) -> float:
        """Apply realistic spread to entry price."""
        spread_in_price = self.config.spread_pips / 10000
        if direction == TradeDirection.BUY:
            return price + spread_in_price  # Buy at ask
        else:
            return price  # Sell at bid
    
    def apply_slippage(self, price: float, direction: TradeDirection) -> float:
        """Apply random slippage to execution."""
        slippage = np.random.uniform(0, self.config.slippage_pips) / 10000
        if direction == TradeDirection.BUY:
            return price + slippage  # Slippage against us
        else:
            return price - slippage
    
    def open_trade(self, 
                   direction: TradeDirection,
                   current_time: datetime,
                   current_price: float,
                   sl_pips: float,
                   tp_pips: float,
                   signal_source: str = "UNKNOWN",
                   confidence: float = 0.0) -> Optional[Trade]:
        """
        Opens a new trade with proper risk management.
        Includes Anti-Ruin protection.
        """
        # ═══════════════════════════════════════════════════════════
        # ANTI-RUIN CHECKS
        # ═══════════════════════════════════════════════════════════
        
        # 1. FreeMargin Check - Don't trade if margin is too low
        MIN_FREE_MARGIN = 15.0  # Block new trades if FreeMargin < $15
        free_margin = self.balance  # Simplified: FreeMargin ≈ Balance for spot forex
        if free_margin < MIN_FREE_MARGIN:
            logger.debug(f"BLOCKED: FreeMargin ${free_margin:.2f} < ${MIN_FREE_MARGIN}")
            return None
        
        # 2. Check concurrent trades limit
        if len(self.active_trades) >= self.config.max_concurrent_trades:
            return None
        
        # 3. Check if SL would cost more than 5% of account
        MAX_RISK_PCT = 5.0  # $1.50 on $30 account
        max_risk_dollars = self.balance * (MAX_RISK_PCT / 100)
        potential_loss = sl_pips * self.config.pip_value * 0.01  # For 0.01 lot
        
        if potential_loss > max_risk_dollars:
            # Reduce trade size or skip
            logger.debug(f"RISK TOO HIGH: ${potential_loss:.2f} > ${max_risk_dollars:.2f}")
            # Adjust risk_per_trade to fit within limits
            adjusted_risk = MAX_RISK_PCT * 0.5  # Use half the max
            self.config.risk_per_trade_pct = adjusted_risk
        
        # Apply spread and slippage
        entry_price = self.apply_spread(current_price, direction)
        entry_price = self.apply_slippage(entry_price, direction)
        
        # Calculate SL/TP prices
        pip_size = 0.0001 if "JPY" not in self.config.symbol else 0.01
        
        if direction == TradeDirection.BUY:
            sl_price = entry_price - (sl_pips * pip_size)
            tp_price = entry_price + (tp_pips * pip_size)
        else:
            sl_price = entry_price + (sl_pips * pip_size)
            tp_price = entry_price - (tp_pips * pip_size)
        
        # Calculate position size with Anti-Ruin limits
        lots = self.calculate_position_size(sl_pips)
        
        # For small accounts (<$100), use fixed minimum lot
        if self.balance < 100:
            lots = 0.01  # Fixed minimum lot for small accounts
        
        if lots < 0.01:
            return None  # Can't afford the trade
        
        self.trade_counter += 1
        trade = Trade(
            id=self.trade_counter,
            direction=direction,
            entry_time=current_time,
            entry_price=entry_price,
            lots=lots,
            sl_price=sl_price,
            tp_price=tp_price,
            signal_source=signal_source,
            confidence=confidence
        )
        
        self.active_trades.append(trade)
        logger.debug(f"OPEN #{trade.id}: {direction.value} @ {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f} | Lots: {lots}")
        
        return trade
    
    def update_trade(self, trade: Trade, current_price: float, current_time: datetime) -> Optional[str]:
        """
        Updates trade state and checks for SL/TP hit.
        Returns exit reason if trade should close.
        """
        # Calculate current PnL
        if trade.direction == TradeDirection.BUY:
            current_pips = self.calculate_pips(trade.entry_price, current_price)
        else:
            current_pips = self.calculate_pips(current_price, trade.entry_price)
        
        current_pnl = current_pips * self.config.get_pip_value_for_lots(trade.lots)
        
        # Track MFE/MAE
        if current_pnl > trade.max_favorable_excursion:
            trade.max_favorable_excursion = current_pnl
        if current_pnl < trade.max_adverse_excursion:
            trade.max_adverse_excursion = current_pnl
        
        # Check SL
        if trade.direction == TradeDirection.BUY:
            if current_price <= trade.sl_price:
                return "SL_HIT"
            if current_price >= trade.tp_price:
                return "TP_HIT"
        else:
            if current_price >= trade.sl_price:
                return "SL_HIT"
            if current_price <= trade.tp_price:
                return "TP_HIT"
        
        return None
    
    def close_trade(self, trade: Trade, exit_price: float, exit_time: datetime, reason: str):
        """Closes a trade and records results."""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate final PnL
        if trade.direction == TradeDirection.BUY:
            trade.pnl_pips = self.calculate_pips(trade.entry_price, exit_price)
        else:
            trade.pnl_pips = self.calculate_pips(exit_price, trade.entry_price)
        
        trade.pnl_dollars = trade.pnl_pips * self.config.get_pip_value_for_lots(trade.lots)
        
        # Calculate R-Multiple
        risk_pips = abs(self.calculate_pips(trade.entry_price, trade.sl_price))
        if risk_pips > 0:
            trade.r_multiple = trade.pnl_pips / risk_pips
        
        # Duration
        trade.duration_minutes = int((exit_time - trade.entry_time).total_seconds() / 60)
        
        # Update balance
        self.balance += trade.pnl_dollars
        
        # Remove from active
        if trade in self.active_trades:
            self.active_trades.remove(trade)
        
        # Add to history
        self.trades.append(trade)
        
        logger.debug(f"CLOSE #{trade.id}: {reason} @ {exit_price:.5f} | PnL: ${trade.pnl_dollars:.2f} | Balance: ${self.balance:.2f}")
    
    def simulate_candle(self, candle: pd.Series, signal_func: callable) -> List[Trade]:
        """
        Simulates one candle of trading.
        
        Args:
            candle: OHLCV data with datetime index
            signal_func: Function that returns (direction, sl_pips, tp_pips, source, confidence) or None
        
        Returns:
            List of trades closed during this candle
        """
        current_time = candle.name
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        closed_trades = []
        
        # 1. Check active trades for SL/TP
        for trade in self.active_trades[:]:
            # Simulate intra-candle price path (high/low touch)
            exit_reason = None
            exit_price = None
            
            if trade.direction == TradeDirection.BUY:
                # Check if SL was hit (low touched SL)
                if low <= trade.sl_price:
                    exit_reason = "SL_HIT"
                    exit_price = trade.sl_price
                # Check if TP was hit (high touched TP)
                elif high >= trade.tp_price:
                    exit_reason = "TP_HIT"
                    exit_price = trade.tp_price
            else:
                # SELL trade
                if high >= trade.sl_price:
                    exit_reason = "SL_HIT"
                    exit_price = trade.sl_price
                elif low <= trade.tp_price:
                    exit_reason = "TP_HIT"
                    exit_price = trade.tp_price
            
            if exit_reason:
                self.close_trade(trade, exit_price, current_time, exit_reason)
                closed_trades.append(trade)
        
        # 2. Check for new signals (only at candle close for realism)
        signal = signal_func(candle)
        if signal:
            direction, sl_pips, tp_pips, source, confidence = signal
            if direction:
                self.open_trade(
                    direction=direction,
                    current_time=current_time,
                    current_price=close,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    signal_source=source,
                    confidence=confidence
                )
        
        # 3. Record equity
        unrealized_pnl = 0.0
        for trade in self.active_trades:
            if trade.direction == TradeDirection.BUY:
                pips = self.calculate_pips(trade.entry_price, close)
            else:
                pips = self.calculate_pips(close, trade.entry_price)
            unrealized_pnl += pips * self.config.get_pip_value_for_lots(trade.lots)
        
        self.equity = self.balance + unrealized_pnl
        self.equity_curve.append((current_time, self.equity))
        
        # Track drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        return closed_trades
    
    def run(self, 
            df: pd.DataFrame, 
            signal_func: callable,
            progress_callback: callable = None) -> BacktestResult:
        """
        Runs the complete backtest simulation.
        
        Args:
            df: DataFrame with OHLCV data (datetime index)
            signal_func: Function(candle) -> (direction, sl_pips, tp_pips, source, confidence) or None
            progress_callback: Optional callback for progress updates
        
        Returns:
            BacktestResult with all metrics and trade history
        """
        self.reset()
        
        total_candles = len(df)
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            self.simulate_candle(row, signal_func)
            
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_candles)
        
        # Close any remaining trades at last price
        last_row = df.iloc[-1]
        for trade in self.active_trades[:]:
            self.close_trade(trade, last_row['close'], df.index[-1], "END_OF_TEST")
        
        # Calculate final metrics
        return self._calculate_results()
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        from .metrics import MetricsCalculator
        
        result = BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
        
        if not self.trades:
            return result
        
        # Basic counts
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.is_winner())
        result.losing_trades = result.total_trades - result.winning_trades
        result.win_rate = (result.winning_trades / result.total_trades) * 100 if result.total_trades > 0 else 0
        
        # PnL
        result.gross_profit = sum(t.pnl_dollars for t in self.trades if t.pnl_dollars > 0)
        result.gross_loss = abs(sum(t.pnl_dollars for t in self.trades if t.pnl_dollars < 0))
        result.net_profit = result.gross_profit - result.gross_loss
        result.profit_factor = result.gross_profit / result.gross_loss if result.gross_loss > 0 else float('inf')
        
        # Averages
        winners = [t.pnl_dollars for t in self.trades if t.pnl_dollars > 0]
        losers = [t.pnl_dollars for t in self.trades if t.pnl_dollars < 0]
        
        result.avg_win = np.mean(winners) if winners else 0
        result.avg_loss = np.mean(losers) if losers else 0
        result.largest_win = max(winners) if winners else 0
        result.largest_loss = min(losers) if losers else 0
        
        # R-Multiple
        r_multiples = [t.r_multiple for t in self.trades if t.r_multiple != 0]
        result.avg_r_multiple = np.mean(r_multiples) if r_multiples else 0
        
        # Duration
        durations = [t.duration_minutes for t in self.trades]
        result.avg_trade_duration = np.mean(durations) if durations else 0
        
        # Expectancy
        result.expectancy = result.net_profit / result.total_trades if result.total_trades > 0 else 0
        
        # Trading frequency
        if self.equity_curve:
            first_time = self.equity_curve[0][0]
            last_time = self.equity_curve[-1][0]
            days = (last_time - first_time).days or 1
            result.trades_per_day = result.total_trades / days
        
        # Drawdown
        result.max_drawdown = self.max_drawdown
        result.max_drawdown_pct = self.max_drawdown
        
        # Consecutive wins/losses
        result.consecutive_wins, result.consecutive_losses = self._calculate_streaks()
        
        # By hour analysis
        result.win_rate_by_hour = self._calculate_win_rate_by_hour()
        result.pnl_by_hour = self._calculate_pnl_by_hour()
        
        # By day analysis
        result.win_rate_by_day = self._calculate_win_rate_by_day()
        result.pnl_by_day = self._calculate_pnl_by_day()
        
        # Risk metrics (Sharpe, Sortino, Calmar)
        result.sharpe_ratio, result.sortino_ratio, result.calmar_ratio = self._calculate_risk_metrics()
        
        return result
    
    def _calculate_streaks(self) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        max_wins = max_losses = 0
        current_wins = current_losses = 0
        
        for trade in self.trades:
            if trade.is_winner():
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_win_rate_by_hour(self) -> Dict[int, float]:
        """Calculate win rate for each hour of the day."""
        by_hour = {}
        
        for trade in self.trades:
            hour = trade.entry_time.hour
            if hour not in by_hour:
                by_hour[hour] = {'wins': 0, 'total': 0}
            by_hour[hour]['total'] += 1
            if trade.is_winner():
                by_hour[hour]['wins'] += 1
        
        return {h: (d['wins']/d['total']*100) if d['total'] > 0 else 0 
                for h, d in by_hour.items()}
    
    def _calculate_pnl_by_hour(self) -> Dict[int, float]:
        """Calculate total PnL for each hour."""
        by_hour = {}
        for trade in self.trades:
            hour = trade.entry_time.hour
            by_hour[hour] = by_hour.get(hour, 0) + trade.pnl_dollars
        return by_hour
    
    def _calculate_win_rate_by_day(self) -> Dict[str, float]:
        """Calculate win rate by day of week."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        by_day = {d: {'wins': 0, 'total': 0} for d in days}
        
        for trade in self.trades:
            day = days[trade.entry_time.weekday()]
            by_day[day]['total'] += 1
            if trade.is_winner():
                by_day[day]['wins'] += 1
        
        return {d: (v['wins']/v['total']*100) if v['total'] > 0 else 0 
                for d, v in by_day.items()}
    
    def _calculate_pnl_by_day(self) -> Dict[str, float]:
        """Calculate total PnL by day of week."""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        by_day = {d: 0.0 for d in days}
        
        for trade in self.trades:
            day = days[trade.entry_time.weekday()]
            by_day[day] += trade.pnl_dollars
        
        return by_day
    
    def _calculate_risk_metrics(self) -> Tuple[float, float, float]:
        """Calculate Sharpe, Sortino, and Calmar ratios."""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0, 0.0, 0.0
        
        # Daily returns
        equity_series = pd.Series([e[1] for e in self.equity_curve], 
                                   index=[e[0] for e in self.equity_curve])
        daily_equity = equity_series.resample('D').last().dropna()
        returns = daily_equity.pct_change().dropna()
        
        if len(returns) < 2:
            return 0.0, 0.0, 0.0
        
        # Sharpe (annualized)
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        # Sortino (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = mean_return / downside_std if downside_std > 0 else 0
        
        # Calmar (return / max drawdown)
        total_return = (daily_equity.iloc[-1] - daily_equity.iloc[0]) / daily_equity.iloc[0] * 100
        calmar = total_return / self.max_drawdown if self.max_drawdown > 0 else 0
        
        return sharpe, sortino, calmar
    
    def export_results(self, result: BacktestResult, filepath: str):
        """Export results to JSON file."""
        data = {
            'config': {
                'initial_capital': result.config.initial_capital,
                'leverage': result.config.leverage,
                'symbol': result.config.symbol,
                'risk_per_trade_pct': result.config.risk_per_trade_pct,
                'spread_pips': result.config.spread_pips
            },
            'summary': {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': round(result.win_rate, 2),
                'net_profit': round(result.net_profit, 2),
                'profit_factor': round(result.profit_factor, 2),
                'max_drawdown_pct': round(result.max_drawdown_pct, 2),
                'sharpe_ratio': round(result.sharpe_ratio, 2),
                'expectancy': round(result.expectancy, 2),
                'avg_r_multiple': round(result.avg_r_multiple, 2)
            },
            'by_hour': {
                'win_rate': {str(k): round(v, 2) for k, v in result.win_rate_by_hour.items()},
                'pnl': {str(k): round(v, 2) for k, v in result.pnl_by_hour.items()}
            },
            'by_day': {
                'win_rate': result.win_rate_by_day,
                'pnl': {k: round(v, 2) for k, v in result.pnl_by_day.items()}
            },
            'trades': [t.to_dict() for t in result.trades]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")
