"""
Genesis Risk Monitor

Real-time risk management and monitoring system:
- Drawdown alerts
- Position size validation
- Daily loss limits
- Risk metric tracking
- Real-time alerts
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import json

logger = logging.getLogger("RiskMonitor")


@dataclass
class RiskAlert:
    """Risk alert data"""
    alert_id: str
    timestamp: datetime
    level: str  # INFO, WARNING, CRITICAL
    category: str  # DRAWDOWN, POSITION, DAILY_LOSS, EXPOSURE
    message: str
    value: float
    threshold: float
    action_required: bool = False


@dataclass
class RiskMetrics:
    """Current risk metrics"""
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    open_positions: int = 0
    total_exposure: float = 0.0
    win_streak: int = 0
    loss_streak: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class RiskMonitor:
    """
    Real-time Risk Management System
    
    Monitors:
    - Drawdown levels
    - Position sizes
    - Daily P/L limits
    - Exposure levels
    - Win/loss streaks
    """
    
    def __init__(self, config: Dict = None):
        # Default configuration
        self.config = config or {
            'max_daily_loss': 500.0,       # Max daily loss
            'max_daily_trades': 12,         # Max trades per day
            'max_drawdown_warning': 5.0,    # Warning at 5% drawdown
            'max_drawdown_critical': 10.0,  # Critical at 10% drawdown
            'max_position_size': 1.0,       # Max lot size
            'max_open_positions': 3,        # Max concurrent positions
            'max_loss_streak': 4,           # Alert after 4 consecutive losses
            'starting_balance': 10000.0,    # Account balance for % calculations
        }
        
        self.metrics = RiskMetrics()
        self.alerts: List[RiskAlert] = []
        self.alert_history_file = Path("reports/risk_alerts.json")
        self.alert_history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_history()
        logger.info("Risk Monitor initialized")
    
    def _load_history(self):
        """Load alert history"""
        if self.alert_history_file.exists():
            try:
                with open(self.alert_history_file, 'r') as f:
                    data = json.load(f)
                    # Just load count for now
                    logger.info(f"Loaded {len(data)} historical alerts")
            except Exception as e:
                logger.warning(f"Could not load alert history: {e}")
    
    def _save_alert(self, alert: RiskAlert):
        """Save alert to history"""
        try:
            history = []
            if self.alert_history_file.exists():
                with open(self.alert_history_file, 'r') as f:
                    history = json.load(f)
            
            history.append({
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'category': alert.category,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'action_required': alert.action_required
            })
            
            # Keep last 100 alerts
            history = history[-100:]
            
            with open(self.alert_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    def check_all(self) -> List[RiskAlert]:
        """Run all risk checks"""
        alerts = []
        
        alerts.extend(self._check_drawdown())
        alerts.extend(self._check_daily_loss())
        alerts.extend(self._check_daily_trades())
        alerts.extend(self._check_loss_streak())
        alerts.extend(self._check_exposure())
        
        for alert in alerts:
            self.alerts.append(alert)
            self._save_alert(alert)
            self._log_alert(alert)
        
        return alerts
    
    def _check_drawdown(self) -> List[RiskAlert]:
        """Check drawdown levels"""
        alerts = []
        dd = self.metrics.current_drawdown
        
        if dd >= self.config['max_drawdown_critical']:
            alerts.append(RiskAlert(
                alert_id=f"DD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="CRITICAL",
                category="DRAWDOWN",
                message=f"CRITICAL: Drawdown at {dd:.1f}% - STOP TRADING!",
                value=dd,
                threshold=self.config['max_drawdown_critical'],
                action_required=True
            ))
        elif dd >= self.config['max_drawdown_warning']:
            alerts.append(RiskAlert(
                alert_id=f"DD_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="WARNING",
                category="DRAWDOWN",
                message=f"WARNING: Drawdown at {dd:.1f}% - Reduce position sizes",
                value=dd,
                threshold=self.config['max_drawdown_warning'],
                action_required=False
            ))
        
        return alerts
    
    def _check_daily_loss(self) -> List[RiskAlert]:
        """Check daily loss limits"""
        alerts = []
        daily_loss = -self.metrics.daily_pnl  # Negative PnL = loss
        
        if daily_loss >= self.config['max_daily_loss']:
            alerts.append(RiskAlert(
                alert_id=f"DL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="CRITICAL",
                category="DAILY_LOSS",
                message=f"CRITICAL: Daily loss ${daily_loss:.2f} exceeds limit - STOP TRADING TODAY!",
                value=daily_loss,
                threshold=self.config['max_daily_loss'],
                action_required=True
            ))
        elif daily_loss >= self.config['max_daily_loss'] * 0.8:
            alerts.append(RiskAlert(
                alert_id=f"DL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="WARNING",
                category="DAILY_LOSS",
                message=f"WARNING: Daily loss ${daily_loss:.2f} at 80% of limit",
                value=daily_loss,
                threshold=self.config['max_daily_loss'] * 0.8,
                action_required=False
            ))
        
        return alerts
    
    def _check_daily_trades(self) -> List[RiskAlert]:
        """Check daily trade count"""
        alerts = []
        trades = self.metrics.daily_trades
        max_trades = self.config['max_daily_trades']
        
        if trades >= max_trades:
            alerts.append(RiskAlert(
                alert_id=f"DT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="CRITICAL",
                category="DAILY_TRADES",
                message=f"CRITICAL: Daily trade limit ({max_trades}) reached - NO MORE TRADES TODAY!",
                value=trades,
                threshold=max_trades,
                action_required=True
            ))
        elif trades >= max_trades - 2:
            alerts.append(RiskAlert(
                alert_id=f"DT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="WARNING",
                category="DAILY_TRADES",
                message=f"WARNING: {trades}/{max_trades} daily trades used - {max_trades - trades} remaining",
                value=trades,
                threshold=max_trades,
                action_required=False
            ))
        
        return alerts
    
    def _check_loss_streak(self) -> List[RiskAlert]:
        """Check consecutive loss streak"""
        alerts = []
        streak = self.metrics.loss_streak
        max_streak = self.config['max_loss_streak']
        
        if streak >= max_streak:
            alerts.append(RiskAlert(
                alert_id=f"LS_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="CRITICAL",
                category="LOSS_STREAK",
                message=f"CRITICAL: {streak} consecutive losses - STOP AND REVIEW STRATEGY!",
                value=streak,
                threshold=max_streak,
                action_required=True
            ))
        elif streak >= max_streak - 1:
            alerts.append(RiskAlert(
                alert_id=f"LS_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="WARNING",
                category="LOSS_STREAK",
                message=f"WARNING: {streak} consecutive losses - Be cautious",
                value=streak,
                threshold=max_streak,
                action_required=False
            ))
        
        return alerts
    
    def _check_exposure(self) -> List[RiskAlert]:
        """Check total exposure"""
        alerts = []
        positions = self.metrics.open_positions
        max_positions = self.config['max_open_positions']
        
        if positions >= max_positions:
            alerts.append(RiskAlert(
                alert_id=f"EX_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now(),
                level="WARNING",
                category="EXPOSURE",
                message=f"WARNING: Max positions ({max_positions}) reached - Close before opening new",
                value=positions,
                threshold=max_positions,
                action_required=True
            ))
        
        return alerts
    
    def _log_alert(self, alert: RiskAlert):
        """Log alert to console"""
        if alert.level == "CRITICAL":
            logger.critical(f"🚨 {alert.message}")
        elif alert.level == "WARNING":
            logger.warning(f"⚠️ {alert.message}")
        else:
            logger.info(f"ℹ️ {alert.message}")
    
    def update_metrics(self, 
                       daily_pnl: float = None,
                       daily_trades: int = None,
                       open_positions: int = None,
                       win: bool = None,
                       drawdown: float = None):
        """Update risk metrics"""
        if daily_pnl is not None:
            self.metrics.daily_pnl = daily_pnl
        
        if daily_trades is not None:
            self.metrics.daily_trades = daily_trades
        
        if open_positions is not None:
            self.metrics.open_positions = open_positions
        
        if win is not None:
            if win:
                self.metrics.win_streak += 1
                self.metrics.loss_streak = 0
            else:
                self.metrics.loss_streak += 1
                self.metrics.win_streak = 0
        
        if drawdown is not None:
            self.metrics.current_drawdown = drawdown
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)
        
        self.metrics.last_updated = datetime.now()
    
    def validate_trade(self, lot_size: float, direction: str) -> tuple[bool, List[str]]:
        """
        Validate if a trade should be allowed
        
        Returns: (allowed, reasons)
        """
        allowed = True
        reasons = []
        
        # Check position size
        if lot_size > self.config['max_position_size']:
            allowed = False
            reasons.append(f"Position size {lot_size} exceeds max {self.config['max_position_size']}")
        
        # Check daily trades
        if self.metrics.daily_trades >= self.config['max_daily_trades']:
            allowed = False
            reasons.append(f"Daily trade limit reached ({self.config['max_daily_trades']})")
        
        # Check daily loss
        if -self.metrics.daily_pnl >= self.config['max_daily_loss']:
            allowed = False
            reasons.append(f"Daily loss limit reached (${self.config['max_daily_loss']})")
        
        # Check drawdown
        if self.metrics.current_drawdown >= self.config['max_drawdown_critical']:
            allowed = False
            reasons.append(f"Critical drawdown ({self.metrics.current_drawdown:.1f}%)")
        
        # Check open positions
        if self.metrics.open_positions >= self.config['max_open_positions']:
            allowed = False
            reasons.append(f"Max open positions reached ({self.config['max_open_positions']})")
        
        # Check loss streak
        if self.metrics.loss_streak >= self.config['max_loss_streak']:
            allowed = False
            reasons.append(f"Loss streak limit reached ({self.config['max_loss_streak']})")
        
        return allowed, reasons
    
    def generate_report(self) -> str:
        """Generate risk status report"""
        m = self.metrics
        c = self.config
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║               GENESIS RISK MONITOR - STATUS                  ║
╚══════════════════════════════════════════════════════════════╝

📊 CURRENT METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Drawdown:           {m.current_drawdown:.1f}% / {c['max_drawdown_critical']:.1f}% (max)
Daily P/L:          ${m.daily_pnl:.2f} / -${c['max_daily_loss']:.2f} (limit)
Daily Trades:       {m.daily_trades} / {c['max_daily_trades']} (max)
Open Positions:     {m.open_positions} / {c['max_open_positions']} (max)
Win Streak:         {m.win_streak}
Loss Streak:        {m.loss_streak} / {c['max_loss_streak']} (max)

🚦 STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Status checks
        issues = []
        if m.current_drawdown >= c['max_drawdown_critical']:
            issues.append("🚨 CRITICAL DRAWDOWN - STOP TRADING")
        elif m.current_drawdown >= c['max_drawdown_warning']:
            issues.append("⚠️ Drawdown warning - Reduce size")
        
        if -m.daily_pnl >= c['max_daily_loss']:
            issues.append("🚨 DAILY LOSS LIMIT - STOP TODAY")
        
        if m.daily_trades >= c['max_daily_trades']:
            issues.append("🚨 TRADE LIMIT - NO MORE TODAY")
        
        if m.loss_streak >= c['max_loss_streak']:
            issues.append("🚨 LOSS STREAK - REVIEW STRATEGY")
        
        if issues:
            for issue in issues:
                report += f"{issue}\n"
        else:
            report += "✅ All systems nominal - Trading allowed\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Last Updated: {m.last_updated.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def reset_daily(self):
        """Reset daily metrics (call at start of new trading day)"""
        self.metrics.daily_pnl = 0.0
        self.metrics.daily_trades = 0
        self.metrics.last_updated = datetime.now()
        logger.info("Daily metrics reset")


if __name__ == "__main__":
    # Test the risk monitor
    monitor = RiskMonitor()
    
    # Simulate some activity
    monitor.update_metrics(daily_pnl=-150.0, daily_trades=5, open_positions=2)
    
    # Run checks
    alerts = monitor.check_all()
    print(f"\nAlerts: {len(alerts)}")
    
    # Validate a trade
    allowed, reasons = monitor.validate_trade(0.5, "BUY")
    print(f"\nTrade allowed: {allowed}")
    if reasons:
        print(f"Reasons: {reasons}")
    
    # Generate report
    print(monitor.generate_report())
    
    # Simulate losses
    for i in range(5):
        monitor.update_metrics(win=False)
    
    # Check again
    alerts = monitor.check_all()
    print(f"\nAfter losses - Alerts: {len(alerts)}")
    print(monitor.generate_report())
