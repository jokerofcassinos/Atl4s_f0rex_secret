"""
Genesis Trade Analyzer - Deep Performance Analytics

Analyzes every trade in detail to identify patterns, strengths, and weaknesses.
Provides actionable insights for continuous improvement.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

logger = logging.getLogger("TradeAnalyzer")


@dataclass
class TradeRecord:
    """Complete record of a single trade with full context"""
    # Basic Info
    trade_id: str
    timestamp: datetime
    symbol: str
    direction: str  # BUY/SELL
    
    # Entry/Exit
    entry_price: float
    exit_price: Optional[float] = None
    sl_price: float = 0.0
    tp_price: float = 0.0
    
    # Results
    profit_loss: float = 0.0
    profit_pips: float = 0.0
    win: Optional[bool] = None
    
    # Context
    setup_type: str = ""
    confidence: float = 0.0
    signal_score: float = 0.0
    agi_score: float = 0.0
    swarm_score: float = 0.0
    
    # Market Conditions
    market_regime: str = "UNKNOWN"
    volatility_regime: str = "UNKNOWN"
    trend: str = "UNKNOWN"
    session: str = "UNKNOWN"
    
    # Decision Making
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    swarm_votes: Dict = field(default_factory=dict)
    
    # Timing
    hour_of_day: int = 0
    day_of_week: int = 0
    duration_minutes: int = 0
    
    # Performance Metrics
    risk_reward_ratio: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0


class TradeAnalyzer:
    """
    Deep Trade Analysis Engine
    
    Analyzes trades to identify:
    - Best performing setups
    - Optimal trading times
    - Win/loss patterns
    - Setup-specific performance
    - Risk/reward optimization
    """
    
    def __init__(self, db_path: str = "data/trade_history.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.trades: List[TradeRecord] = []
        self._load_history()
        
        logger.info(f"TradeAnalyzer initialized with {len(self.trades)} historical trades")
    
    def _load_history(self):
        """Load trade history from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    for trade_dict in data:
                        # Convert string timestamp back to datetime
                        trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'])
                        self.trades.append(TradeRecord(**trade_dict))
                logger.info(f"Loaded {len(self.trades)} trades from history")
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")
    
    def _save_history(self):
        """Save trade history to disk"""
        try:
            data = []
            for trade in self.trades:
                trade_dict = {
                    'trade_id': trade.trade_id,
                    'timestamp': trade.timestamp.isoformat(),
                    'symbol': trade.symbol,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'sl_price': trade.sl_price,
                    'tp_price': trade.tp_price,
                    'profit_loss': trade.profit_loss,
                    'profit_pips': trade.profit_pips,
                    'win': trade.win,
                    'setup_type': trade.setup_type,
                    'confidence': trade.confidence,
                    'signal_score': trade.signal_score,
                    'agi_score': trade.agi_score,
                    'swarm_score': trade.swarm_score,
                    'market_regime': trade.market_regime,
                    'volatility_regime': trade.volatility_regime,
                    'trend': trade.trend,
                    'session': trade.session,
                    'reasons': trade.reasons,
                    'vetoes': trade.vetoes,
                    'swarm_votes': trade.swarm_votes,
                    'hour_of_day': trade.hour_of_day,
                    'day_of_week': trade.day_of_week,
                    'duration_minutes': trade.duration_minutes,
                    'risk_reward_ratio': trade.risk_reward_ratio
                }
                data.append(trade_dict)
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.trades)} trades to history")
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def record_trade(self, trade: TradeRecord):
        """Record a new trade"""
        self.trades.append(trade)
        self._save_history()
        logger.info(f"Recorded trade {trade.trade_id}: {trade.direction} @ {trade.entry_price}")
    
    def analyze_performance(self, days: int = 30) -> Dict:
        """
        Comprehensive performance analysis
        
        Returns detailed metrics including:
        - Overall performance
        - Setup-specific performance
        - Time-based performance
        - Pattern analysis
        """
        if not self.trades:
            return {"error": "No trades to analyze"}
        
        # Filter recent trades
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trades if t.timestamp >= cutoff]
        
        if not recent_trades:
            return {"error": f"No trades in last {days} days"}
        
        wins = [t for t in recent_trades if t.win == True]
        losses = [t for t in recent_trades if t.win == False]
        
        total_profit = sum(t.profit_loss for t in recent_trades)
        total_pips = sum(t.profit_pips for t in recent_trades)
        
        analysis = {
            "period_days": days,
            "total_trades": len(recent_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(recent_trades) * 100 if recent_trades else 0,
            
            "profitability": {
                "total_profit": round(total_profit, 2),
                "total_pips": round(total_pips, 1),
                "avg_profit_per_trade": round(total_profit / len(recent_trades), 2) if recent_trades else 0,
                "avg_win": round(sum(t.profit_loss for t in wins) / len(wins), 2) if wins else 0,
                "avg_loss": round(sum(t.profit_loss for t in losses) / len(losses), 2) if losses else 0,
                "profit_factor": abs(sum(t.profit_loss for t in wins) / sum(t.profit_loss for t in losses)) if losses and sum(t.profit_loss for t in losses) != 0 else 0
            },
            
            "setup_analysis": self._analyze_by_setup(recent_trades),
            "time_analysis": self._analyze_by_time(recent_trades),
            "market_conditions": self._analyze_by_market(recent_trades),
            "score_analysis": self._analyze_scores(recent_trades),
            
            "best_performers": self._identify_best_trades(wins),
            "worst_performers": self._identify_worst_trades(losses),
            "insights": self._generate_insights(recent_trades)
        }
        
        return analysis
    
    def _analyze_by_setup(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by setup type"""
        setups = {}
        
        for trade in trades:
            setup = trade.setup_type or "UNKNOWN"
            if setup not in setups:
                setups[setup] = {"trades": [], "wins": 0, "total_profit": 0.0}
            
            setups[setup]["trades"].append(trade)
            if trade.win:
                setups[setup]["wins"] += 1
            setups[setup]["total_profit"] += trade.profit_loss
        
        analysis = {}
        for setup, data in setups.items():
            count = len(data["trades"])
            analysis[setup] = {
                "count": count,
                "win_rate": round(data["wins"] / count * 100, 1) if count else 0,
                "total_profit": round(data["total_profit"], 2),
                "avg_profit": round(data["total_profit"] / count, 2) if count else 0,
                "confidence": round(sum(t.confidence for t in data["trades"]) / count, 1) if count else 0
            }
        
        # Sort by win rate
        sorted_setups = sorted(analysis.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        
        return {
            "by_setup": dict(sorted_setups[:10]),  # Top 10
            "best_setup": sorted_setups[0][0] if sorted_setups else "N/A",
            "worst_setup": sorted_setups[-1][0] if sorted_setups else "N/A"
        }
    
    def _analyze_by_time(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by time of day and day of week"""
        by_hour = {}
        by_day = {}
        
        for trade in trades:
            hour = trade.hour_of_day
            day = trade.day_of_week
            
            if hour not in by_hour:
                by_hour[hour] = {"trades": 0, "wins": 0, "profit": 0.0}
            if day not in by_day:
                by_day[day] = {"trades": 0, "wins": 0, "profit": 0.0}
            
            by_hour[hour]["trades"] += 1
            by_day[day]["trades"] += 1
            
            if trade.win:
                by_hour[hour]["wins"] += 1
                by_day[day]["wins"] += 1
            
            by_hour[hour]["profit"] += trade.profit_loss
            by_day[day]["profit"] += trade.profit_loss
        
        # Calculate win rates
        for hour_data in by_hour.values():
            hour_data["win_rate"] = round(hour_data["wins"] / hour_data["trades"] * 100, 1) if hour_data["trades"] else 0
        
        for day_data in by_day.values():
            day_data["win_rate"] = round(day_data["wins"] / day_data["trades"] * 100, 1) if day_data["trades"] else 0
        
        # Find best times
        best_hour = max(by_hour.items(), key=lambda x: x[1]["win_rate"]) if by_hour else (0, {"win_rate": 0})
        best_day = max(by_day.items(), key=lambda x: x[1]["win_rate"]) if by_day else (0, {"win_rate": 0})
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        return {
            "by_hour": by_hour,
            "by_day": {days[k]: v for k, v in by_day.items()} if by_day else {},
            "best_hour": f"{best_hour[0]:02d}:00 ({best_hour[1]['win_rate']}% WR)",
            "best_day": f"{days[best_day[0]]} ({best_day[1]['win_rate']}% WR)" if best_day[0] < len(days) else "N/A"
        }
    
    def _analyze_by_market(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by market conditions"""
        by_regime = {}
        by_trend = {}
        
        for trade in trades:
            regime = trade.market_regime
            trend = trade.trend
            
            if regime not in by_regime:
                by_regime[regime] = {"wins": 0, "total": 0}
            if trend not in by_trend:
                by_trend[trend] = {"wins": 0, "total": 0}
            
            by_regime[regime]["total"] += 1
            by_trend[trend]["total"] += 1
            
            if trade.win:
                by_regime[regime]["wins"] += 1
                by_trend[trend]["wins"] += 1
        
        return {
            "by_regime": {k: f"{v['wins']}/{v['total']} ({round(v['wins']/v['total']*100,1)}%)" 
                          for k, v in by_regime.items()},
            "by_trend": {k: f"{v['wins']}/{v['total']} ({round(v['wins']/v['total']*100,1)}%)" 
                         for k, v in by_trend.items()}
        }
    
    def _analyze_scores(self, trades: List[TradeRecord]) -> Dict:
        """Analyze correlation between scores and outcomes"""
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        
        return {
            "winning_trades": {
                "avg_signal": round(sum(t.signal_score for t in wins) / len(wins), 1) if wins else 0,
                "avg_agi": round(sum(t.agi_score for t in wins) / len(wins), 1) if wins else 0,
                "avg_swarm": round(sum(t.swarm_score for t in wins) / len(wins), 1) if wins else 0,
                "avg_confidence": round(sum(t.confidence for t in wins) / len(wins), 1) if wins else 0
            },
            "losing_trades": {
                "avg_signal": round(sum(t.signal_score for t in losses) / len(losses), 1) if losses else 0,
                "avg_agi": round(sum(t.agi_score for t in losses) / len(losses), 1) if losses else 0,
                "avg_swarm": round(sum(t.swarm_score for t in losses) / len(losses), 1) if losses else 0,
                "avg_confidence": round(sum(t.confidence for t in losses) / len(losses), 1) if losses else 0
            }
        }
    
    def _identify_best_trades(self, wins: List[TradeRecord], n: int = 5) -> List[Dict]:
        """Identify best performing trades"""
        sorted_wins = sorted(wins, key=lambda t: t.profit_loss, reverse=True)[:n]
        
        return [{
            "trade_id": t.trade_id,
            "timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M"),
            "setup": t.setup_type,
            "profit": f"${t.profit_loss:.2f}",
            "pips": round(t.profit_pips, 1),
            "confidence": f"{t.confidence:.0f}%"
        } for t in sorted_wins]
    
    def _identify_worst_trades(self, losses: List[TradeRecord], n: int = 5) -> List[Dict]:
        """Identify worst performing trades"""
        sorted_losses = sorted(losses, key=lambda t: t.profit_loss)[:n]
        
        return [{
            "trade_id": t.trade_id,
            "timestamp": t.timestamp.strftime("%Y-%m-%d %H:%M"),
            "setup": t.setup_type,
            "loss": f"${t.profit_loss:.2f}",
            "pips": round(t.profit_pips, 1),
            "confidence": f"{t.confidence:.0f}%"
        } for t in sorted_losses]
    
    def _generate_insights(self, trades: List[TradeRecord]) -> List[str]:
        """Generate actionable insights from trade data"""
        insights = []
        
        if not trades:
            return ["Not enough data for insights"]
        
        # Win rate insight
        win_rate = len([t for t in trades if t.win]) / len(trades) * 100
        if win_rate >= 70:
            insights.append(f"✅ Excellent win rate ({win_rate:.1f}%) - System performing as expected")
        elif win_rate >= 60:
            insights.append(f"⚠️ Good win rate ({win_rate:.1f}%) - Room for improvement")
        else:
            insights.append(f"❌ Low win rate ({win_rate:.1f}%) - Review strategy parameters")
        
        # Setup analysis
        setup_analysis = self._analyze_by_setup(trades)
        if setup_analysis.get("by_setup"):
            best = setup_analysis["best_setup"]
            worst = setup_analysis["worst_setup"]
            insights.append(f"🎯 Best setup: {best} | Avoid: {worst}")
        
        # Time analysis
        time_analysis = self._analyze_by_time(trades)
        if time_analysis.get("best_hour"):
            insights.append(f"⏰ Best trading time: {time_analysis['best_hour']}")
        
        # Confidence correlation
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        
        if wins and losses:
            avg_conf_wins = sum(t.confidence for t in wins) / len(wins)
            avg_conf_losses = sum(t.confidence for t in losses) / len(losses)
            
            if avg_conf_wins > avg_conf_losses + 10:
                insights.append(f"💡 Higher confidence trades perform better ({avg_conf_wins:.0f}% vs {avg_conf_losses:.0f}%)")
            else:
                insights.append(f"⚠️ Confidence not correlating with success - Review decision logic")
        
        return insights
    
    def generate_report(self, days: int = 30, save_path: Optional[str] = None) -> str:
        """Generate detailed performance report"""
        analysis = self.analyze_performance(days)
        
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         GENESIS TRADE ANALYSIS REPORT                        ║
║         Period: Last {days} days                                  ║
╚══════════════════════════════════════════════════════════════╝

📊 OVERALL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:     {analysis['total_trades']}
Wins:             {analysis['wins']}
Losses:           {analysis['losses']}
Win Rate:         {analysis['win_rate']:.1f}%
Target (70%):     {'✅ MET' if analysis['win_rate'] >= 70 else '❌ BELOW TARGET'}

💰 PROFITABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Profit:     ${analysis['profitability']['total_profit']}
Total Pips:       {analysis['profitability']['total_pips']} pips
Avg/Trade:        ${analysis['profitability']['avg_profit_per_trade']}
Avg Win:          ${analysis['profitability']['avg_win']}
Avg Loss:         ${analysis['profitability']['avg_loss']}
Profit Factor:    {analysis['profitability']['profit_factor']:.2f}

🎯 TOP PERFORMING SETUPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Setup:       {analysis['setup_analysis']['best_setup']}
Worst Setup:      {analysis['setup_analysis']['worst_setup']}

⏰ TIME ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best Hour:        {analysis['time_analysis']['best_hour']}
Best Day:         {analysis['time_analysis']['best_day']}

💡 KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, insight in enumerate(analysis['insights'], 1):
            report += f"{i}. {insight}\n"
        
        report += f"""
🏆 BEST TRADES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for trade in analysis['best_performers']:
            report += f"{trade['timestamp']} | {trade['setup']:20s} | {trade['profit']:8s} | {trade['pips']} pips\n"
        
        if save_path:
            Path(save_path).write_text(report)
            logger.info(f"Report saved to {save_path}")
        
        return report


if __name__ == "__main__":
    # Test the analyzer
    analyzer = TradeAnalyzer()
    
    # Create sample trade
    sample_trade = TradeRecord(
        trade_id="TEST001",
        timestamp=datetime.now(),
        symbol="GBPUSD",
        direction="BUY",
        entry_price=1.2650,
        exit_price=1.2680,
        sl_price=1.2630,
        tp_price=1.2700,
        profit_loss=30.0,
        profit_pips=30.0,
        win=True,
        setup_type="GENESIS_PULLBACK",
        confidence=75.0,
        signal_score=70,
        agi_score=80,
        swarm_score=75,
        market_regime="TRENDING",
        hour_of_day=14,
        day_of_week=2
    )
    
    analyzer.record_trade(sample_trade)
    
    # Generate report
    print(analyzer.generate_report(days=30))
