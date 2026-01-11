"""
Genesis Analytics Integration

Connects TradeAnalyzer to Genesis System for automatic trade recording and analysis.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

# Avoid circular import
if TYPE_CHECKING:
    from main_genesis import GenesisSignal

from analytics.trade_analyzer import TradeAnalyzer, TradeRecord
from analytics.trade_journal import TradeJournal, JournalEntry

logger = logging.getLogger("GenesisAnalytics")


class GenesisAnalyticsIntegration:
    """
    Integrates Trade Analytics into Genesis
    
    Automatically:
    - Records every trade
    - Tracks performance
    - Generates insights
    - Optimizes parameters
    """
    
    def __init__(self):
        self.analyzer = TradeAnalyzer(db_path="data/genesis_trades.json")
        self.journal = TradeJournal()  # Auto trade journal
        self.active_trades = {}  # trade_id -> TradeRecord
        logger.info("Genesis Analytics Integration initialized (with Trade Journal)")
    
    def on_signal(self, signal: 'GenesisSignal', current_price: float):
        """
        Called when Genesis generates a signal
        
        Records trade entry with full context
        """
        if not signal.execute:
            return
        
        trade_id = f"{signal.timestamp.strftime('%Y%m%d_%H%M%S')}_{signal.direction}"
        
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=signal.timestamp,
            symbol="GBPUSD",  # TODO: Make dynamic
            direction=signal.direction,
            entry_price=signal.entry_price or current_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            setup_type=signal.primary_signal,
            confidence=signal.confidence,
            signal_score=signal.signal_layer_score,
            agi_score=signal.agi_layer_score,
            swarm_score=signal.swarm_layer_score,
            market_regime=signal.market_regime,
            volatility_regime=signal.volatility_regime,
            reasons=signal.reasons,
            vetoes=signal.vetoes,
            hour_of_day=signal.timestamp.hour,
            day_of_week=signal.timestamp.weekday()
        )
        
        self.active_trades[trade_id] = trade
        logger.info(f"Recorded trade entry: {trade_id}")
        
        # Create journal entry
        journal_entry = JournalEntry(
            trade_id=trade_id,
            timestamp=signal.timestamp,
            direction=signal.direction,
            symbol="GBPUSD",
            entry_price=signal.entry_price or current_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            setup_type=signal.primary_signal,
            confidence=signal.confidence,
            signal_score=signal.signal_layer_score,
            agi_score=signal.agi_layer_score,
            swarm_score=signal.swarm_layer_score,
            market_regime=signal.market_regime,
            volatility_regime=signal.volatility_regime,
            trend="BULLISH" if signal.direction == "BUY" else "BEARISH",
            hour_of_day=signal.timestamp.hour,
            day_of_week=signal.timestamp.weekday(),
            reasons=signal.reasons,
            vetoes=signal.vetoes
        )
        
        self.journal.create_entry(journal_entry)
        logger.info(f"📝 Journal entry created: {trade_id}")
    
    def on_trade_close(self, trade_id: str, exit_price: float, profit_loss: float, profit_pips: float):
        """
        Called when a trade closes
        
        Completes the trade record and analyzes
        """
        if trade_id not in self.active_trades:
            logger.warning(f"Unknown trade closed: {trade_id}")
            return
        
        trade = self.active_trades[trade_id]
        trade.exit_price = exit_price
        trade.profit_loss = profit_loss
        trade.profit_pips = profit_pips
        trade.win = profit_loss > 0
        
        # Calculate duration
        duration = (datetime.now() - trade.timestamp).total_seconds() / 60
        trade.duration_minutes = int(duration)
        
        # Calculate risk/reward
        if trade.sl_price and trade.tp_price and trade.entry_price:
            risk = abs(trade.entry_price - trade.sl_price)
            reward = abs(trade.tp_price - trade.entry_price)
            trade.risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Save to database
        self.analyzer.record_trade(trade)
        
        # Update journal
        self.journal.update_entry(
            trade_id,
            exit_price=exit_price,
            profit_loss=profit_loss,
            profit_pips=profit_pips,
            win=trade.win
        )
        
        # Remove from active
        del self.active_trades[trade_id]
        
        logger.info(f"Trade closed: {trade_id} | P/L: ${profit_loss:.2f} | {'WIN' if trade.win else 'LOSS'}")
        logger.info(f"📝 Journal updated with results!")
        
        # Auto-analyze after every 10 trades
        if len(self.analyzer.trades) % 10 == 0:
            self._auto_analyze()
    
    def _auto_analyze(self):
        """Automatically generate insights after milestone trades"""
        analysis = self.analyzer.analyze_performance(days=7)
        
        if "error" not in analysis:
            logger.info("="*60)
            logger.info("AUTO-GENERATED INSIGHTS:")
            for insight in analysis.get('insights', []):
                logger.info(f"  {insight}")
            logger.info("="*60)
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        return self.analyzer.generate_report(days=1, save_path="reports/daily_report.txt")
    
    def generate_weekly_report(self) -> str:
        """Generate weekly performance report"""
        return self.analyzer.generate_report(days=7, save_path="reports/weekly_report.txt")
    
    def generate_monthly_report(self) -> str:
        """Generate monthly performance report"""
        return self.analyzer.generate_report(days=30, save_path="reports/monthly_report.txt")
    
    def get_real_time_stats(self) -> dict:
        """Get real-time performance statistics"""
        if not self.analyzer.trades:
            return {"status": "No trades yet"}
        
        recent = self.analyzer.analyze_performance(days=7)
        
        return {
            "total_trades": recent.get("total_trades", 0),
            "win_rate": f"{recent.get('win_rate', 0):.1f}%",
            "total_profit": f"${recent.get('profitability', {}).get('total_profit', 0):.2f}",
            "active_trades": len(self.active_trades),
            "status": "✅ ON TARGET" if recent.get('win_rate', 0) >= 70 else "⚠️ BELOW TARGET"
        }


# Global instance for easy access
_analytics_instance = None

def get_analytics() -> GenesisAnalyticsIntegration:
    """Get or create analytics instance"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = GenesisAnalyticsIntegration()
    return _analytics_instance


if __name__ == "__main__":
    # Test integration
    print("Genesis Analytics Integration Ready!")
    
    analytics = get_analytics()
    print("\nReal-time stats:", analytics.get_real_time_stats())
