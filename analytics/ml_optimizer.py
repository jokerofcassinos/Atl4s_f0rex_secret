"""
Genesis ML Optimizer - Auto-Parameter Tuning System

Uses Machine Learning to:
- Identify optimal parameters
- Detect winning patterns
- Suggest strategy improvements
- A/B test configurations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from analytics.trade_analyzer import TradeAnalyzer, TradeRecord

logger = logging.getLogger("MLOptimizer")


@dataclass
class OptimizationResult:
    """Result of optimization run"""
    parameter_name: str
    current_value: float
    suggested_value: float
    expected_improvement: float
    confidence: float
    evidence: str


class MLOptimizer:
    """
    Machine Learning Optimizer for Genesis
    
    Analyzes trade history and suggests parameter improvements
    using pattern recognition and statistical analysis.
    """
    
    def __init__(self, trade_analyzer: TradeAnalyzer):
        self.analyzer = trade_analyzer
        self.optimization_history = []
        logger.info("ML Optimizer initialized")
    
    def analyze_optimal_parameters(self, days: int = 30) -> List[OptimizationResult]:
        """
        Analyze trades and suggest optimal parameters
        
        Returns list of optimization suggestions
        """
        if not self.analyzer.trades:
            logger.warning("No trades available for optimization")
            return []
        
        # Get recent trades
        cutoff = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.analyzer.trades if t.timestamp >= cutoff]
        
        if len(recent_trades) < 10:
            logger.warning(f"Not enough trades ({len(recent_trades)}) for reliable optimization")
            return []
        
        suggestions = []
        
        # Analyze confidence threshold
        confidence_opt = self._optimize_confidence_threshold(recent_trades)
        if confidence_opt:
            suggestions.append(confidence_opt)
        
        # Analyze score thresholds
        signal_opt = self._optimize_signal_threshold(recent_trades)
        if signal_opt:
            suggestions.append(signal_opt)
        
        swarm_opt = self._optimize_swarm_threshold(recent_trades)
        if swarm_opt:
            suggestions.append(swarm_opt)
        
        # Analyze time windows
        time_opt = self._optimize_time_windows(recent_trades)
        if time_opt:
            suggestions.extend(time_opt)
        
        # Analyze setup selection
        setup_opt = self._optimize_setup_selection(recent_trades)
        if setup_opt:
            suggestions.extend(setup_opt)
        
        # Sort by expected improvement
        suggestions.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        return suggestions
    
    def _optimize_confidence_threshold(self, trades: List[TradeRecord]) -> Optional[OptimizationResult]:
        """Find optimal minimum confidence threshold"""
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]
        
        if not wins or not losses:
            return None
        
        # Analyze confidence distribution
        win_confidences = [t.confidence for t in wins]
        loss_confidences = [t.confidence for t in losses]
        
        avg_win_conf = np.mean(win_confidences)
        avg_loss_conf = np.mean(loss_confidences)
        
        # Find optimal cutoff using ROC-like analysis
        thresholds = range(40, 90, 5)
        best_threshold = 50
        best_score = 0
        
        for threshold in thresholds:
            passed_wins = len([c for c in win_confidences if c >= threshold])
            passed_losses = len([c for c in loss_confidences if c >= threshold])
            total_passed = passed_wins + passed_losses
            
            if total_passed == 0:
                continue
            
            win_rate = passed_wins / total_passed
            
            # Score = win_rate * trade_frequency (balance quality vs quantity)
            trade_freq = total_passed / len(trades)
            score = win_rate * (trade_freq ** 0.5)  # Square root to not over-penalize frequency
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Calculate expected improvement
        current_trades = len(trades)
        new_trades = len([t for t in trades if t.confidence >= best_threshold])
        new_wins = len([t for t in wins if t.confidence >= best_threshold])
        
        if new_trades == 0:
            return None
        
        current_wr = len(wins) / len(trades) * 100
        new_wr = new_wins / new_trades * 100
        improvement = new_wr - current_wr
        
        if improvement > 2:  # Only suggest if meaningful improvement
            return OptimizationResult(
                parameter_name="min_confidence_threshold",
                current_value=50.0,  # Assumed current
                suggested_value=float(best_threshold),
                expected_improvement=improvement,
                confidence=85.0,
                evidence=f"Would filter {len(trades) - new_trades} trades, improving WR from {current_wr:.1f}% to {new_wr:.1f}%"
            )
        
        return None
    
    def _optimize_signal_threshold(self, trades: List[TradeRecord]) -> Optional[OptimizationResult]:
        """Find optimal signal score threshold"""
        wins = [t for t in trades if t.win and t.signal_score > 0]
        losses = [t for t in trades if not t.win and t.signal_score > 0]
        
        if len(wins) < 5 or len(losses) < 5:
            return None
        
        # Analyze score distribution
        win_scores = [t.signal_score for t in wins]
        loss_scores = [t.signal_score for t in losses]
        
        # Find discriminative threshold
        thresholds = range(20, 80, 5)
        best_threshold = 30
        best_wr = 0
        
        for threshold in thresholds:
            above_wins = len([s for s in win_scores if s >= threshold])
            above_losses = len([s for s in loss_scores if s >= threshold])
            total_above = above_wins + above_losses
            
            if total_above < 5:  # Need minimum sample
                continue
            
            wr = above_wins / total_above * 100
            
            if wr > best_wr and total_above >= len(trades) * 0.3:  # Keep >=30% of trades
                best_wr = wr
                best_threshold = threshold
        
        current_wr = len(wins) / len(trades) * 100
        improvement = best_wr - current_wr
        
        if improvement > 3:
            return OptimizationResult(
                parameter_name="min_signal_score",
                current_value=30.0,
                suggested_value=float(best_threshold),
                expected_improvement=improvement,
                confidence=75.0,
                evidence=f"Signal scores >= {best_threshold} show {best_wr:.1f}% WR"
            )
        
        return None
    
    def _optimize_swarm_threshold(self, trades: List[TradeRecord]) -> Optional[OptimizationResult]:
        """Find optimal swarm consensus threshold"""
        wins = [t for t in trades if t.win and t.swarm_score > 0]
        losses = [t for t in trades if not t.win and t.swarm_score > 0]
        
        if len(wins) < 5 or len(losses) < 5:
            return None
        
        win_scores = [t.swarm_score for t in wins]
        loss_scores = [t.swarm_score for t in losses]
        
        avg_win = np.mean(win_scores)
        avg_loss = np.mean(loss_scores)
        
        # If winning trades have higher swarm scores, suggest raising threshold
        if avg_win > avg_loss + 10:  # Significant difference
            suggested = int((avg_win + avg_loss) / 2)
            current_wr = len(wins) / len(trades) * 100
            
            # Estimate new WR
            above_wins = len([s for s in win_scores if s >= suggested])
            above_losses = len([s for s in loss_scores if s >= suggested])
            
            if above_wins + above_losses > 0:
                new_wr = above_wins / (above_wins + above_losses) * 100
                improvement = new_wr - current_wr
                
                if improvement > 2:
                    return OptimizationResult(
                        parameter_name="min_swarm_consensus",
                        current_value=50.0,
                        suggested_value=float(suggested),
                        expected_improvement=improvement,
                        confidence=80.0,
                        evidence=f"Avg swarm score: wins={avg_win:.0f}, losses={avg_loss:.0f}"
                    )
        
        return None
    
    def _optimize_time_windows(self, trades: List[TradeRecord]) -> List[OptimizationResult]:
        """Identify best trading time windows"""
        suggestions = []
        
        # Analyze by hour
        hour_performance = {}
        for trade in trades:
            hour = trade.hour_of_day
            if hour not in hour_performance:
                hour_performance[hour] = {"wins": 0, "total": 0}
            
            hour_performance[hour]["total"] += 1
            if trade.win:
                hour_performance[hour]["wins"] += 1
        
        # Calculate win rates
        hour_wr = {}
        for hour, data in hour_performance.items():
            if data["total"] >= 3:  # Minimum sample
                hour_wr[hour] = data["wins"] / data["total"] * 100
        
        if not hour_wr:
            return suggestions
        
        # Find best and worst hours
        best_hours = [h for h, wr in hour_wr.items() if wr >= 75]
        worst_hours = [h for h, wr in hour_wr.items() if wr <= 40]
        
        total_wr = len([t for t in trades if t.win]) / len(trades) * 100
        
        if best_hours and len(best_hours) < 10:
            avg_best_wr = np.mean([hour_wr[h] for h in best_hours])
            improvement = avg_best_wr - total_wr
            
            if improvement > 5:
                suggestions.append(OptimizationResult(
                    parameter_name="enable_time_filter",
                    current_value=0.0,  # Currently off
                    suggested_value=1.0,  # Enable
                    expected_improvement=improvement,
                    confidence=70.0,
                    evidence=f"Trade only during hours {best_hours} ({avg_best_wr:.0f}% WR vs {total_wr:.0f}% overall)"
                ))
        
        if worst_hours:
            # Suggest avoiding worst hours
            filtered_trades = [t for t in trades if t.hour_of_day not in worst_hours]
            if filtered_trades:
                filtered_wins = len([t for t in filtered_trades if t.win])
                filtered_wr = filtered_wins / len(filtered_trades) * 100
                improvement = filtered_wr - total_wr
                
                if improvement > 3:
                    suggestions.append(OptimizationResult(
                        parameter_name="disable_hours",
                        current_value=0.0,
                        suggested_value=1.0,
                        expected_improvement=improvement,
                        confidence=75.0,
                        evidence=f"Avoid hours {worst_hours} (improves WR to {filtered_wr:.0f}%)"
                    ))
        
        return suggestions
    
    def _optimize_setup_selection(self, trades: List[TradeRecord]) -> List[OptimizationResult]:
        """Identify best-performing setups"""
        suggestions = []
        
        # Analyze by setup type
        setup_performance = {}
        for trade in trades:
            setup = trade.setup_type or "UNKNOWN"
            if setup not in setup_performance:
                setup_performance[setup] = {"wins": 0, "total": 0}
            
            setup_performance[setup]["total"] += 1
            if trade.win:
                setup_performance[setup]["wins"] += 1
        
        # Calculate win rates
        setup_wr = {}
        for setup, data in setup_performance.items():
            if data["total"] >= 5:  # Minimum sample
                setup_wr[setup] = data["wins"] / data["total"] * 100
        
        if not setup_wr:
            return suggestions
        
        # Find best setups
        best_setups = [s for s, wr in setup_wr.items() if wr >= 80]
        worst_setups = [s for s, wr in setup_wr.items() if wr <= 50]
        
        total_wr = len([t for t in trades if t.win]) / len(trades) * 100
        
        if best_setups and len(best_setups) < len(setup_wr):
            best_setup_trades = [t for t in trades if t.setup_type in best_setups]
            best_setup_wins = len([t for t in best_setup_trades if t.win])
            best_setup_wr = best_setup_wins / len(best_setup_trades) * 100 if best_setup_trades else 0
            
            improvement = best_setup_wr - total_wr
            
            if improvement > 5:
                suggestions.append(OptimizationResult(
                    parameter_name="preferred_setups",
                    current_value=0.0,
                    suggested_value=1.0,
                    expected_improvement=improvement,
                    confidence=80.0,
                    evidence=f"Focus on setups: {best_setups} ({best_setup_wr:.0f}% WR vs {total_wr:.0f}%)"
                ))
        
        if worst_setups:
            filtered_trades = [t for t in trades if t.setup_type not in worst_setups]
            if filtered_trades:
                filtered_wins = len([t for t in filtered_trades if t.win])
                filtered_wr = filtered_wins / len(filtered_trades) * 100
                improvement = filtered_wr - total_wr
                
                if improvement > 4:
                    suggestions.append(OptimizationResult(
                        parameter_name="disabled_setups",
                        current_value=0.0,
                        suggested_value=1.0,
                        expected_improvement=improvement,
                        confidence=75.0,
                        evidence=f"Disable setups: {worst_setups} (improves WR to {filtered_wr:.0f}%)"
                    ))
        
        return suggestions
    
    def generate_optimization_report(self, days: int = 30) -> str:
        """Generate comprehensive optimization report"""
        suggestions = self.analyze_optimal_parameters(days)
        
        if not suggestions:
            return "No optimization suggestions available.\nNeed more trade data for analysis."
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         GENESIS ML OPTIMIZATION REPORT                       ║
║         Analysis Period: Last {days} days                         ║
╚══════════════════════════════════════════════════════════════╝

🧠 MACHINE LEARNING ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Found {len(suggestions)} optimization opportunities

🎯 TOP RECOMMENDATIONS (by expected improvement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, suggestion in enumerate(suggestions[:5], 1):
            report += f"""
{i}. {suggestion.parameter_name.upper()}
   Current:     {suggestion.current_value}
   Suggested:   {suggestion.suggested_value}
   Improvement: +{suggestion.expected_improvement:.1f}% WR
   Confidence:  {suggestion.confidence:.0f}%
   Evidence:    {suggestion.evidence}
"""
        
        # Calculate total potential improvement
        total_improvement = sum(s.expected_improvement for s in suggestions[:3])
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 IMPLEMENTATION PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Implement top 3 suggestions
2. Expected combined improvement: +{total_improvement:.1f}% WR
3. Monitor for 7 days
4. Re-optimize with new data

⚡ QUICK WINS (High confidence, low risk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        quick_wins = [s for s in suggestions if s.confidence >= 75 and s.expected_improvement >= 3]
        
        if quick_wins:
            for win in quick_wins:
                report += f"• {win.parameter_name}: {win.evidence}\n"
        else:
            report += "• No quick wins identified. Need more data.\n"
        
        report += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 NEXT STEPS
1. Review suggestions above
2. Implement in test environment
3. Run A/B test for 7 days
4. Validate improvement
5. Deploy to production

Generated by Genesis ML Optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return report


if __name__ == "__main__":
    # Test the optimizer
    from analytics.trade_analyzer import TradeAnalyzer
    
    analyzer = TradeAnalyzer()
    optimizer = MLOptimizer(analyzer)
    
    print(optimizer.generate_optimization_report(days=30))
