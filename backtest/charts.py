"""
Atl4s Chart Generator v2.0

Generates professional-grade backtest analysis charts:
- Equity curve
- Drawdown chart
- Trade distribution
- Win rate by hour/day
- Monthly returns heatmap
- R-Multiple distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import os

# Style configuration
plt.style.use('dark_background')
COLORS = {
    'profit': '#00ff88',
    'loss': '#ff4444',
    'neutral': '#888888',
    'equity': '#00d4ff',
    'drawdown': '#ff6b6b',
    'accent': '#ffd700',
    'bg': '#0a0a0a',
    'grid': '#222222'
}


class ChartGenerator:
    """Generates all backtest analysis charts."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_all(self, result, prefix: str = "backtest") -> List[str]:
        """
        Generates all charts and returns list of file paths.
        """
        charts = []
        
        # Core charts
        charts.append(self.plot_equity_curve(result, f"{prefix}_equity.png"))
        charts.append(self.plot_drawdown(result, f"{prefix}_drawdown.png"))
        charts.append(self.plot_trade_distribution(result, f"{prefix}_trades.png"))
        charts.append(self.plot_win_rate_by_hour(result, f"{prefix}_hourly.png"))
        charts.append(self.plot_pnl_by_day(result, f"{prefix}_daily.png"))
        charts.append(self.plot_r_multiple_distribution(result, f"{prefix}_rmultiple.png"))
        
        # Combined dashboard
        charts.append(self.plot_dashboard(result, f"{prefix}_dashboard.png"))
        
        return [c for c in charts if c]
    
    def plot_equity_curve(self, result, filename: str) -> str:
        """Plot equity curve with key statistics."""
        if not result.equity_curve:
            return None
            
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['bg'])
        ax.set_facecolor(COLORS['bg'])
        
        times = [e[0] for e in result.equity_curve]
        equity = [e[1] for e in result.equity_curve]
        
        # Main equity line
        ax.plot(times, equity, color=COLORS['equity'], linewidth=1.5, label='Equity')
        
        # Fill under curve
        ax.fill_between(times, result.config.initial_capital, equity, 
                        where=[e >= result.config.initial_capital for e in equity],
                        color=COLORS['profit'], alpha=0.2)
        ax.fill_between(times, result.config.initial_capital, equity,
                        where=[e < result.config.initial_capital for e in equity],
                        color=COLORS['loss'], alpha=0.2)
        
        # Peak equity line
        peak = result.config.initial_capital
        peak_line = []
        for e in equity:
            peak = max(peak, e)
            peak_line.append(peak)
        ax.plot(times, peak_line, color=COLORS['accent'], linewidth=0.8, 
                linestyle='--', alpha=0.5, label='Peak')
        
        # Initial capital line
        ax.axhline(y=result.config.initial_capital, color=COLORS['neutral'], 
                   linestyle=':', alpha=0.5, label='Initial Capital')
        
        # Annotations
        ax.set_title(f"Equity Curve | Net: ${result.net_profit:.2f} ({(result.net_profit/result.config.initial_capital)*100:.1f}%)",
                     fontsize=14, fontweight='bold', color='white')
        ax.set_xlabel("Time", color='white')
        ax.set_ylabel("Equity ($)", color='white')
        ax.legend(loc='upper left', facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
        ax.grid(True, color=COLORS['grid'], alpha=0.3)
        ax.tick_params(colors='white')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'], edgecolor='none')
        plt.close()
        
        return filepath
    
    def plot_drawdown(self, result, filename: str) -> str:
        """Plot drawdown chart."""
        if not result.equity_curve:
            return None
            
        fig, ax = plt.subplots(figsize=(14, 4), facecolor=COLORS['bg'])
        ax.set_facecolor(COLORS['bg'])
        
        times = [e[0] for e in result.equity_curve]
        equity = [e[1] for e in result.equity_curve]
        
        # Calculate drawdown
        peak = result.config.initial_capital
        drawdowns = []
        for e in equity:
            peak = max(peak, e)
            dd = ((peak - e) / peak) * 100
            drawdowns.append(-dd)  # Negative for visualization
        
        ax.fill_between(times, 0, drawdowns, color=COLORS['drawdown'], alpha=0.7)
        ax.plot(times, drawdowns, color=COLORS['loss'], linewidth=1)
        
        # Max drawdown annotation
        min_dd = min(drawdowns)
        min_idx = drawdowns.index(min_dd)
        ax.annotate(f'Max DD: {abs(min_dd):.1f}%', 
                    xy=(times[min_idx], min_dd),
                    xytext=(times[min_idx], min_dd - 5),
                    color=COLORS['accent'],
                    fontweight='bold',
                    ha='center')
        
        ax.set_title(f"Drawdown Analysis | Max: {result.max_drawdown_pct:.1f}%", 
                     fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Time", color='white')
        ax.set_ylabel("Drawdown (%)", color='white')
        ax.grid(True, color=COLORS['grid'], alpha=0.3)
        ax.tick_params(colors='white')
        ax.set_ylim(min(drawdowns) * 1.2, 5)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'])
        plt.close()
        
        return filepath
    
    def plot_trade_distribution(self, result, filename: str) -> str:
        """Plot trade PnL distribution histogram."""
        if not result.trades:
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])
        
        pnls = [t.pnl_dollars for t in result.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # PnL Distribution
        ax = axes[0]
        ax.set_facecolor(COLORS['bg'])
        
        bins = 30
        ax.hist(wins, bins=bins, color=COLORS['profit'], alpha=0.7, label=f'Wins ({len(wins)})')
        ax.hist(losses, bins=bins, color=COLORS['loss'], alpha=0.7, label=f'Losses ({len(losses)})')
        
        ax.axvline(x=0, color='white', linestyle='-', linewidth=2)
        ax.axvline(x=result.avg_win, color=COLORS['profit'], linestyle='--', 
                   label=f'Avg Win: ${result.avg_win:.2f}')
        ax.axvline(x=result.avg_loss, color=COLORS['loss'], linestyle='--',
                   label=f'Avg Loss: ${result.avg_loss:.2f}')
        
        ax.set_title(f"Trade Distribution | Win Rate: {result.win_rate:.1f}%", 
                     fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("PnL ($)", color='white')
        ax.set_ylabel("Frequency", color='white')
        ax.legend(loc='upper right', facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
        ax.grid(True, color=COLORS['grid'], alpha=0.3)
        ax.tick_params(colors='white')
        
        # Win/Loss Pie
        ax2 = axes[1]
        ax2.set_facecolor(COLORS['bg'])
        
        sizes = [len(wins), len(losses)]
        colors = [COLORS['profit'], COLORS['loss']]
        explode = (0.05, 0)
        
        ax2.pie(sizes, explode=explode, labels=['Wins', 'Losses'], 
                colors=colors, autopct='%1.1f%%', startangle=90,
                textprops={'color': 'white', 'fontweight': 'bold'})
        ax2.set_title(f"Win/Loss Ratio | {len(wins)}W / {len(losses)}L", 
                      fontsize=12, fontweight='bold', color='white')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'])
        plt.close()
        
        return filepath
    
    def plot_win_rate_by_hour(self, result, filename: str) -> str:
        """Plot win rate and PnL by hour of day."""
        if not result.win_rate_by_hour:
            return None
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=COLORS['bg'])
        
        hours = sorted(result.win_rate_by_hour.keys())
        win_rates = [result.win_rate_by_hour.get(h, 0) for h in hours]
        pnls = [result.pnl_by_hour.get(h, 0) for h in hours]
        
        # Win Rate by Hour
        ax = axes[0]
        ax.set_facecolor(COLORS['bg'])
        
        colors = [COLORS['profit'] if wr >= 50 else COLORS['loss'] for wr in win_rates]
        bars = ax.bar(hours, win_rates, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.axhline(y=50, color=COLORS['accent'], linestyle='--', linewidth=1.5, label='Breakeven (50%)')
        ax.axhline(y=70, color=COLORS['profit'], linestyle=':', linewidth=1, alpha=0.5, label='Target (70%)')
        
        # Highlight best hours
        london_hours = [8, 9, 10, 11, 12, 13, 14, 15]
        for i, h in enumerate(hours):
            if h in london_hours:
                bars[i].set_edgecolor(COLORS['accent'])
                bars[i].set_linewidth(2)
        
        ax.set_title("Win Rate by Hour (UTC) | London Session Highlighted", 
                     fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Hour", color='white')
        ax.set_ylabel("Win Rate (%)", color='white')
        ax.set_xticks(range(0, 24))
        ax.legend(loc='upper right', facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
        ax.grid(True, axis='y', color=COLORS['grid'], alpha=0.3)
        ax.tick_params(colors='white')
        ax.set_ylim(0, 100)
        
        # PnL by Hour
        ax2 = axes[1]
        ax2.set_facecolor(COLORS['bg'])
        
        colors2 = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in pnls]
        ax2.bar(hours, pnls, color=colors2, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.axhline(y=0, color='white', linestyle='-', linewidth=1)
        
        ax2.set_title("PnL by Hour (UTC)", fontsize=12, fontweight='bold', color='white')
        ax2.set_xlabel("Hour", color='white')
        ax2.set_ylabel("PnL ($)", color='white')
        ax2.set_xticks(range(0, 24))
        ax2.grid(True, axis='y', color=COLORS['grid'], alpha=0.3)
        ax2.tick_params(colors='white')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'])
        plt.close()
        
        return filepath
    
    def plot_pnl_by_day(self, result, filename: str) -> str:
        """Plot performance by day of week."""
        if not result.win_rate_by_day:
            return None
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        win_rates = [result.win_rate_by_day.get(d, 0) for d in days]
        pnls = [result.pnl_by_day.get(d, 0) for d in days]
        
        # Win Rate
        ax = axes[0]
        ax.set_facecolor(COLORS['bg'])
        
        colors = [COLORS['profit'] if wr >= 50 else COLORS['loss'] for wr in win_rates]
        ax.bar(days, win_rates, color=colors, alpha=0.8, edgecolor='white')
        ax.axhline(y=50, color=COLORS['accent'], linestyle='--', linewidth=1.5)
        ax.axhline(y=70, color=COLORS['profit'], linestyle=':', alpha=0.5)
        
        ax.set_title("Win Rate by Day of Week", fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel("Win Rate (%)", color='white')
        ax.set_ylim(0, 100)
        ax.tick_params(colors='white', axis='both')
        ax.grid(True, axis='y', color=COLORS['grid'], alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # PnL
        ax2 = axes[1]
        ax2.set_facecolor(COLORS['bg'])
        
        colors2 = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in pnls]
        ax2.bar(days, pnls, color=colors2, alpha=0.8, edgecolor='white')
        ax2.axhline(y=0, color='white', linestyle='-', linewidth=1)
        
        ax2.set_title("PnL by Day of Week", fontsize=12, fontweight='bold', color='white')
        ax2.set_ylabel("PnL ($)", color='white')
        ax2.tick_params(colors='white', axis='both')
        ax2.grid(True, axis='y', color=COLORS['grid'], alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'])
        plt.close()
        
        return filepath
    
    def plot_r_multiple_distribution(self, result, filename: str) -> str:
        """Plot R-Multiple distribution."""
        if not result.trades:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS['bg'])
        ax.set_facecolor(COLORS['bg'])
        
        r_multiples = [t.r_multiple for t in result.trades if t.r_multiple != 0]
        
        if not r_multiples:
            return None
        
        # Histogram
        bins = np.linspace(min(r_multiples), max(r_multiples), 40)
        colors = [COLORS['profit'] if r >= 0 else COLORS['loss'] for r in r_multiples]
        
        n, bins_out, patches = ax.hist(r_multiples, bins=bins, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Color bars based on value
        for i, (patch, r) in enumerate(zip(patches, bins_out[:-1])):
            if r >= 0:
                patch.set_facecolor(COLORS['profit'])
            else:
                patch.set_facecolor(COLORS['loss'])
        
        ax.axvline(x=0, color='white', linestyle='-', linewidth=2)
        ax.axvline(x=1, color=COLORS['accent'], linestyle='--', alpha=0.8, label='1R (Breakeven Risk)')
        ax.axvline(x=2, color=COLORS['profit'], linestyle='--', alpha=0.8, label='2R (Target)')
        ax.axvline(x=result.avg_r_multiple, color='cyan', linestyle='-', linewidth=2, 
                   label=f'Avg R: {result.avg_r_multiple:.2f}')
        
        ax.set_title(f"R-Multiple Distribution | Avg R: {result.avg_r_multiple:.2f}", 
                     fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("R-Multiple", color='white')
        ax.set_ylabel("Frequency", color='white')
        ax.legend(loc='upper right', facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
        ax.grid(True, color=COLORS['grid'], alpha=0.3)
        ax.tick_params(colors='white')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'])
        plt.close()
        
        return filepath
    
    def plot_dashboard(self, result, filename: str) -> str:
        """Generate comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 14), facecolor=COLORS['bg'])
        
        # Title
        fig.suptitle(f"ATLS-FOREX BACKTEST REPORT | {result.config.symbol}", 
                     fontsize=18, fontweight='bold', color='white', y=0.98)
        
        # Create grid
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 1. Equity Curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(COLORS['bg'])
        
        if result.equity_curve:
            times = [e[0] for e in result.equity_curve]
            equity = [e[1] for e in result.equity_curve]
            ax1.plot(times, equity, color=COLORS['equity'], linewidth=1.5)
            ax1.fill_between(times, result.config.initial_capital, equity,
                            where=[e >= result.config.initial_capital for e in equity],
                            color=COLORS['profit'], alpha=0.2)
            ax1.fill_between(times, result.config.initial_capital, equity,
                            where=[e < result.config.initial_capital for e in equity],
                            color=COLORS['loss'], alpha=0.2)
            ax1.axhline(y=result.config.initial_capital, color=COLORS['neutral'], linestyle=':', alpha=0.5)
            
        ax1.set_title("Equity Curve", fontsize=11, fontweight='bold', color='white')
        ax1.grid(True, color=COLORS['grid'], alpha=0.3)
        ax1.tick_params(colors='white')
        
        # 2. Drawdown (second row, left half)
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.set_facecolor(COLORS['bg'])
        
        if result.equity_curve:
            peak = result.config.initial_capital
            drawdowns = []
            times_dd = [e[0] for e in result.equity_curve]
            for e in [e[1] for e in result.equity_curve]:
                peak = max(peak, e)
                dd = ((peak - e) / peak) * 100
                drawdowns.append(-dd)
            ax2.fill_between(times_dd, 0, drawdowns, color=COLORS['drawdown'], alpha=0.7)
            
        ax2.set_title(f"Drawdown | Max: {result.max_drawdown_pct:.1f}%", fontsize=11, fontweight='bold', color='white')
        ax2.grid(True, color=COLORS['grid'], alpha=0.3)
        ax2.tick_params(colors='white')
        
        # 3. Win Rate Pie (second row, right)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor(COLORS['bg'])
        
        if result.total_trades > 0:
            sizes = [result.winning_trades, result.losing_trades]
            colors_pie = [COLORS['profit'], COLORS['loss']]
            ax3.pie(sizes, colors=colors_pie, autopct='%1.1f%%', startangle=90,
                   textprops={'color': 'white', 'fontsize': 10})
            ax3.set_title(f"Win Rate: {result.win_rate:.1f}%", fontsize=11, fontweight='bold', color='white')
        
        # 4. Key Metrics Table (second row, far right)
        ax4 = fig.add_subplot(gs[1, 3])
        ax4.set_facecolor(COLORS['bg'])
        ax4.axis('off')
        
        metrics_text = f"""
━━━━ PERFORMANCE ━━━━
Total Trades: {result.total_trades}
Win Rate: {result.win_rate:.1f}%
Profit Factor: {result.profit_factor:.2f}

━━━━━ PnL ($) ━━━━━━
Net Profit: ${result.net_profit:.2f}
Gross Profit: ${result.gross_profit:.2f}
Gross Loss: ${result.gross_loss:.2f}

━━━━━ RISK ━━━━━━━━
Max Drawdown: {result.max_drawdown_pct:.1f}%
Sharpe Ratio: {result.sharpe_ratio:.2f}
Avg R-Multiple: {result.avg_r_multiple:.2f}

━━━━ TRADING STATS ━━━
Avg Win: ${result.avg_win:.2f}
Avg Loss: ${result.avg_loss:.2f}
Expectancy: ${result.expectancy:.2f}
"""
        ax4.text(0.1, 0.95, metrics_text, transform=ax4.transAxes, 
                fontsize=9, color='white', verticalalignment='top',
                fontfamily='monospace')
        
        # 5. Win Rate by Hour (third row, left half)
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.set_facecolor(COLORS['bg'])
        
        if result.win_rate_by_hour:
            hours = sorted(result.win_rate_by_hour.keys())
            win_rates = [result.win_rate_by_hour.get(h, 0) for h in hours]
            colors_h = [COLORS['profit'] if wr >= 50 else COLORS['loss'] for wr in win_rates]
            ax5.bar(hours, win_rates, color=colors_h, alpha=0.8, edgecolor='white', linewidth=0.5)
            ax5.axhline(y=50, color=COLORS['accent'], linestyle='--', linewidth=1)
            ax5.axhline(y=70, color=COLORS['profit'], linestyle=':', alpha=0.5)
        
        ax5.set_title("Win Rate by Hour (UTC)", fontsize=11, fontweight='bold', color='white')
        ax5.set_xticks(range(0, 24))
        ax5.set_ylim(0, 100)
        ax5.grid(True, axis='y', color=COLORS['grid'], alpha=0.3)
        ax5.tick_params(colors='white')
        
        # 6. PnL Distribution (third row, right half)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.set_facecolor(COLORS['bg'])
        
        if result.trades:
            pnls = [t.pnl_dollars for t in result.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            ax6.hist(wins, bins=20, color=COLORS['profit'], alpha=0.7, label='Wins')
            ax6.hist(losses, bins=20, color=COLORS['loss'], alpha=0.7, label='Losses')
            ax6.axvline(x=0, color='white', linestyle='-', linewidth=1.5)
            ax6.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'])
        
        ax6.set_title("Trade Distribution", fontsize=11, fontweight='bold', color='white')
        ax6.grid(True, color=COLORS['grid'], alpha=0.3)
        ax6.tick_params(colors='white')
        
        # 7. R-Multiple Distribution (bottom left)
        ax7 = fig.add_subplot(gs[3, :2])
        ax7.set_facecolor(COLORS['bg'])
        
        if result.trades:
            r_multiples = [t.r_multiple for t in result.trades if t.r_multiple != 0]
            if r_multiples:
                bins = np.linspace(min(r_multiples), max(r_multiples), 30)
                n, bins_out, patches = ax7.hist(r_multiples, bins=bins, alpha=0.8, edgecolor='white', linewidth=0.3)
                for i, (patch, r) in enumerate(zip(patches, bins_out[:-1])):
                    patch.set_facecolor(COLORS['profit'] if r >= 0 else COLORS['loss'])
                ax7.axvline(x=0, color='white', linestyle='-', linewidth=1.5)
                ax7.axvline(x=1, color=COLORS['accent'], linestyle='--', alpha=0.8)
        
        ax7.set_title("R-Multiple Distribution", fontsize=11, fontweight='bold', color='white')
        ax7.grid(True, color=COLORS['grid'], alpha=0.3)
        ax7.tick_params(colors='white')
        
        # 8. Day of Week Performance (bottom right)
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.set_facecolor(COLORS['bg'])
        
        if result.pnl_by_day:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            full_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            pnls = [result.pnl_by_day.get(d, 0) for d in full_days]
            colors_d = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in pnls]
            ax8.bar(days, pnls, color=colors_d, alpha=0.8, edgecolor='white')
            ax8.axhline(y=0, color='white', linestyle='-', linewidth=1)
        
        ax8.set_title("PnL by Day of Week", fontsize=11, fontweight='bold', color='white')
        ax8.grid(True, axis='y', color=COLORS['grid'], alpha=0.3)
        ax8.tick_params(colors='white')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, facecolor=COLORS['bg'], edgecolor='none', 
                   bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        return filepath
