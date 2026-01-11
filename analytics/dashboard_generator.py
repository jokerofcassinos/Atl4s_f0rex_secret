"""
Genesis Performance Dashboard Generator

Creates beautiful HTML dashboard with:
- Performance charts
- Real-time stats
- Trade journal integration
- ML optimization insights
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Fix import  
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.trade_analyzer import TradeAnalyzer
from analytics.ml_optimizer import MLOptimizer


class DashboardGenerator:
    """Generates interactive HTML performance dashboard"""
    
    def __init__(self, output_dir: str = "reports/dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = TradeAnalyzer()
        self.optimizer = MLOptimizer(self.analyzer)
    
    def generate(self, days: int = 30) -> str:
        """Generate complete dashboard"""
        
        # Get data
        analysis = self.analyzer.analyze_performance(days)
        optimizations = self.optimizer.analyze_optimal_parameters(days)
        
        # Generate HTML
        html = self._generate_html(analysis, optimizations, days)
        
        # Save
        filepath = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        # Also save as "latest"
        latest_path = self.output_dir / "latest.html"
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def _generate_html(self, analysis: Dict, optimizations: List, days: int) -> str:
        """Generate HTML content"""
        
        # Extract key metrics
        total_trades = analysis.get('total_trades', 0)
        win_rate = analysis.get('win_rate', 0)
        profitability = analysis.get('profitability', {})
        total_profit = profitability.get('total_profit', 0)
        
        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis Performance Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .card h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .stat {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            color: #666;
            font-weight: 500;
        }}
        
        .stat-value {{
            color: #667eea;
            font-weight: bold;
            font-size: 1.3em;
        }}
        
        .stat-value.positive {{
            color: #10b981;
        }}
        
        .stat-value.negative {{
            color: #ef4444;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
        
        .optimization-list {{
            list-style: none;
        }}
        
        .optimization-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .opt-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .opt-detail {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 5px;
        }}
        
        .badge.success {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .badge.warning {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .timestamp {{
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Genesis Performance Dashboard</h1>
            <p class="subtitle">Last {days} days • Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="grid">
            <!-- Performance Overview -->
            <div class="card">
                <h2>📊 Performance Overview</h2>
                <div class="stat">
                    <span class="stat-label">Total Trades</span>
                    <span class="stat-value">{total_trades}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Win Rate</span>
                    <span class="stat-value {'positive' if win_rate >= 70 else 'warning'}">{win_rate:.1f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {win_rate}%">{win_rate:.1f}%</div>
                </div>
                <div class="stat">
                    <span class="stat-label">Total Profit</span>
                    <span class="stat-value {'positive' if total_profit > 0 else 'negative'}">${total_profit:.2f}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Target (70%)</span>
                    <span class="stat-value">{'✅ MET' if win_rate >= 70 else '⚠️ BELOW'}</span>
                </div>
            </div>
            
            <!-- ML Optimizations -->
            <div class="card">
                <h2>🧠 ML Optimizations</h2>
"""
        
        if optimizations:
            html += '<ul class="optimization-list">\n'
            for opt in optimizations[:5]:
                html += f"""
                    <li class="optimization-item">
                        <div class="opt-title">{opt.parameter_name}</div>
                        <div class="opt-detail">
                            Suggested: {opt.suggested_value} (Current: {opt.current_value})
                        </div>
                        <div class="opt-detail">
                            Expected: +{opt.expected_improvement:.1f}% WR
                        </div>
                        <span class="badge success">{opt.confidence:.0f}% confidence</span>
                    </li>
"""
            html += '</ul>\n'
        else:
            html += '<p>No optimizations available. Need more trade data.</p>\n'
        
        html += """
            </div>
            
            <!-- Setup Performance -->
            <div class="card">
                <h2>🎯 Setup Performance</h2>
"""
        
        setup_analysis = analysis.get('setup_analysis', {})
        by_setup = setup_analysis.get('by_setup', {})
        
        if by_setup:
            for setup, data in sorted(by_setup.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]:
                win_rate_setup = data['win_rate']
                count = data['count']
                html += f"""
                <div class="stat">
                    <span class="stat-label">{setup}</span>
                    <span class="stat-value">{win_rate_setup:.0f}% ({count} trades)</span>
                </div>
"""
        else:
            html += '<p>No setup data available yet.</p>\n'
        
        html += """
            </div>
            
            <!-- Time Analysis -->
            <div class="card">
                <h2>⏰ Best Trading Times</h2>
"""
        
        time_analysis = analysis.get('time_analysis', {})
        best_hours = time_analysis.get('best_hours', [])
        
        if best_hours:
            for hour, wr in best_hours[:5]:
                html += f"""
                <div class="stat">
                    <span class="stat-label">{hour:02d}:00</span>
                    <span class="stat-value positive">{wr:.0f}%</span>
                </div>
"""
        else:
            html += '<p>Need more trades to identify patterns.</p>\n'
        
        html += """
            </div>
        </div>
        
        <div class="timestamp">
            🏆 Genesis Trading System - Powered by AGI + Analytics + ML
        </div>
    </div>
</body>
</html>
"""
        
        return html


if __name__ == "__main__":
    # Generate dashboard
    generator = DashboardGenerator()
    filepath = generator.generate(days=30)
    print(f"✅ Dashboard generated: {filepath}")
    print(f"🌐 Open in browser: {Path(filepath).absolute()}")
