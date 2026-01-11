import os
from pathlib import Path
import datetime

class DashboardGenerator:
    def __init__(self, report_dir="reports"):
        self.report_dir = Path(report_dir)
        if not self.report_dir.exists():
            os.makedirs(self.report_dir)
            
    def generate_report(self, history_df=None, account_info=None):
        """Generates the HTML dashboard."""
        
        # Color Palette
        # Background: #0f172a (Dark Blue/Slate)
        # Glass: rgba(255, 255, 255, 0.05)
        # Neon Cyan: #00f3ff
        # Neon Purple: #bc13fe
        
        # Placeholder data if None
        equity = account_info.get('equity', 30.0) if account_info else 30.0
        balance = account_info.get('balance', 30.0) if account_info else 30.0
        profit = account_info.get('profit', 0.0) if account_info else 0.0
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Atl4s-Forex Dashboard</title>
    <style>
        :root {{
            --bg-color: #050b14;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.1);
            --neon-cyan: #00f3ff;
            --neon-purple: #bc13fe;
            --text-main: #ffffff;
            --text-muted: #94a3b8;
        }}
        
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(188, 19, 254, 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(0, 243, 255, 0.1) 0%, transparent 20%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--glass-border);
        }}
        
        .logo {{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(to right, var(--neon-cyan), var(--neon-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .card {{
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 0 20px rgba(0, 243, 255, 0.1);
            border-color: rgba(0, 243, 255, 0.3);
        }}
        
        .card h3 {{
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0;
        }}
        
        .card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }}
        
        .value.green {{ color: #4ade80; text-shadow: 0 0 10px rgba(74, 222, 128, 0.4); }}
        .value.red {{ color: #f87171; text-shadow: 0 0 10px rgba(248, 113, 113, 0.4); }}
        
        .status-dot {{
            height: 10px;
            width: 10px;
            background-color: #4ade80;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px #4ade80;
            margin-right: 8px;
        }}
        
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="logo">ATL4S // FOREX</div>
            <div style="display: flex; align-items: center; color: var(--text-muted);">
                <span class="status-dot"></span> ORCHESTRATOR ACTIVE &nbsp; | &nbsp; {timestamp}
            </div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h3>Total Equity</h3>
                <div class="value">${equity:.2f}</div>
                <div style="color: var(--text-muted);">Balance: ${balance:.2f}</div>
            </div>
            
            <div class="card">
                <h3>Current Session Profit</h3>
                <div class="value { 'green' if profit >= 0 else 'red' }">
                    {'+' if profit >= 0 else ''}{profit:.2f}
                </div>
            </div>
            
            <div class="card">
                <h3>Win Rate</h3>
                <div class="value">0%</div>
                <div style="color: var(--text-muted);">0 Trades Today</div>
            </div>
            
            <div class="card">
                <h3>Active Strategy</h3>
                <div class="value" style="font-size: 1.5rem; color: var(--neon-cyan);">
                    QUANTUM // HYBRID
                </div>
                <div style="color: var(--text-muted);">Z-Score + Entropy + Hurst</div>
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <div class="card" style="height: 400px; display: flex; align-items: center; justify-content: center; color: var(--text-muted);">
                [CHART VISUALIZATION PLACEHOLDER - INTEGRATE PLOTLY JS]
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        output_path = self.report_dir / "dashboard.html"
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
