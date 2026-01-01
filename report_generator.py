import os
import logging
from jinja2 import Template
from datetime import datetime

logger = logging.getLogger("Atl4s-Reports")

class ReportGenerator:
    def __init__(self, report_dir="reports"):
        self.report_dir = report_dir
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
    def generate_daily_report(self, account_info, trades, analysis_summary):
        """
        Generates a daily HTML report with Dark Moody Glassmorphism design and functional tabs.
        """
        try:
            template_str = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Atl4s Dashboard</title>
                <link rel="stylesheet" href="style.css">
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body>
                <div class="dashboard-container">
                    <!-- Sidebar -->
                    <div class="sidebar">
                        <div class="nav-icon active" onclick="switchTab('dashboard')">📊</div>
                        <div class="nav-icon" onclick="switchTab('strategy')">🧠</div>
                        <div class="nav-icon" onclick="switchTab('portfolio')">💼</div>
                        <div class="nav-icon" onclick="switchTab('logs')">📜</div>
                        <div class="nav-icon" onclick="switchTab('settings')">⚙️</div>
                    </div>

                    <!-- Main Content Area -->
                    <div class="main-content-wrapper" style="grid-column: 2 / -1; overflow-y: auto;">
                        
                        <!-- TAB: DASHBOARD -->
                        <div id="dashboard" class="tab-content active main-content">
                            <!-- Hero Card: Equity -->
                            <div class="glass-card hero-card">
                                <div>
                                    <div class="hero-title">Total Equity</div>
                                    <div class="hero-value">${{ equity }}</div>
                                    <div class="hero-title">Monthly Yield: <span style="color: var(--neon-green)">+{{ monthly_yield }}%</span></div>
                                </div>
                                <canvas id="equityChart" style="height: 100px; width: 100%;"></canvas>
                            </div>

                            <!-- Strategy Card -->
                            <div class="glass-card strategy-card">
                                <div class="hero-title">Intelligence Core</div>
                                <div class="badge">
                                    <span class="badge-label">Active Regime</span>
                                    <span class="badge-value">{{ regime }}</span>
                                </div>
                                <div class="badge">
                                    <span class="badge-label">Win Rate</span>
                                    <span class="badge-value">{{ win_rate }}%</span>
                                </div>
                                <div class="badge">
                                    <span class="badge-label">Profit Factor</span>
                                    <span class="badge-value">{{ profit_factor }}</span>
                                </div>
                                <div class="badge">
                                    <span class="badge-label">Active Modules</span>
                                    <span class="badge-value">7/7</span>
                                </div>
                            </div>
                            
                             <!-- Health Card -->
                            <div class="glass-card health-card">
                                <div class="status-indicator">
                                    <div class="pulsing-dot"></div>
                                    System Operational
                                </div>
                                <div class="drawdown-circle">
                                    <div class="drawdown-value">{{ drawdown }}%</div>
                                    <div class="drawdown-label">Drawdown</div>
                                </div>
                            </div>

                            <!-- History Card -->
                            <div class="glass-card history-card">
                                <div class="hero-title" style="margin-bottom: 20px;">Recent Operations</div>
                                <div class="table-header">
                                    <div>Time</div>
                                    <div>Symbol</div>
                                    <div>Type</div>
                                    <div>Profit</div>
                                </div>
                                {% for trade in trades %}
                                <div class="table-row">
                                    <div>{{ trade.time }}</div>
                                    <div>{{ trade.symbol }}</div>
                                    <div>{{ trade.type }}</div>
                                    <div class="{{ 'profit-positive' if trade.profit >= 0 else 'profit-negative' }}">
                                        {{ "+" if trade.profit >= 0 else "" }}{{ trade.profit }}
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>

                        <!-- TAB: STRATEGY -->
                        <div id="strategy" class="tab-content" style="display: none; grid-template-columns: 1fr 1fr; gap: 24px;">
                            <div class="glass-card" style="grid-column: span 2;">
                                <div class="hero-title">Deep Reasoning Matrix 2.0</div>
                                <p style="color: var(--text-secondary); margin-top: 10px;">Real-time analysis of active decision modules.</p>
                            </div>
                            
                            <!-- Module Status -->
                            <div class="glass-card">
                                <div class="hero-title">Module Status</div>
                                <div style="margin-top: 20px; display: flex; flex-direction: column; gap: 15px;">
                                    <div class="badge"><span class="badge-label">Trend Architect</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                    <div class="badge"><span class="badge-label">The Sniper</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                    <div class="badge"><span class="badge-label">The Quant</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                    <div class="badge"><span class="badge-label">Pattern Recon</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                    <div class="badge"><span class="badge-label">Market Cycle</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                    <div class="badge"><span class="badge-label">Supply & Demand</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                    <div class="badge"><span class="badge-label">Divergence Hunter</span><span class="badge-value" style="color: var(--neon-green)">ACTIVE</span></div>
                                </div>
                            </div>

                            <!-- Radar Chart -->
                            <div class="glass-card">
                                <div class="hero-title">Consensus Weighting</div>
                                <canvas id="radarChart"></canvas>
                            </div>
                        </div>

                        <!-- TAB: PORTFOLIO -->
                        <div id="portfolio" class="tab-content" style="display: none; grid-template-columns: 1fr 1fr; gap: 24px;">
                            <div class="glass-card">
                                <div class="hero-title">Capital Overview</div>
                                <div style="margin-top: 20px;">
                                    <div class="hero-value" style="font-size: 2.5rem;">${{ balance }}</div>
                                    <div class="hero-title">Balance</div>
                                </div>
                                <div style="margin-top: 20px;">
                                    <div class="hero-value" style="font-size: 2.5rem;">${{ equity }}</div>
                                    <div class="hero-title">Equity</div>
                                </div>
                            </div>
                            <div class="glass-card">
                                <div class="hero-title">Asset Allocation</div>
                                <canvas id="allocationChart"></canvas>
                            </div>
                        </div>

                        <!-- TAB: LOGS -->
                        <div id="logs" class="tab-content" style="display: none;">
                            <div class="glass-card" style="height: 80vh; overflow-y: auto; font-family: monospace; font-size: 0.9rem;">
                                <div class="hero-title">System Logs</div>
                                <div style="color: var(--text-secondary); margin-top: 20px;">
                                    <span style="color: var(--neon-blue)">[INFO]</span> System initialized.<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> Connected to MT5 Bridge.<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> Loading M5 Data...<br>
                                    <span style="color: var(--neon-green)">[SUCCESS]</span> Data Loaded (5000 candles).<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> Running Consensus Engine...<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> Trend: BULLISH | Score: 85<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> SupplyDemand: No Zones Detected.<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> Divergence: Hidden Bullish Detected.<br>
                                    <span style="color: var(--neon-green)">[DECISION]</span> BUY Signal Generated (Score: 65).<br>
                                    <span style="color: var(--neon-blue)">[INFO]</span> Trade Manager: Trailing Stop active.<br>
                                </div>
                            </div>
                        </div>

                        <!-- TAB: SETTINGS -->
                        <div id="settings" class="tab-content" style="display: none;">
                            <div class="glass-card">
                                <div class="hero-title">System Configuration</div>
                                <div class="table-header" style="margin-top: 20px;">
                                    <div>Parameter</div>
                                    <div>Value</div>
                                </div>
                                <div class="table-row"><div>Symbol</div><div>XAUUSD</div></div>
                                <div class="table-row"><div>Timeframe</div><div>M5</div></div>
                                <div class="table-row"><div>Risk Per Trade</div><div>2%</div></div>
                                <div class="table-row"><div>Max Drawdown</div><div>5%</div></div>
                                <div class="table-row"><div>Leverage</div><div>1:500</div></div>
                                <div class="table-row"><div>Magic Number</div><div>123456</div></div>
                            </div>
                        </div>

                    </div>
                </div>

                <script>
                    function switchTab(tabId) {
                        // Hide all tabs
                        document.querySelectorAll('.tab-content').forEach(el => {
                            el.style.display = 'none';
                            el.classList.remove('active');
                        });
                        // Show selected tab
                        const selected = document.getElementById(tabId);
                        selected.style.display = tabId === 'dashboard' ? 'grid' : (tabId === 'logs' || tabId === 'settings' ? 'block' : 'grid');
                        selected.classList.add('active');
                        
                        // Update Sidebar Icons
                        document.querySelectorAll('.nav-icon').forEach(el => el.classList.remove('active'));
                        event.currentTarget.classList.add('active');
                    }

                    // Charts
                    const ctxEquity = document.getElementById('equityChart').getContext('2d');
                    new Chart(ctxEquity, {
                        type: 'line',
                        data: {
                            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                            datasets: [{
                                label: 'Equity',
                                data: [30, 32, 31, 34, 35],
                                borderColor: '#2AF598',
                                backgroundColor: 'rgba(42, 245, 152, 0.1)',
                                tension: 0.4,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: { legend: { display: false } },
                            scales: { x: { display: false }, y: { display: false } }
                        }
                    });

                    const ctxRadar = document.getElementById('radarChart').getContext('2d');
                    new Chart(ctxRadar, {
                        type: 'radar',
                        data: {
                            labels: ['Trend', 'Sniper', 'Quant', 'Patterns', 'Cycle', 'S&D', 'Divergence'],
                            datasets: [{
                                label: 'Influence',
                                data: [80, 60, 40, 50, 70, 65, 55],
                                backgroundColor: 'rgba(0, 158, 253, 0.2)',
                                borderColor: '#009EFD',
                                pointBackgroundColor: '#fff'
                            }]
                        },
                        options: {
                            scales: {
                                r: {
                                    angleLines: { color: 'rgba(255,255,255,0.1)' },
                                    grid: { color: 'rgba(255,255,255,0.1)' },
                                    pointLabels: { color: '#fff' }
                                }
                            }
                        }
                    });
                    
                    const ctxAlloc = document.getElementById('allocationChart').getContext('2d');
                    new Chart(ctxAlloc, {
                        type: 'doughnut',
                        data: {
                            labels: ['XAUUSD', 'Cash'],
                            datasets: [{
                                data: [80, 20],
                                backgroundColor: ['#2AF598', 'rgba(255,255,255,0.1)'],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            plugins: { legend: { position: 'bottom', labels: { color: '#fff' } } }
                        }
                    });
                </script>
            </body>
            </html>
            """
            
            template = Template(template_str)
            
            # Calculate Metrics
            equity = account_info.get('equity', 0.0)
            balance = account_info.get('balance', 0.0)
            
            # Mock Data for missing metrics (since we are in dev mode)
            monthly_yield = 12.5 
            regime = analysis_summary.get('regime', 'Unknown')
            win_rate = 68
            profit_factor = 2.4
            drawdown = 1.2
            
            html_content = template.render(
                equity=f"{equity:.2f}",
                balance=f"{balance:.2f}",
                monthly_yield=monthly_yield,
                regime=regime,
                win_rate=win_rate,
                profit_factor=profit_factor,
                drawdown=drawdown,
                trades=trades
            )
            
            filename = f"report_{datetime.now().strftime('%Y%m%d')}.html"
            filepath = os.path.join(self.report_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            logger.info(f"Report Generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None
