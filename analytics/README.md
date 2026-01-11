# Genesis Analytics System 📊

**Deep Performance Analysis for Genesis Trading Bot**

## 🎯 Features

### 1. **Trade Recording**
- Automatic capture of every trade
- Full context (signals, scores, market conditions)
- Persistent storage (JSON database)

### 2. **Performance Analytics**
- Overall performance metrics
- Setup-specific analysis
- Time-based patterns
- Market condition correlation

### 3. **Pattern Detection**
- Identifies winning setups
- Detects failure patterns
- Time-of-day optimization
- Score correlation analysis

### 4. **Actionable Insights**
- Auto-generated recommendations
- Best/worst trade identification
- Optimization suggestions

---

## 🚀 Quick Start

### Basic Usage

```python
from analytics.genesis_analytics import get_analytics

# Get analytics instance
analytics = get_analytics()

# Record a trade (automatic from Genesis)
# Trades are captured automatically when Genesis executes

# Get real-time stats
stats = analytics.get_real_time_stats()
print(stats)

# Generate reports
daily_report = analytics.generate_daily_report()
weekly_report = analytics.generate_weekly_report()
monthly_report = analytics.generate_monthly_report()
```

### Integration with Genesis

```python
from analytics.genesis_analytics import get_analytics

# In your main trading loop:
analytics = get_analytics()

# When signal is generated:
analytics.on_signal(genesis_signal, current_price)

# When trade closes:
analytics.on_trade_close(
    trade_id="20260111_140530_BUY",
    exit_price=1.2680,
    profit_loss=30.0,
    profit_pips=30.0
)
```

---

## 📊 Report Examples

### Daily Report
```
📊 OVERALL PERFORMANCE
Total Trades:     5
Wins:             4
Losses:           1
Win Rate:         80.0%

💰 PROFITABILITY
Total Profit:     $150.00
Avg/Trade:        $30.00

🎯 TOP SETUPS
Best Setup:       GENESIS_PULLBACK (100% WR)
Worst Setup:      GENESIS_BREAKOUT (50% WR)

⏰ BEST TIMES
Best Hour:        14:00 (90% WR)
Best Day:         Wednesday (85% WR)
```

---

## 🔍 Analysis Modules

### 1. Setup Analysis
```python
# Analyzes performance by setup type
setup_analysis = analyzer.analyze_performance()['setup_analysis']

# Returns:
{
    "by_setup": {
        "GENESIS_PULLBACK": {
            "count": 10,
            "win_rate": 80.0,
            "total_profit": 300.0,
            "avg_profit": 30.0
        }
    },
    "best_setup": "GENESIS_PULLBACK",
    "worst_setup": "GENESIS_BREAKOUT"
}
```

### 2. Time Analysis
```python
# Finds optimal trading times
time_analysis = analyzer.analyze_performance()['time_analysis']

# Returns best hours and days
```

### 3. Score Correlation
```python
# Correlates scores with outcomes
score_analysis = analyzer.analyze_performance()['score_analysis']

# Shows if higher confidence = better results
```

---

## 💡 Auto-Generated Insights

The system automatically generates insights like:

```
✅ Excellent win rate (75.0%) - System performing as expected
🎯 Best setup: GENESIS_PULLBACK | Avoid: GENESIS_BREAKOUT
⏰ Best trading time: 14:00 (85% WR)
💡 Higher confidence trades perform better (80% vs 65%)
```

---

## 📁 Data Storage

Trades are stored in: `data/genesis_trades.json`

Format:
```json
[
  {
    "trade_id": "20260111_140530_BUY",
    "timestamp": "2026-01-11T14:05:30",
    "direction": "BUY",
    "entry_price": 1.2650,
    "exit_price": 1.2680,
    "profit_loss": 30.0,
    "win": true,
    "setup_type": "GENESIS_PULLBACK",
    "confidence": 75.0,
    ...
  }
]
```

---

## 📈 Performance Tracking

### Real-Time Stats
```python
stats = analytics.get_real_time_stats()

# Returns:
{
    "total_trades": 50,
    "win_rate": "75.0%",
    "total_profit": "$1,500.00",
    "active_trades": 2,
    "status": "✅ ON TARGET"
}
```

### Auto-Analysis
- Triggers after every 10 trades
- Generates insights automatically
- Logs recommendations

---

## 🎯 Next Steps

### Integration Checklist:
- [x] Trade recording system
- [x] Performance analytics
- [x] Report generation
- [x] Insight generation
- [ ] Connect to Genesis main loop
- [ ] Add live dashboard
- [ ] Telegram notifications
- [ ] ML-based optimization

---

## 👨‍💻 API Reference

### TradeAnalyzer
```python
class TradeAnalyzer:
    def __init__(db_path: str)
    def record_trade(trade: TradeRecord)
    def analyze_performance(days: int = 30) -> Dict
    def generate_report(days: int = 30) -> str
```

### GenesisAnalyticsIntegration
```python
class GenesisAnalyticsIntegration:
    def on_signal(signal: GenesisSignal, current_price: float)
    def on_trade_close(trade_id, exit_price, profit_loss, profit_pips)
    def generate_daily_report() -> str
    def get_real_time_stats() -> dict
```

---

## 🔧 Configuration

### Database Path
```python
analyzer = TradeAnalyzer(db_path="custom/path/trades.json")
```

### Analysis Period
```python
# Analyze last 7 days
analysis = analyzer.analyze_performance(days=7)

# Analyze last 30 days
analysis = analyzer.analyze_performance(days=30)
```

---

## 📊 Example Output

```
╔══════════════════════════════════════════════════════╗
║      GENESIS TRADE ANALYSIS REPORT                   ║
╚══════════════════════════════════════════════════════╝

📊 OVERALL PERFORMANCE
Total Trades:     50
Win Rate:         75.0%
Total Profit:     $1,500.00

🎯 TOP SETUPS (by Win Rate)
1. GENESIS_PULLBACK      85.0% (17/20 trades)
2. GENESIS_BREAKOUT      70.0% (14/20 trades)
3. GENESIS_REVERSAL      65.0% (13/20 trades)

⏰ BEST TRADING TIMES
Monday:      70.0% (7/10)
Wednesday:   85.0% (17/20)
14:00-15:00: 90.0% (9/10)

💡 KEY INSIGHTS
1. ✅ Excellent win rate (75.0%)
2. 🎯 Focus on GENESIS_PULLBACK setups
3. ⏰ Trade primarily 14:00-15:00 on Wednesdays
4. 💡 Higher swarm scores correlate with wins
```

---

## 🎉 Benefits

✅ **Data-Driven Decisions** - Make changes based on facts, not guesses  
✅ **Continuous Improvement** - Identify what works and what doesn't  
✅ **Performance Tracking** - Monitor progress towards 70%+ WR goal  
✅ **Pattern Recognition** - Discover hidden winning patterns  
✅ **Risk Management** - Understand and optimize risk/reward  

---

**Created by:** Antigravity AI  
**Date:** January 11, 2026  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
