# Genesis Trade Journal Integration

Shows how to integrate the Auto Trade Journal into Genesis for automatic trade documentation.

## 🎯 Overview

The Trade Journal automatically creates beautiful markdown entries for every trade with:
- ✅ Full trade context (entry, exit, SL, TP)
- ✅ Market conditions and analysis
- ✅ Decision-making rationale
- ✅ Post-trade review and lessons
- ✅ Searchable tags and ratings

---

## 🚀 Quick Start

### Basic Usage

```python
from analytics.trade_journal import TradeJournal, JournalEntry

# Initialize journal
journal = TradeJournal()

# Create entry when trade opens
entry = JournalEntry(
    trade_id="20260111_140530_BUY",
    timestamp=datetime.now(),
    direction="BUY",
    symbol="GBPUSD",
    entry_price=1.2650,
    sl_price=1.2630,
    tp_price=1.2700,
    setup_type="GENESIS_PULLBACK",
    confidence=75.0,
    signal_score=70,
    agi_score=80,
    swarm_score=75,
    market_regime="TRENDING",
    reasons=["Strong pullback", "EMA confluence"],
    hour_of_day=14,
    day_of_week=2
)

journal.create_entry(entry)
```

### Update When Trade Closes

```python
# Update with results
journal.update_entry(
    "20260111_140530_BUY",
    exit_price=1.2680,
    profit_loss=30.0,
    profit_pips=30.0,
    win=True
)
```

### Add Post-Trade Review

```python
# Add your analysis
journal.add_review(
    "20260111_140530_BUY",
    notes="Perfect execution at support level",
    lessons="Wait for pullback confirmation",
    rating=5,
    tags=["pullback", "perfect-entry"]
)
```

---

## 🔗 Genesis Integration

### Add to genesis_analytics.py

```python
# In genesis_analytics.py
from analytics.trade_journal import TradeJournal, JournalEntry

class GenesisAnalyticsIntegration:
    def __init__(self):
        self.analyzer = TradeAnalyzer()
        self.journal = TradeJournal()  # Add this!
        
    def on_signal(self, signal, current_price):
        # Existing recording...
        self.analytics.on_signal(signal, current_price)
        
        # Create journal entry
        entry = JournalEntry(
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
            reasons=signal.reasons,
            vetoes=signal.vetoes,
            hour_of_day=signal.timestamp.hour,
            day_of_week=signal.timestamp.weekday()
        )
        
        self.journal.create_entry(entry)
        logger.info(f"📝 Journal entry created!")
    
    def on_trade_close(self, trade_id, exit_price, profit_loss, profit_pips):
        # Existing analytics...
        self.analytics.on_trade_close(...)
        
        # Update journal
        self.journal.update_entry(
            trade_id,
            exit_price=exit_price,
            profit_loss=profit_loss,
            profit_pips=profit_pips,
            win=profit_loss > 0
        )
        
        logger.info(f"📝 Journal updated with results!")
```

---

## 📊 Generated Journal Example

Each trade creates a markdown file like:

```
reports/trade_journal/
  2026-01/
    20260111_140530_BUY_001.md
    20260111_153045_SELL_002.md
  2026-02/
    20260201_090015_BUY_003.md
```

**Contents:**
- Trade summary table
- Entry/exit prices
- Results (P/L, pips, outcome)
- Setup analysis with layer scores
- Market conditions
- Reasons and warnings
- Post-trade review
- Lessons learned
- Rating and tags

---

## 🔍 Search & Analysis

```python
# Search by setup
pullback_trades = journal.search_entries(setup_type="GENESIS_PULLBACK")

# Search winners
winners = journal.search_entries(win=True)

# Search by time
morning_trades = journal.search_entries(hour_of_day=9)

# Get specific entry
entry = journal.get_entry("20260111_140530_BUY")

# Generate summary
summary = journal.generate_summary(days=30)
print(summary)
```

---

## 🎯 Features

### Automatic Documentation
- ✅ Every trade documented
- ✅ No manual work required
- ✅ Beautiful markdown format

### Rich Context
- ✅ Full market analysis
- ✅ Decision rationale
- ✅ Layer scores (Signal, AGI, Swarm)

### Review System
- ✅ Post-trade notes
- ✅ Lessons learned extraction
- ✅ 5-star rating
- ✅ Searchable tags

### Organization
- ✅ Monthly folders
- ✅ Indexed for search
- ✅ Easy to browse

---

## 💡 Use Cases

### 1. Daily Review
Review all trades from today:
```python
from datetime import datetime
today = datetime.now().date()
today_trades = [e for e in journal.entries.values() 
                if e.timestamp.date() == today]
```

### 2. Study Best Trades
Find your 5-star trades:
```python
best_trades = journal.search_entries(rating=5)
for trade in best_trades:
    print(f"Study: {trade.trade_id} - {trade.setup_type}")
```

### 3. Identify Patterns
Find what works:
```python
# Best performing setups
from collections import Counter
wins = journal.search_entries(win=True)
setups = Counter(t.setup_type for t in wins)
print(setups.most_common(5))
```

### 4. Learn from Mistakes
Review losses for improvement:
```python
losses = journal.search_entries(win=False)
for trade in losses:
    if trade.lessons_learned:
        print(f"Lesson: {trade.lessons_learned}")
```

---

## 📝 Review Workflow

### End of Day:
```python
# Get today's trades
today_trades = journal.search_entries(...)

# Review each one
for trade in today_trades:
    # Open the markdown file
    # Add notes and lessons
    journal.add_review(
        trade.trade_id,
        notes="Your thoughts...",
        lessons="What you learned...",
        rating=4,
        tags=["tag1", "tag2"]
    )
```

### Weekly Review:
```python
# Generate weekly summary
summary = journal.generate_summary(days=7)
print(summary)

# Review patterns
# Identify improvements
# Update strategies
```

---

## 🎉 Benefits

✅ **Never forget a trade** - All documented automatically  
✅ **Learn faster** - Review and extract lessons  
✅ **Track improvement** - See patterns over time  
✅ **Professional** - Beautiful markdown journals  
✅ **Searchable** - Find any trade instantly  
✅ **Organized** - Monthly structure  

---

## 🚀 Next Steps

1. Integrate with Genesis (add to `genesis_analytics.py`)
2. Start trading and journal automatically creates
3. Review trades daily/weekly
4. Extract lessons and improve
5. Watch your trading evolve!

---

**Created:** January 11, 2026  
**Status:** ✅ Ready to Use  
**Location:** `analytics/trade_journal.py`
