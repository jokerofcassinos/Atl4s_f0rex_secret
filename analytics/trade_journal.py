"""
Genesis Auto Trade Journal

Automatically documents every trade with full context, analysis, and learning extraction.
Creates beautiful markdown journal entries for each trade.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("TradeJournal")


@dataclass
class JournalEntry:
    """Complete trade journal entry"""
    # Trade Info
    trade_id: str
    timestamp: datetime
    direction: str  # BUY/SELL
    symbol: str
    
    # Prices
    entry_price: float
    exit_price: Optional[float] = None
    sl_price: float = 0.0
    tp_price: float = 0.0
    
    # Results
    profit_loss: Optional[float] = None
    profit_pips: Optional[float] = None
    win: Optional[bool] = None
    
    # Context
    setup_type: str = ""
    confidence: float = 0.0
    signal_score: float = 0.0
    agi_score: float = 0.0
    swarm_score: float = 0.0
    
    # Market Conditions
    market_regime: str = ""
    volatility_regime: str = ""
    trend: str = ""
    hour_of_day: int = 0
    day_of_week: int = 0
    
    # Decision Making
    reasons: List[str] = None
    vetoes: List[str] = None
    
    # Post-Trade Analysis
    review_notes: str = ""
    lessons_learned: str = ""
    rating: int = 0  # 1-5 stars
    tags: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.vetoes is None:
            self.vetoes = []
        if self.tags is None:
            self.tags = []


class TradeJournal:
    """
    Auto Trade Journal System
    
    Automatically creates detailed journal entries for every trade.
    Generates markdown files with full context and analysis.
    """
    
    def __init__(self, journal_dir: str = "reports/trade_journal"):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        self.entries: Dict[str, JournalEntry] = {}
        self.index_file = self.journal_dir / "index.json"
        
        self._load_index()
        logger.info(f"Trade Journal initialized at {self.journal_dir}")
    
    def _load_index(self):
        """Load journal index"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    for entry_dict in data:
                        entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                        if entry_dict.get('reasons') is None:
                            entry_dict['reasons'] = []
                        if entry_dict.get('vetoes') is None:
                            entry_dict['vetoes'] = []
                        if entry_dict.get('tags') is None:
                            entry_dict['tags'] = []
                        entry = JournalEntry(**entry_dict)
                        self.entries[entry.trade_id] = entry
                logger.info(f"Loaded {len(self.entries)} journal entries")
            except Exception as e:
                logger.error(f"Error loading journal index: {e}")
    
    def _save_index(self):
        """Save journal index"""
        try:
            data = []
            for entry in self.entries.values():
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                data.append(entry_dict)
            
            with open(self.index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving journal index: {e}")
    
    def create_entry(self, entry: JournalEntry) -> str:
        """Create a new journal entry"""
        self.entries[entry.trade_id] = entry
        
        # Generate markdown file
        md_file = self._generate_markdown(entry)
        
        # Save index
        self._save_index()
        
        logger.info(f"Created journal entry: {entry.trade_id}")
        return md_file
    
    def update_entry(self, trade_id: str, **kwargs):
        """Update an existing journal entry"""
        if trade_id not in self.entries:
            logger.warning(f"Trade {trade_id} not found in journal")
            return
        
        entry = self.entries[trade_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        
        # Regenerate markdown
        self._generate_markdown(entry)
        
        # Save index
        self._save_index()
        
        logger.info(f"Updated journal entry: {trade_id}")
    
    def _generate_markdown(self, entry: JournalEntry) -> str:
        """Generate markdown journal entry"""
        # Create monthly directory structure
        month_dir = self.journal_dir / entry.timestamp.strftime("%Y-%m")
        month_dir.mkdir(exist_ok=True)
        
        # Generate filename
        filename = f"{entry.timestamp.strftime('%Y%m%d_%H%M%S')}_{entry.trade_id}.md"
        filepath = month_dir / filename
        
        # Generate markdown content
        md_content = self._format_entry(entry)
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(filepath)
    
    def _format_entry(self, entry: JournalEntry) -> str:
        """Format journal entry as markdown"""
        
        # Status emoji
        if entry.win is None:
            status = "⏳ OPEN"
        elif entry.win:
            status = "✅ WIN"
        else:
            status = "❌ LOSS"
        
        # Calculate R:R if closed
        rr_ratio = "N/A"
        if entry.exit_price and entry.sl_price and entry.tp_price:
            risk = abs(entry.entry_price - entry.sl_price)
            reward = abs(entry.tp_price - entry.entry_price)
            if risk > 0:
                rr_ratio = f"{reward/risk:.2f}"
        
        # Day name
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = days[entry.day_of_week] if entry.day_of_week < len(days) else "Unknown"
        
        # Build markdown
        md = f"""# Trade Journal Entry - {status}

**Trade ID:** {entry.trade_id}  
**Date:** {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Day:** {day_name}  
**Symbol:** {entry.symbol}

---

## 📊 Trade Summary

| Metric | Value |
|--------|-------|
| **Direction** | {entry.direction} |
| **Entry Price** | {entry.entry_price:.5f} |
| **Exit Price** | {entry.exit_price if entry.exit_price else "Open"} |
| **Stop Loss** | {entry.sl_price:.5f} |
| **Take Profit** | {entry.tp_price:.5f} |
| **Risk:Reward** | {rr_ratio} |

"""

        # Results section (if closed)
        if entry.win is not None:
            md += f"""
## 💰 Results

| Metric | Value |
|--------|-------|
| **Outcome** | {"WIN ✅" if entry.win else "LOSS ❌"} |
| **Profit/Loss** | ${entry.profit_loss:.2f} |
| **Pips** | {entry.profit_pips:.1f} |

"""

        # Setup and confidence
        md += f"""
## 🎯 Setup Analysis

**Setup Type:** {entry.setup_type}  
**Confidence:** {entry.confidence:.0f}%

### Layer Scores
- **Signal Layer:** {entry.signal_score:.0f}/100
- **AGI Layer:** {entry.agi_score:.0f}/100
- **Swarm Layer:** {entry.swarm_score:.0f}/100

"""

        # Market conditions
        md += f"""
## 🌍 Market Conditions

| Condition | Value |
|-----------|-------|
| **Market Regime** | {entry.market_regime} |
| **Volatility** | {entry.volatility_regime} |
| **Trend** | {entry.trend} |
| **Time** | {entry.hour_of_day:02d}:00 |

"""

        # Decision making
        if entry.reasons:
            md += "## ✅ Reasons to Trade\n\n"
            for reason in entry.reasons:
                md += f"- {reason}\n"
            md += "\n"
        
        if entry.vetoes:
            md += "## ⚠️ Warnings/Vetoes\n\n"
            for veto in entry.vetoes:
                md += f"- {veto}\n"
            md += "\n"
        
        # Review section
        if entry.review_notes or entry.lessons_learned:
            md += "## 📝 Post-Trade Review\n\n"
            
            if entry.rating > 0:
                stars = "⭐" * entry.rating
                md += f"**Rating:** {stars} ({entry.rating}/5)\n\n"
            
            if entry.review_notes:
                md += f"### Notes\n\n{entry.review_notes}\n\n"
            
            if entry.lessons_learned:
                md += f"### Lessons Learned\n\n{entry.lessons_learned}\n\n"
        
        # Tags
        if entry.tags:
            md += "## 🏷️ Tags\n\n"
            md += " | ".join([f"`{tag}`" for tag in entry.tags])
            md += "\n\n"
        
        # Footer
        md += "---\n\n"
        md += f"*Journal entry auto-generated by Genesis Trade Journal*  \n"
        md += f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return md
    
    def add_review(self, trade_id: str, notes: str = "", lessons: str = "", 
                   rating: int = 0, tags: List[str] = None):
        """Add post-trade review"""
        updates = {}
        if notes:
            updates['review_notes'] = notes
        if lessons:
            updates['lessons_learned'] = lessons
        if rating > 0:
            updates['rating'] = rating
        if tags:
            updates['tags'] = tags
        
        self.update_entry(trade_id, **updates)
    
    def get_entry(self, trade_id: str) -> Optional[JournalEntry]:
        """Get journal entry"""
        return self.entries.get(trade_id)
    
    def search_entries(self, **criteria) -> List[JournalEntry]:
        """Search journal entries by criteria"""
        results = []
        
        for entry in self.entries.values():
            match = True
            
            for key, value in criteria.items():
                if not hasattr(entry, key):
                    match = False
                    break
                
                entry_value = getattr(entry, key)
                
                # Handle different comparison types
                if isinstance(value, (list, tuple)):
                    if entry_value not in value:
                        match = False
                        break
                else:
                    if entry_value != value:
                        match = False
                        break
            
            if match:
                results.append(entry)
        
        return results
    
    def generate_summary(self, days: int = 30) -> str:
        """Generate journal summary"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [e for e in self.entries.values() if e.timestamp >= cutoff]
        
        wins = [e for e in recent if e.win == True]
        losses = [e for e in recent if e.win == False]
        
        summary = f"""
# Trade Journal Summary - Last {days} Days

**Total Entries:** {len(recent)}  
**Closed Trades:** {len(wins) + len(losses)}  
**Open Trades:** {len(recent) - len(wins) - len(losses)}

## Performance
- **Wins:** {len(wins)}
- **Losses:** {len(losses)}
- **Win Rate:** {len(wins)/(len(wins)+len(losses))*100:.1f}% if {len(wins)+len(losses)} > 0 else 'N/A'

## Top Setups
"""
        
        # Count setups
        setup_counts = {}
        for e in wins:
            setup = e.setup_type or "Unknown"
            if setup not in setup_counts:
                setup_counts[setup] = 0
            setup_counts[setup] += 1
        
        for setup, count in sorted(setup_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            summary += f"- {setup}: {count} wins\n"
        
        return summary


if __name__ == "__main__":
    # Test the journal
    journal = TradeJournal()
    
    # Create sample entry
    entry = JournalEntry(
        trade_id="TEST001",
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
        volatility_regime="NORMAL",
        trend="BULLISH",
        hour_of_day=14,
        day_of_week=2,
        reasons=["Strong pullback to support", "Confluence with 50 EMA", "High swarm consensus"],
        vetoes=[]
    )
    
    filepath = journal.create_entry(entry)
    print(f"✅ Created journal entry: {filepath}")
    
    # Simulate trade close
    journal.update_entry(
        "TEST001",
        exit_price=1.2680,
        profit_loss=30.0,
        profit_pips=30.0,
        win=True
    )
    
    # Add review
    journal.add_review(
        "TEST001",
        notes="Perfect execution. Entry was precise at support level.",
        lessons="Wait for pullback confirmation before entry.",
        rating=5,
        tags=["pullback", "perfect-entry", "support"]
    )
    
    print("✅ Trade journal system ready!")
