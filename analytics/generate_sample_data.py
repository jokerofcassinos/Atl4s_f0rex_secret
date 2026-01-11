"""
Generate Sample Trades for ML Optimizer Demo

Creates realistic trade data to demonstrate analytics and optimization
"""

import sys
import random
from datetime import datetime, timedelta
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.trade_analyzer import TradeAnalyzer, TradeRecord

def generate_sample_trades(count: int = 50):
    """Generate realistic sample trades"""
    analyzer = TradeAnalyzer(db_path="data/genesis_trades.json")
    
    # Clear existing
    analyzer.trades = []
    
    base_time = datetime.now() - timedelta(days=30)
    
    setups = {
        "GENESIS_PULLBACK": 0.80,  # 80% WR
        "GENESIS_BREAKOUT": 0.65,  # 65% WR
        "GENESIS_REVERSAL": 0.55,  # 55% WR
        "GENESIS_CONTINUATION": 0.75  # 75% WR
    }
    
    # Good hours: 13-16 (London-NY overlap)
    # Bad hours: 20-23, 0-6
    
    for i in range(count):
        # Random time within 30 days
        days_ago = random.randint(0, 29)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        
        timestamp = base_time + timedelta(days=days_ago, hours=hour, minutes=minute)
        
        # Choose setup
        setup = random.choice(list(setups.keys()))
        base_wr = setups[setup]
        
        # Adjust WR based on time
        if 13 <= hour <= 16:
            wr = base_wr + 0.10  # Better during good hours
        elif hour >= 20 or hour <= 6:
            wr = base_wr - 0.15  # Worse during bad hours
        else:
            wr = base_wr
        
        # Determine win/loss
        win = random.random() < wr
        
        # Generate scores
        if win:
            signal_score = random.randint(50, 80)
            agi_score = random.randint(60, 85)
            swarm_score = random.randint(55, 80)
            confidence = random.randint(65, 90)
        else:
            signal_score = random.randint(20, 60)
            agi_score = random.randint(30, 70)
            swarm_score = random.randint(25, 65)
            confidence = random.randint(40, 75)
        
        # Generate price action
        direction = random.choice(["BUY", "SELL"])
        entry_price = 1.2650 + random.uniform(-0.0100, 0.0100)
        
        if win:
            profit_pips = random.uniform(20, 60)
            profit_loss = profit_pips * 1.0  # $1 per pip
        else:
            profit_pips = -random.uniform(15, 40)
            profit_loss = profit_pips * 1.0
        
        if direction == "BUY":
            exit_price = entry_price + (profit_pips * 0.0001)
        else:
            exit_price = entry_price - (profit_pips * 0.0001)
        
        sl_price = entry_price - (30 * 0.0001) if direction == "BUY" else entry_price + (30 * 0.0001)
        tp_price = entry_price + (50 * 0.0001) if direction == "BUY" else entry_price - (50 * 0.0001)
        
        trade = TradeRecord(
            trade_id=f"DEMO_{timestamp.strftime('%Y%m%d_%H%M%S')}_{i}",
            timestamp=timestamp,
            symbol="GBPUSD",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl_price=sl_price,
            tp_price=tp_price,
            profit_loss=round(profit_loss, 2),
            profit_pips=round(profit_pips, 1),
            win=win,
            setup_type=setup,
            confidence=confidence,
            signal_score=signal_score,
            agi_score=agi_score,
            swarm_score=swarm_score,
            market_regime=random.choice(["TRENDING", "RANGING"]),
            volatility_regime=random.choice(["LOW", "NORMAL", "HIGH"]),
            trend=random.choice(["BULLISH", "BEARISH", "NEUTRAL"]),
            session=random.choice(["LONDON", "NY", "ASIA"]),
            reasons=[f"Signal from {setup}", f"Confidence {confidence}%"],
            vetoes=[],
            swarm_votes={},
            hour_of_day=hour,
            day_of_week=timestamp.weekday(),
            duration_minutes=random.randint(30, 240),
            risk_reward_ratio=random.uniform(1.5, 3.0)
        )
        
        analyzer.trades.append(trade)
    
    analyzer._save_history()
    
    wins = len([t for t in analyzer.trades if t.win])
    print(f"✅ Generated {count} sample trades")
    print(f"   Win Rate: {wins/count*100:.1f}%")
    print(f"   Database: {analyzer.db_path}")
    
    return analyzer

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    generate_sample_trades(count)
