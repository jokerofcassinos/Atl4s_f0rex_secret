import json

with open('reports/laplace_gbpusd_results.json', 'r') as f:
    data = json.load(f)

trades = data['trades']

# Filter losing trades (VSL_HIT or negative PnL)
losers = [t for t in trades if t['pnl_dollars'] < 0]

print(f"=== FORENSIC ANALYSIS: {len(losers)} Losing Trades ===\n")

for i, trade in enumerate(losers):
    tid = trade['id']
    print(f"--- LOSS #{i+1} (Trade ID: {tid}) ---")
    print(f"  Direction:     {trade['direction']}")
    print(f"  Entry Time:    {trade['entry_time']}")
    print(f"  Exit Time:     {trade['exit_time']}")
    print(f"  Exit Reason:   {trade['exit_reason']}")
    print(f"  PnL:           ${trade['pnl_dollars']:.2f}")
    print(f"  Setup:         {trade['signal_source']}")
    print(f"  Confidence:    {trade['confidence']}%")
    print(f"  Duration:      {trade['duration_minutes']} min")
    
    # Find nearby trades
    nearby_winners = [t for t in trades if t['id'] != tid and abs(t['id'] - tid) <= 5 and t['pnl_dollars'] > 0]
    
    if nearby_winners:
        print(f"  Nearby Winners ({len(nearby_winners)}):")
        for w in nearby_winners[:2]:
            print(f"    - ID {w['id']}: {w['direction']} | {w['signal_source']} | ${w['pnl_dollars']:.2f} | {w['entry_time']}")
    print()

# Group by signal_source
from collections import Counter
setup_counts = Counter(t['signal_source'] for t in losers)
print("=== LOSS BY SETUP ===")
for setup, count in setup_counts.most_common():
    total_loss = sum(t['pnl_dollars'] for t in losers if t['signal_source'] == setup)
    print(f"  {setup}: {count} trades | ${total_loss:.2f}")

# Group by hour
hour_losses = {}
for t in losers:
    hour = int(t['entry_time'].split(' ')[1].split(':')[0])
    if hour not in hour_losses:
        hour_losses[hour] = []
    hour_losses[hour].append(t)

print("\n=== LOSS BY HOUR ===")
for hour in sorted(hour_losses.keys()):
    trades_in_hour = hour_losses[hour]
    total_loss = sum(t['pnl_dollars'] for t in trades_in_hour)
    setups = [t['signal_source'] for t in trades_in_hour]
    print(f"  Hour {hour:02d}: {len(trades_in_hour)} trades | ${total_loss:.2f} | Setups: {set(setups)}")
