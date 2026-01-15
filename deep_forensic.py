"""
Forensic Log Analyzer - Extracts decision structures for specific trades
"""
import json
import re
from datetime import datetime, timedelta

# Load trade data
with open('reports/laplace_gbpusd_results.json', 'r') as f:
    data = json.load(f)

trades = data['trades']
losers = [t for t in trades if t['pnl_dollars'] < 0]
winners = [t for t in trades if t['pnl_dollars'] > 0]

# Read the full log file
try:
    with open('laplace_backtest.log', 'r', encoding='utf-8', errors='ignore') as f:
        log_lines = f.readlines()
except:
    log_lines = []
    print("WARNING: Could not read log file. Using JSON data only.\n")

print("=" * 70)
print("DEEP FORENSIC ANALYSIS: Decision Structure Comparison")
print("=" * 70)

# Analyze first 5 losers in detail
for i, loser in enumerate(losers[:5]):
    print(f"\n{'='*70}")
    print(f"LOSS #{i+1}: Trade ID {loser['id']}")
    print(f"{'='*70}")
    print(f"Entry: {loser['entry_time']} | Exit: {loser['exit_time']}")
    print(f"Direction: {loser['direction']} | Setup: {loser['signal_source']}")
    print(f"PnL: ${loser['pnl_dollars']:.2f} | Duration: {loser['duration_minutes']} min")
    
    # Parse entry time
    entry_dt = datetime.strptime(loser['entry_time'], "%Y-%m-%d %H:%M:%S")
    
    # Find nearest winners (same day, within 2 hours)
    nearby_winners = []
    for w in winners:
        w_entry = datetime.strptime(w['entry_time'], "%Y-%m-%d %H:%M:%S")
        time_diff = abs((entry_dt - w_entry).total_seconds() / 3600)
        if time_diff < 2 and w['id'] != loser['id']:
            nearby_winners.append((w, time_diff))
    
    nearby_winners.sort(key=lambda x: x[1])
    
    if nearby_winners:
        print(f"\n--- Nearest Winner (for comparison) ---")
        winner, tdiff = nearby_winners[0]
        print(f"Entry: {winner['entry_time']} (Δ {tdiff:.1f} hours)")
        print(f"Direction: {winner['direction']} | Setup: {winner['signal_source']}")
        print(f"PnL: ${winner['pnl_dollars']:.2f} | Duration: {winner['duration_minutes']} min")
        
        # KEY COMPARISON
        print(f"\n--- COMPARISON ---")
        print(f"  LOSER Duration:  {loser['duration_minutes']} min")
        print(f"  WINNER Duration: {winner['duration_minutes']} min")
        print(f"  Difference: Winner lasted {winner['duration_minutes'] - loser['duration_minutes']} min longer")
        
        # Hour comparison
        loser_hour = entry_dt.hour
        winner_dt = datetime.strptime(winner['entry_time'], "%Y-%m-%d %H:%M:%S")
        winner_hour = winner_dt.hour
        print(f"  LOSER Hour:  {loser_hour:02d}:00")
        print(f"  WINNER Hour: {winner_hour:02d}:00")
        
        if loser_hour in [7, 8, 9, 13]:
            print(f"  ⚠️ LOSER entered during HIGH-RISK hour!")
        if winner_hour not in [7, 8, 9, 13]:
            print(f"  ✅ WINNER avoided high-risk hours")
    else:
        print("  No nearby winners found for comparison")

# Summary statistics
print("\n" + "=" * 70)
print("STATISTICAL PATTERNS")
print("=" * 70)

# Hour distribution
loser_hours = [datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S").hour for t in losers]
winner_hours = [datetime.strptime(t['entry_time'], "%Y-%m-%d %H:%M:%S").hour for t in winners]

print("\nLoser entries by hour:")
from collections import Counter
loser_hour_counts = Counter(loser_hours)
for hour in sorted(loser_hour_counts.keys()):
    print(f"  Hour {hour:02d}: {'█' * loser_hour_counts[hour]} ({loser_hour_counts[hour]})")

print("\nWinner entries by hour (top 5):")
winner_hour_counts = Counter(winner_hours)
for hour, count in winner_hour_counts.most_common(5):
    print(f"  Hour {hour:02d}: {'█' * min(count, 30)} ({count})")

# Duration comparison
loser_durations = [t['duration_minutes'] for t in losers]
winner_durations = [t['duration_minutes'] for t in winners]

print(f"\nDuration Analysis:")
print(f"  Avg Loser Duration:  {sum(loser_durations)/len(loser_durations):.1f} min")
print(f"  Avg Winner Duration: {sum(winner_durations)/len(winner_durations):.1f} min")
print(f"  Losers die {sum(winner_durations)/len(winner_durations) - sum(loser_durations)/len(loser_durations):.1f} min faster on average")

# Setup comparison
loser_setups = Counter(t['signal_source'] for t in losers)
winner_setups = Counter(t['signal_source'] for t in winners)

print(f"\nSetup Win Rates:")
all_setups = set(loser_setups.keys()) | set(winner_setups.keys())
for setup in all_setups:
    wins = winner_setups.get(setup, 0)
    losses = loser_setups.get(setup, 0)
    total = wins + losses
    if total > 0:
        wr = (wins / total) * 100
        print(f"  {setup}: {wins}W / {losses}L = {wr:.1f}% WR")
