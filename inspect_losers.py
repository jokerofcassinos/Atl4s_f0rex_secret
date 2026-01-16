
import json

file_path = 'reports/laplace_gbpusd_results.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    trades = data.get('trades', [])
    losers = [t for t in trades if t['pnl_dollars'] < 0]

    print(f"Found {len(losers)} losing trades.")

    for t in losers:
        print(f"ID: {t['id']} | Dir: {t['direction']} | PnL: ${t['pnl_dollars']:.2f} | Reason: {t['exit_reason']} | Entry: {t['entry_time']} | Source: {t['signal_source']}")

except Exception as e:
    print(f"Error: {e}")
