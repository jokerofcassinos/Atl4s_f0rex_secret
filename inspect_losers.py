import json
import pandas as pd

try:
    with open('d:/Atl4s-Forex/reports/laplace_gbpusd_results.json', 'r') as f:
        data = json.load(f)
        
    trades = data.get('trades', [])
    df = pd.DataFrame(trades)
    
    losers = df[df['pnl_dollars'] < 0]
    
    print(f"Total Losers: {len(losers)}")
    print("-" * 60)
    for index, row in losers.iterrows():
        print(f"Trade #{row['id']} | {row['entry_time']} | {row['direction']} | {row['signal_source']} | PnL: ${row['pnl_dollars']:.2f}")
except Exception as e:
    print(f"Error: {e}")
