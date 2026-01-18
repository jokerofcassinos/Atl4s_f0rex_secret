import json
import pandas as pd
import glob
import os

try:
    # Find latest history report
    list_of_files = glob.glob('reports/history_*_results.json') 
    if not list_of_files:
        print("No historical reports found in reports/")
        exit()
        
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"🕵️ Inspecting Report: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    trades = data.get('summary', {}).get('trades', []) # Structure might differ in export_results vs internal. 
    # Check if 'trades' is top level or inside summary. run_laplace_backtest.py export_results puts it in root usually?
    # Let's check run_laplace_backtest.py export_results method.
    # Actually, export_results in engine.py:
    # 'trades': [t.to_dict() for t in result.trades], is at ROOT level.
    
    trades = data.get('trades', [])
    if not trades:
         print("No trades found in json.")
         exit()

    df = pd.DataFrame(trades)
    
    losers = df[df['pnl_dollars'] < 0]
    
    print(f"\n🔴 Total Losers: {len(losers)}")
    print("-" * 100)
    print(f"{'ID':<6} | {'Date':<20} | {'Type':<6} | {'Source':<20} | {'PnL':<10} | {'Exit':<10}")
    print("-" * 100)
    
    for index, row in losers.iterrows():
        print(f"#{row['id']:<5} | {row['entry_time']:<20} | {row['direction']:<6} | {row['signal_source']:<20} | ${row['pnl_dollars']:<9.2f} | {row['exit_reason']}")
        
    print("-" * 100)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
