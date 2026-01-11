import pandas as pd
import config
import os

cache_file = os.path.join(config.CACHE_DIR, f"{config.SYMBOL}_{config.TIMEFRAME}.parquet")
if os.path.exists(cache_file):
    df = pd.read_parquet(cache_file)
    print("Columns:", df.columns)
    print("Head:\n", df.head())
    print("Type of close:", type(df['close'].iloc[-1]))
    print("Value of close:", df['close'].iloc[-1])
else:
    print("Cache not found.")
