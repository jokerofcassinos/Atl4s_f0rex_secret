
import os
import sys
import json
import pandas as pd
import numpy as np
import talib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
DATA_PATH = r"D:\Atl4s-Forex\data\GBPUSD_M1.csv"
TRADES_PATH = r"D:\Atl4s-Forex\reports\laplace_gbpusd_results.json"
MODEL_PATH = r"D:\Atl4s-Forex\core\agi\training\oracle.pkl"
SCALER_PATH = r"D:\Atl4s-Forex\core\agi\training\scaler.pkl"

def load_data():
    """Loads M1 data and resamples to M5"""
    print(f"Loading data from {DATA_PATH}...")
    # Load M1
    try:
        df_m1 = pd.read_csv(DATA_PATH)
        # CSV has: time,open,high,low,close,volume
        df_m1.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        df_m1['time'] = pd.to_datetime(df_m1['time'])
        df_m1.set_index('time', inplace=True)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Resample to M5
    print("Resampling to M5...")
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum'
    }
    df_m5 = df_m1.resample('5min').agg(ohlc_dict).dropna()
    return df_m5

def compute_features(df):
    """Computes technical indicators for features"""
    print("Computing indicators...")
    # RSI
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    # ATR Ratio (Volatility)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['ATR_Ratio'] = df['ATR'] / df['close']
    # SMA 200 Distance
    df['SMA200'] = talib.SMA(df['close'], timeperiod=200)
    df['Dist_SMA200'] = (df['close'] - df['SMA200']) / df['SMA200']
    # Bollinger Width
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    df['BB_Width'] = (upper - lower) / middle
    # MACD
    macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_Hist'] = hist
    
    # Clean NaN
    df.dropna(inplace=True)
    return df

def build_dataset(df_m5, trades):
    """Matches trades to features"""
    X = []
    y = []
    
    print(f"Matching {len(trades)} trades to data...")
    for trade in trades:
        entry_time = pd.to_datetime(trade['entry_time'])
        direction = 1 if trade['direction'] == 'BUY' else -1
        pnl = trade['pnl_pips']
        outcome = 1 if pnl > 0 else 0 # 1 = Win, 0 = Loss
        
        # Find candle at or before entry
        # Ideally, lookback 1 candle to avoid look-ahead bias if entry is intra-candle
        # But M5 close is known only after candle closes.
        # Entry time is precise.
        # We need the candle that CLOSED before entry or the current developing candle?
        # Standard: Use the candle that just closed before entry.
        try:
            # Get location of index <= entry_time
            idx = df_m5.index.get_indexer([entry_time], method='pad')[0]
            candle_time = df_m5.index[idx]
            
            # Use the feature vector from that candle
            row = df_m5.iloc[idx]
            
            # Features: [RSI, ATR, Dist_SMA, BB_Width, MACD_Hist, Confidence, Direction]
            # Normalize confidence 0-1
            conf = trade.get('confidence', 50) / 100.0
            
            features = [
                row['RSI'],
                row['ATR_Ratio'],
                row['Dist_SMA200'],
                row['BB_Width'],
                row['MACD_Hist'],
                conf,
                direction
            ]
            
            X.append(features)
            y.append(outcome)
            
        except Exception as e:
            # print(f"Skipping trade at {entry_time}: {e}")
            continue
            
    return np.array(X), np.array(y)

def main():
    # 1. Load Data
    df_m5 = load_data()
    if df_m5 is None: return
    
    # 2. Compute Features
    df_m5 = compute_features(df_m5)
    
    # 3. Load Trades
    with open(TRADES_PATH, 'r') as f:
        report = json.load(f)
    trades = report.get('trades', [])
    
    # 4. Build Dataset
    X, y = build_dataset(df_m5, trades)
    print(f"Dataset Created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Win Rate in Data: {np.mean(y):.2%}")
    
    if len(X) < 10:
        print("Not enough data to train.")
        return

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train MLP
    print("Training Neural Oracle (MLP)...")
    clf = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=2000, random_state=42, activation='relu', solver='adam')
    clf.fit(X_train_scaled, y_train)
    
    # 8. Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc:.2%}")
    print(classification_report(y_test, y_pred))
    
    # 9. Save
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Optimization Complete.")

if __name__ == "__main__":
    main()
