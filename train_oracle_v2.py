import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OracleTrainer")

import sys
import subprocess

DATA_FILE = "data/training/live_trades.csv"
MODEL_PATH = "core/agi/training/oracle.pkl"
SCALER_PATH = "core/agi/training/scaler.pkl"

def run_auto_training():
    """
    Spawns the training process in a separate background process.
    """
    try:
        # Run this file as a script
        subprocess.Popen([sys.executable, __file__], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
        logger.info("Auto-Training Subprocess Spawned.")
    except Exception as e:
        logger.error(f"Failed to spawn auto-training: {e}")

def train_oracle():
    """
    Retrains the Tier 4 Neural Oracle using collected live trade data.
    """
    if not os.path.exists(DATA_FILE):
        logger.warning(f"No training data found at {DATA_FILE}. Skipping training.")
        return

    logger.info("loading training data...")
    try:
        df = pd.read_csv(DATA_FILE)
        if len(df) < 50:
            logger.info(f"Not enough data to train (Rows: {len(df)} < 50). Waiting for more trades.")
            return
            
        # Parse PnL to Binary Outcome (1=Win, 0=Loss)
        # We consider > $0.50 a Win (Spread/Comm coverage)
        df['target'] = df['pnl'].apply(lambda x: 1 if x > 0.50 else 0)
        
        # Features: [RSI, ATR_Ratio, Dist_SMA200, BB_Width, MACD_Hist, Confidence, Direction]
        # Currently, our CSV layout is:
        # timestamp, ticket, pnl, direction, entry_price, exit_price, setup, confidence
        # WE ARE MISSING TECHNICAL FEATURES IN THE CSV!
        # The Implementation Plan noted we might need to exact these. 
        # For V1 of this feedback loop, we will train on: [Confidence, DirectionInt, EntryPrice(Norm)]
        
        # NOTE: Ideally, record_trade should have logged the features.
        # Since it didn't, we will use a simplified feature set for this "Patch".
        # Future TODO: Update record_trade to dump the full feature vector.
        
        df['dir_int'] = df['direction'].apply(lambda x: 1 if x == "BUY" else -1)
        df['conf_norm'] = df['confidence'] / 100.0
        
        # Inputs: Confidence, Direction
        X = df[['conf_norm', 'dir_int']]
        y = df['target']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model: MLP (Lightweight)
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        score = clf.score(X_test_scaled, y_test)
        logger.info(f"Training Complete. Test Accuracy: {score:.2%}")
        
        # Save
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Training Failed: {e}")

if __name__ == "__main__":
    train_oracle()
