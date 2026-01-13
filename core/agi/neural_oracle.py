
import os
import joblib
import pandas as pd
import numpy as np
import talib
import logging
from typing import Optional

logger = logging.getLogger("NeuralOracle")

class NeuralOracle:
    """
    Tier 4 Neural Oracle - Predictive Filtering
    Uses a trained MLP model to filter signals based on high-probability outcomes.
    """
    def __init__(self, 
                 model_path: str = r"D:\Atl4s-Forex\core\agi\training\oracle.pkl",
                 scaler_path: str = r"D:\Atl4s-Forex\core\agi\training\scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self._loaded = True
                logger.info("Neural Oracle: Model and Scaler loaded successfully.")
            else:
                logger.warning(f"Neural Oracle: Model files not found at {self.model_path}. Running in bypass mode.")
        except Exception as e:
            logger.error(f"Neural Oracle: Error loading model: {e}")

    def extract_features(self, df_m5: pd.DataFrame, direction: str, confidence: float) -> Optional[np.ndarray]:
        """
        Extract Features matching train_oracle.py:
        [RSI, ATR_Ratio, Dist_SMA200, BB_Width, MACD_Hist, Confidence, Direction]
        """
        if df_m5 is None or len(df_m5) < 200:
            return None

        try:
            # 1. RSI
            rsi = talib.RSI(df_m5['close'], timeperiod=14).iloc[-1]
            
            # 2. ATR Ratio
            atr = talib.ATR(df_m5['high'], df_m5['low'], df_m5['close'], timeperiod=14).iloc[-1]
            atr_ratio = atr / df_m5['close'].iloc[-1]
            
            # 3. SMA 200 Distance
            sma200 = talib.SMA(df_m5['close'], timeperiod=200).iloc[-1]
            dist_sma200 = (df_m5['close'].iloc[-1] - sma200) / sma200
            
            # 4. BB Width
            upper, middle, lower = talib.BBANDS(df_m5['close'], timeperiod=20)
            bb_width = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
            
            # 5. MACD Hist
            macd, signal, hist = talib.MACD(df_m5['close'])
            macd_hist = hist.iloc[-1]
            
            # 6. Confidence (0-1)
            conf_norm = confidence / 100.0
            
            # 7. Direction (1 for BUY, -1 for SELL)
            dir_val = 1 if direction == "BUY" else -1
            
            features = [rsi, atr_ratio, dist_sma200, bb_width, macd_hist, conf_norm, dir_val]
            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Neural Oracle: Feature extraction failed: {e}")
            return None

    def predict_win_probability(self, df_m5: pd.DataFrame, direction: str, confidence: float) -> float:
        """
        Returns probability of a Winning trade (Outcome 1).
        In bypass mode, returns 1.0 (allow everything).
        """
        if not self._loaded:
            return 1.0

        features = self.extract_features(df_m5, direction, confidence)
        if features is None:
            return 0.5 

        try:
            # Scale
            features_scaled = self.scaler.transform(features)
            
            # Predict Probabilities
            # probs is [[prob_class_0, prob_class_1]]
            probs = self.model.predict_proba(features_scaled)[0]
            win_prob = probs[1]
            return win_prob
        except Exception as e:
            logger.error(f"Neural Oracle: Prediction failed: {e}")
            return 0.5
