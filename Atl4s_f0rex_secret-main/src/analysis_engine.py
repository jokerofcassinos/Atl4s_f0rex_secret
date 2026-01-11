import pandas as pd
import logging
from .technical_analysis import TechnicalAnalysis
from .quantum_math import QuantumMath
from .price_action import PriceAction
from .markov_chain import MarkovChainPredictor

logger = logging.getLogger("AnalysisEngine")

class AnalysisEngine:
    def __init__(self):
        self.markov = MarkovChainPredictor(order=3)
        self.trained = False

    def analyze(self, df: pd.DataFrame):
        """
        Runs all analysis modules on the dataframe.
        data: DataFrame with open, high, low, close, volume.
        """
        if df.empty:
            return df

        # 1. Technical Analysis
        df['RSI'] = TechnicalAnalysis.rsi(df['close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = TechnicalAnalysis.bollinger_bands(df['close'])
        df['ATR'] = TechnicalAnalysis.atr(df['high'], df['low'], df['close'])
        df['EMA_50'] = TechnicalAnalysis.ema(df['close'], span=50) # Trend filter

        # 2. Quantum / Advanced Math
        df['Entropy'] = QuantumMath.calculate_entropy(df['close'])
        # df['Hurst'] = QuantumMath.calculate_hurst_exponent(df['close']) # Disable if slow
        df['Kalman'] = QuantumMath.kalman_filter(df['close'])
        df['Z_Score'] = QuantumMath.z_score(df['close'])

        # 3. Price Action
        df['FVG'] = PriceAction.detect_fvg(df)
        
        # 4. Probabilistic Prediction (Markov)
        # Train on history (first time or periodically)
        if not self.trained:
             self.transition_matrix = self.markov.train_transition_matrix(df)
             self.trained = True
        
        # Predict next likely move
        prediction = self.markov.predict_next(df, self.transition_matrix)
        # Store prediction in a separate way or just log it for Strategy to use
        # For now, we attach it to the latest row-index (or handled in StrategyCore)
        # Let's return both the enriched DF and the prediction
        return df, prediction
