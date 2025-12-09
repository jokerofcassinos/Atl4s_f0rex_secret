import pandas as pd
import numpy as np

class MarkovChainPredictor:
    """
    Probabilistic State Machine using Markov Chains.
    Predicts the next market state based on historical transition probabilities.
    """
    def __init__(self, order=2):
        self.order = order # Lookback context (e.g., last 2 candles)

    def get_state(self, row):
        """Encodes a single candle into a state."""
        # Simple encoding: Bullish (1) or Bearish (0)
        # Enhanced encoding: Bullish Volatile (2), Bullish Quiet (1), Bearish Quiet (-1), Bearish Volatile (-2)
        ret = row['close'] - row['open']
        atr = row.get('ATR', 0)
        
        # If ATR is not available, just use sign
        if atr == 0:
            return 1 if ret > 0 else -1

        threshold = 0.5 * atr
        if ret > threshold: return "StrongBull"
        elif ret > 0: return "WeakBull"
        elif ret < -threshold: return "StrongBear"
        else: return "WeakBear"

    def train_transition_matrix(self, df: pd.DataFrame):
        """
        Builds a transition matrix from historical data.
        """
        if 'ATR' not in df.columns:
            # Fallback if ATR not calculated yet, assume it's passed or handled outside.
            # Ideally AnalysisEngine adds ATR before calling this.
            return {}

        states = df.apply(self.get_state, axis=1)
        
        # Create sequences of length 'order' -> next_state
        transitions = {}
        
        for i in range(len(states) - self.order):
            current_sequence = tuple(states.iloc[i : i+self.order].values)
            next_state = states.iloc[i + self.order]
            
            if current_sequence not in transitions:
                transitions[current_sequence] = {}
            
            if next_state not in transitions[current_sequence]:
                transitions[current_sequence][next_state] = 0
            
            transitions[current_sequence][next_state] += 1
            
        # Normalize to probabilities
        probabilities = {}
        for seq, counts in transitions.items():
            total = sum(counts.values())
            probabilities[seq] = {state: count/total for state, count in counts.items()}
            
        return probabilities

    def predict_next(self, df: pd.DataFrame, matrix):
        """
        Predicts probabilities for the next candle based on the last 'order' candles.
        """
        if matrix is None or not matrix:
            return None

        # Get current sequence
        current_states = df.apply(self.get_state, axis=1).tail(self.order)
        current_seq = tuple(current_states.values)
        
        if current_seq in matrix:
            return matrix[current_seq]
        else:
            return None # Unknown sequence
