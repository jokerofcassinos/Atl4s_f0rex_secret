"""
AGI Ultra: History Learning Engine (Active Learning Module)

Responsibilities:
1. Ingest historical market data (CSV/DataFrame).
2. "Dream": Replay history as if it were live.
3. "Learn": Compare AGI Intuition vs Actual Outcome (Future Price).
4. "Reinforce": Store the experience in Holographic Memory.
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Optional
from core.memory.holographic import HolographicMemory
from core.agi.omni_cortex import OmniCortex

logger = logging.getLogger("HistoryLearning")

class HistoryLearningEngine:
    def __init__(self, memory: HolographicMemory):
        self.memory = memory
        self.is_dreaming = False
        self.experiences_learned = 0
        
    def dream_cycle(self, data: pd.DataFrame, batch_size: int = 1000):
        """
        Runs a Dream Cycle on historical data.
        
        Args:
            data: DataFrame with 'close', 'high', 'low' columns.
            batch_size: Number of ticks to process per dream session.
        """
        logger.info(f"STARTING DREAM CYCLE: {len(data)} candles available.")
        self.is_dreaming = True
        
        # We need at least lookahead + lookback
        lookahead = 50 # How far future to check for outcome
        lookback = 100 # How much history needed for context
        
        if len(data) < lookback + lookahead:
            logger.warning("Not enough data to dream.")
            return

        # Iterate through the data
        # We skip the first 'lookback' to have context
        # We stop 'lookahead' before end to have a future truth
        
        count = 0
        start_time = time.time()
        
        # Iterate efficiently
        closes = data['close'].values
        
        # We simulate "Random Access Dreams" - Picking random points is better for non-linear learning
        # But for trends, sequential is easier to implement first.
        
        indices = np.linspace(lookback, len(data) - lookahead - 1, num=batch_size, dtype=int)
        
        for idx in indices:
            current_price = closes[idx]
            future_price = closes[idx + lookahead]
            
            # 1. Determine the "Truth" (Outcome)
            # Did price go up or down significantly?
            roi = (future_price - current_price) / current_price
            
            # Normalize ROI to -1.0 to 1.0 range (Sigmoid-ish)
            # e.g. 1% move -> 0.5 score
            outcome_score = np.tanh(roi * 100.0) 
            
            # 2. Extract Context (The "State")
            # We need features similar to what OmniCortex perceives
            # For simplicity, we use raw price window + chaotic metric proxy
            window = closes[idx-50 : idx] # Last 50 candles
            
            # Calculate simple Fisher/Entropy proxy here or use helper
            # We can use standard deviation relative to price as volatility proxy
            vol = np.std(window) / np.mean(window)
            
            context = {
                'last_price': float(current_price),
                'volatility': float(vol),
                # Encode the shape of the window? 
                # Holographic memory encodes the values efficiently
                'prices': window.tolist() 
            }
            
            # 3. Store in Holographic Memory
            # We treat this as a "Pattern" memory
            self.memory.store_experience(
                context=context,
                outcome=float(outcome_score),
                category="pattern",
                temporal_level="long_term" # Store directly to long-term for history
            )
            
            self.experiences_learned += 1
            count += 1
            
        elapsed = time.time() - start_time
        logger.info(f"DREAM COMPLETE: Learned {count} experiences in {elapsed:.2f}s.")
        self.is_dreaming = False
        
    def get_status(self):
        return f"Learned: {self.experiences_learned} | Status: {'Dreaming' if self.is_dreaming else 'Awake'}"

    def update_active_trades(self, trade_feedback: Dict[str, Any]):
        """
        Receives live trade feedback (PnL, outcomes) and updates the internal model.
        Called by main.py after each tick.
        """
        # For now, we just log it or store it if we want 'online learning'
        # Ideally, this would update the Holographic Memory with short-term reinforcement
        
        # FIX: Handle List input (Active Trades Snapshot) vs Dict input (Trade Result)
        if isinstance(trade_feedback, list):
             # Just a snapshot of open trades, not a result.
             return

        if trade_feedback.get('trade_executed'):
             logger.info(f"HISTORY LEARNING: Observed Trade Result: {trade_feedback}")
             self.experiences_learned += 1
