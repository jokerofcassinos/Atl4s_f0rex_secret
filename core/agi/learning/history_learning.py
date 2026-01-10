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
        self.is_dreaming = False
        self.experiences_learned = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        
        # Phase 4.2: Win Rate by Module
        self.source_performance = {} # {source_name: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'consecutive_losses': 0}}
        
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
        # For now, we utilize this for aggregated tracking via snapshots
        if isinstance(trade_feedback, list):
             # FUTURE: Compare snapshots to detect external closures (stop loss hits)
             return

        if trade_feedback.get('trade_executed'):
             logger.info(f"HISTORY LEARNING: Observed Trade Result: {trade_feedback}")
             self.experiences_learned += 1
             
             pnl = float(trade_feedback.get('profit', 0.0))
             self.total_pnl += pnl
             if pnl > 0: self.wins += 1
             elif pnl < 0: self.losses += 1

    def notify_trade_close(self, ticket: int, symbol: str, profit: float, reason: str, source: str = "UNKNOWN", confidence: float = 50.0):
        """
        Explicit callback from ExecutionEngine when a trade is closed.
        Stores the lesson in Holographic Memory.
        """
        self.experiences_learned += 1
        self.total_pnl += profit
        if profit > 0: self.wins += 1
        else: self.losses += 1
        
        # Phase 4.2/4.3: Update Source Metrics & Calibration
        if source not in self.source_performance:
            self.source_performance[source] = {
                'wins': 0, 'losses': 0, 'pnl': 0.0, 
                'consecutive_losses': 0,
                'total_confidence': 0.0, 'trade_count': 0
            }
            
        metrics = self.source_performance[source]
        metrics['pnl'] += profit
        metrics['total_confidence'] += confidence
        metrics['trade_count'] += 1
        
        if profit > 0:
            metrics['wins'] += 1
            metrics['consecutive_losses'] = 0
        else:
            metrics['losses'] += 1
            metrics['consecutive_losses'] += 1
        
        # Calculate Outcome Score (-1.0 to 1.0)
        outcome_score = np.tanh(profit / 2.0) 
        
        # Construct Lesson Context
        context = {
            'symbol': symbol,
            'reason': reason,
            'source': source, # Track which swarm/logic generated this
            'final_pnl': profit,
            'timestamp': time.time()
        }
        
        # Store in Memory
        if self.memory:
            self.memory.store_experience(
                feature_vector=self.memory.construct_hologram(context), # We need to construct vector from context!
                outcome=float(outcome_score),
                meta={**context, 'category': "trade_result", 'temporal': "short_term"}
            )
            
        logger.info(f"LEARNING: Trade {ticket} ({source}) Closed. Profit: ${profit:.2f}. Lesson Stored.")

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyzes learned patterns to determine overall system health/performance.
        Used by OmegaAGICore to switch modes (e.g. Critical Low Win Rate).
        """
        total_trades = self.wins + self.losses
        win_rate = 0.5 # Default neutral
        
        if total_trades > 0:
            win_rate = self.wins / total_trades
            
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_pnl': self.total_pnl,
            'experiences': self.experiences_learned
        }

    def get_calibration_bias(self, source: str) -> float:
        """
        Calculates Confidence Bias (Avg Conf - Real Win Rate).
        Positive = Overconfident. Negative = Underconfident.
        """
        stats = self.source_performance.get(source)
        if not stats or stats.get('trade_count', 0) < 5: return 0.0
        
        avg_conf = stats['total_confidence'] / stats['trade_count']
        win_rate = stats['wins'] / stats['trade_count']
        
        return avg_conf - (win_rate * 100.0)

    def get_source_performance(self, source: str) -> Dict[str, Any]:
        """
        Returns performance metrics for a specific source.
        """
        return self.source_performance.get(source, {'wins': 0, 'losses': 0, 'pnl': 0.0, 'consecutive_losses': 0})
        
    def get_all_performances(self) -> Dict[str, Dict]:
        """Returns the full performance map."""
        return self.source_performance
