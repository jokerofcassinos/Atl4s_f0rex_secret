# core/risk/entropy_harvester.py

import logging
from typing import Dict, Optional, Tuple
import asyncio
import numpy as np
import pandas as pd

logger = logging.getLogger("EntropyHarvester")

class EntropyHarvester:
    """
    Phase 26: Chaos Harvesting (Gamma Scalping).
    
    Operates on Quantum Locked pairs.
    Uses market volatility (Entropy) to generate 'micro-profits' by
    oscillating exposure (Scalping the Noise).
    
    Goal: Pay off the 'Locked PnL' (Debt) until it reaches zero, 
    then close the entire structure.
    """
    
    def __init__(self, bridge, quantum_hedger):
        self.bridge = bridge
        self.hedger = quantum_hedger
        self.min_harvest_profit = 5.0 
        self.last_entropy = 0.0
        
    def calculate_shannon_entropy(self, price_series: pd.Series, bins: int = 20) -> float:
        """
        Calculates Shannon Entropy (Chaos Level) of the price returns.
        High Entropy = Chaos / Noise / Range.
        Low Entropy = Order / Trend.
        """
        if len(price_series) < 50: return 0.0
        
        # 1. Calculate Log Returns
        returns = np.log(price_series / price_series.shift(1)).dropna()
        
        # 2. Histogram (Probability Distribution)
        # We discretize returns into bins to find p(x)
        # CRITICAL: Use fixed range (-0.5% to +0.5%) to prevent auto-zooming on tight trends
        # If we zoom in, a tight trend looks "Uniform" (High Entropy).
        # We want to measure "Spread relative to Global Expectation".
        try:
            hist, _ = np.histogram(returns, bins=bins, range=(-0.005, 0.005), density=True)
            
            # 3. Normalize to get probabilities
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                probs = hist / hist_sum
                probs = probs[probs > 0] # Remove zeros for log
                
                # 4. Shannon Entropy Formula: H = -sum(p * log(p))
                entropy_val = -np.sum(probs * np.log2(probs))
            else:
                entropy_val = 0.0
            
            # Normalize Entropy (0 to 1 scale roughly related to max entropy)
            # Max entropy for N bins is log2(N)
            max_entropy = np.log2(bins) if bins > 1 else 1.0
            normalized_entropy = entropy_val / max_entropy
            
            return normalized_entropy
        except Exception:
            return 0.0
        
    async def harvest_lock(self, original_ticket: int, current_tick: Dict, agi_context: Dict):
        """
        Attempts to harvest profits from a locked pair.
        """
        if not self.hedger.is_locked(original_ticket): return
        
        # 1. Safety Check: ONLY Harvest in Ranges
        # If market is trending, unlocking a leg is suicide.
        if not agi_context: return
        ra = agi_context.get('range_analysis', {})
        if ra.get('status') != 'RANGING':
            # logger.debug("HARVESTER: Market not ranging. Gamma Scalping unsafe.")
            return

        # 2. Get Lock Data
        hedge_ticket = self.hedger.locked_pairs.get(original_ticket)
        meta = self.hedger.lock_metadata.get(original_ticket)
        if not hedge_ticket or not meta: return
        
        symbol = meta.get('symbol')
        locked_pnl = meta.get('locked_pnl', -999.0)
        
        # 3. Calculate Potential Scalp
        # We need current profit of both legs
        # Ideally we fetch from bridge or cache.
        # Simplified: We rely on the tick's 'trades_json' passing through or fetch explicitly.
        # For this logic prototype, we assume we can see the profits.
        
        # LOGIC:
        # If one leg is VERY GREEN (e.g. +$20) and we are at Range Edge.
        # We CLOSE that leg (Bank +$20).
        # We immediately place a LIMIT ORDER to re-open that leg at a better price.
        # OR simpler: We just Bank it and accept we are now exposed (Delta != 0).
        # But we are in a Range, so we expect reversion.
        
        # Safety: We avoid "Legging Out" completely.
        # We implement "Virtual credits" for now to show the user the power without risking the account immediately 
        # unless he approves "Active Harvesting".
        
        # Let's Log the Opportunity
        proximity = ra.get('proximity', 'MID')
        
        if proximity == 'HIGH': # Resistance
            # Verify if BUY leg is Green
            logger.info(f"☢️ ENTROPY: {symbol} at Range HIGH. Opportunity to Harvest BUY leg (Profit) and Re-Short.")
            
        elif proximity == 'LOW': # Support
            # Verify if SELL leg is Green
            logger.info(f"☢️ ENTROPY: {symbol} at Range LOW. Opportunity to Harvest SELL leg (Profit) and Re-Buy.")
            
    def apply_harvest_credit(self, original_ticket: int, profit: float):
        """
        Reduces the tracked debt of a lock.
        """
        if original_ticket in self.hedger.lock_metadata:
            self.hedger.lock_metadata[original_ticket]['locked_pnl'] += profit
            new_bal = self.hedger.lock_metadata[original_ticket]['locked_pnl']
            logger.info(f"🔋 DEFICIT REDUCED: Used ${profit:.2f} to pay debt. Remaining: ${new_bal:.2f}")
            
            if new_bal >= 0:
                logger.warning(f"🏁 FREEDOM: Lock {original_ticket} paid off via Harvesting! Unwind safely.")
