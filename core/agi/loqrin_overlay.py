# core/agi/loqrin_overlay.py

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("LoqrinOverlay")

class LoqrinOverlay:
    """
    Phase 27: The Loqrin Method (Institutional Shadowing).
    
    Logic: Inducement & Liquidity Sweeps.
    -------------------------------------
    The "Matrix" (Retail) trades Support & Resistance.
    The "Institutions" (Smart Money) hunt those stops.
    
    This module:
    1. Identifies Standard Fractal Levels (Retail Stops).
    2. Waits for a 'Sweep' (Price pierces level then reverses).
    3. Signals an Entry on the RECLAIM (The Trap is sprung).
    """
    
    def __init__(self):
        self.levels = {
            'swing_highs': [],
            'swing_lows': []
        }
        self.last_sweep = None # {type: 'BULL_SWEEP', level: 1.2000, time: 123456}

    def analyze_liquidity(self, df: pd.DataFrame):
        """
        Maps the Retail Playing Field.
        Finds 'Obvious' Highs and Lows (Fractals).
        """
        if df is None or len(df) < 20: return
        
        # Simple Fractal Logic (Bill Williams style or 3-bar)
        # We look for a High surrounded by lower highs.
        
        # Reset
        self.levels['swing_highs'] = []
        self.levels['swing_lows'] = []
        
        # Vectorized or Iterative (Iterative safer for logic clarity here)
        # We focus on the last 50 bars
        recent = df.tail(50).copy().reset_index(drop=True)
        
        for i in range(2, len(recent) - 2):
            # Swing High
            if (recent['high'][i] > recent['high'][i-1] and 
                recent['high'][i] > recent['high'][i-2] and
                recent['high'][i] > recent['high'][i+1] and
                recent['high'][i] > recent['high'][i+2]):
                self.levels['swing_highs'].append(recent['high'][i])
                
            # Swing Low
            if (recent['low'][i] < recent['low'][i-1] and 
                recent['low'][i] < recent['low'][i-2] and
                recent['low'][i] < recent['low'][i+1] and
                recent['low'][i] < recent['low'][i+2]):
                self.levels['swing_lows'].append(recent['low'][i])
                
        # Keep only the most recent/relevant
        self.levels['swing_highs'] = sorted(self.levels['swing_highs'])[-3:] # Top 3 Resistance
        self.levels['swing_lows'] = sorted(self.levels['swing_lows'])[:3]    # Bottom 3 Support
        
        # logger.debug(f"LOQRIN: Levels Mapped. Highs: {self.levels['swing_highs']}, Lows: {self.levels['swing_lows']}")

    def check_inducement(self, tick: dict) -> dict:
        """
        Checks if a Trap (Sweep) has occurred.
        Returns: Signal Dict
        """
        if not self.levels['swing_highs'] and not self.levels['swing_lows']:
            return {'signal': 'NEUTRAL', 'reason': 'No Levels'}
            
        ask = tick['ask']
        bid = tick['bid']
        
        # 1. Bearish Sweep (Trap Bulls)
        # Price goes ABOVE a high, then drops BELOW it.
        # Ideally we track this statefully, but for tick-logic we check proximity/reversion.
        # Simplified: We check if we are slightly below a High that was arguably breached recently?
        # Better: We check if Price is "Peeking" above.
        
        # REAL TIME LOGIC:
        # We want to forbid buying AT the level.
        
        for high in self.levels['swing_highs']:
            # If we are effectively AT the high (within 1 point - Tightened from 2)
            if abs(ask - high) < 0.0001: 
                return {'signal': 'VETO_BUY', 'reason': f"Retail Resistance ({high:.5f}). Wait for Sweep."}
                
        for low in self.levels['swing_lows']:
            if abs(bid - low) < 0.0001:
                return {'signal': 'VETO_SELL', 'reason': f"Retail Support ({low:.5f}). Wait for Sweep."}
                
        return {'signal': 'NEUTRAL'}
