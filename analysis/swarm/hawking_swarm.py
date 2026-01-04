
import logging
import numpy as np
import time
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("HawkingSwarm")

class HawkingSwarm(SubconsciousUnit):
    """
    Phase 110: The Event Horizon (Hawking Radiation).
    
    Models Trends as Black Holes that grow via Mass Accretion (Volume).
    When Mass Accretion stops but 'Surface Area' (Price Range) expands, 
    the Black Hole becomes unstable and evaporates (Reversal).
    
    Key Metric: Mass-Loss Rate vs Horizon Expansion.
    """
    
    def __init__(self):
        super().__init__("Hawking_Swarm")
        self.mass_window = [] # Store recent volume
        self.price_window = [] # Store recent price
        self.window_size = 50 
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        tick = context.get('tick')
        if not tick: return None
        
        price = tick['bid']
        volume = tick.get('volume', 1.0)
        
        # Avoid zero volume
        if volume <= 0: volume = 1.0
        
        self.mass_window.append(volume)
        self.price_window.append(price)
        
        if len(self.mass_window) > self.window_size:
            self.mass_window.pop(0)
            self.price_window.pop(0)
            
        if len(self.mass_window) < 20: return None
        
        # --- PHYSICS CALCULATIONS ---
        
        # 1. Event Horizon Radius (Rs) ~ Price Range (Volatility)
        # Expansion of the trend
        prices = np.array(self.price_window)
        rs = np.max(prices) - np.min(prices)
        
        if rs == 0: return None
        
        # 2. Accretion Rate (Mean Volume of first half vs second half)
        mid = len(self.mass_window) // 2
        old_mass = np.mean(self.mass_window[:mid])
        new_mass = np.mean(self.mass_window[mid:])
        
        # evaporation_rate = (Old - New) / Old
        # Positive = Decaying Volume. Negative = Increasing Volume.
        evaporation_factor = (old_mass - new_mass) / (old_mass + 1e-9)
        
        # 3. Trend Direction
        # Is price currently at the Edge of the Horizon?
        current_price = prices[-1]
        max_price = np.max(prices)
        min_price = np.min(prices)
        
        is_at_high = abs(current_price - max_price) < (rs * 0.05) # Within 5% of High
        is_at_low = abs(current_price - min_price) < (rs * 0.05) # Within 5% of Low
        
        signal = "WAIT"
        conf = 0.0
        meta = {'evaporation': evaporation_factor, 'rs': rs}
        
        # LOGIC: HAWKING RADIATION
        # If we are at the Horizon (High/Low) AND Evaporation is High (> 0.3)
        # The Black Hole is losing mass while trying to expand.
        # Boom.
        
        if is_at_high and evaporation_factor > 0.3:
            # Fake Breakout / Exhaustion
            signal = "SELL"
            conf = min(99.0, 75 + (evaporation_factor * 20))
            meta['reason'] = f"Hawking Radiation: High Evaporation ({evaporation_factor:.2f}) at Event Horizon (Top)."
            
        elif is_at_low and evaporation_factor > 0.3:
            # Fake Dump / Exhaustion
            signal = "BUY"
            conf = min(99.0, 75 + (evaporation_factor * 20))
            meta['reason'] = f"Hawking Radiation: High Evaporation ({evaporation_factor:.2f}) at Event Horizon (Bottom)."
            
        # Optional: Accretion (Strong Trend Confirmation)
        # If Volume is INCREASING (Evaporation < -0.3) and we are at High -> Breakout verified.
        elif is_at_high and evaporation_factor < -0.3:
            signal = "BUY"
            conf = 65.0
            meta['reason'] = "Mass Accretion: Trend Fueled by Volume."
            
        elif is_at_low and evaporation_factor < -0.3:
            signal = "SELL"
            conf = 65.0
            meta['reason'] = "Mass Accretion: Trend Fueled by Volume."
            
        if signal == "WAIT": return None
        
        return SwarmSignal(
            source=self.name,
            signal_type=signal,
            confidence=conf,
            timestamp=time.time(),
            meta_data=meta
        )
