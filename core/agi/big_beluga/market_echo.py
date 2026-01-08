
import numpy as np
import logging
from typing import Dict, Any
from .math_kernels import BigBelugaMath

logger = logging.getLogger("MarketEchoScreener")

class MarketEchoScreener:
    """
    Sistema 1/25: Market Echo Screener
    Rastreamento de alta frequência para shifts de tendência sincronizados usando ActionWave.
    """
    def __init__(self):
        self.history_price = []
        self.history_vol = []
        
    def scan(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        price = tick['bid']
        vol = tick.get('volume', 1)
        
        self.history_price.append(price)
        self.history_vol.append(vol)
        
        if len(self.history_price) > 50:
            self.history_price.pop(0)
            self.history_vol.pop(0)
            
        prices = np.array(self.history_price)
        vols = np.array(self.history_vol)
        
        # ActionWave Calculation
        wave = BigBelugaMath.action_wave(prices, vols)
        
        # Recursive Filtering
        trend_strength = np.mean(wave[-5:]) if len(wave) > 5 else 0.0
        
        return {
            "echo_signal": "BUY" if trend_strength > 0 else "SELL",
            "wave_intensity": float(np.max(wave)) if len(wave) > 0 else 0.0,
            "regime": "TRENDING" if abs(trend_strength) > 0.5 else "RANGING"
        }
