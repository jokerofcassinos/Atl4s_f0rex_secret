
import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("BlackSwanSwarm")

class BlackSwanSwarm(SubconsciousUnit):
    """
    The Abyss (Stress System).
    Restored from 'BlackSwanAdversary'.
    
    Acts as a Veto Gatekeeper.
    Simulates extreme market scenarios (Fat Tails) to detect system fragility.
    """
    def __init__(self):
        super().__init__("Black_Swan_Swarm")

    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Calc Fat Tail Risk (Kurtosis/Volatility)
        returns = df_m5['close'].pct_change().dropna()
        if len(returns) < 30: return None
        
        vol = returns.std()
        kurt = returns.kurtosis()
        
        # 2. Simulation Logic (Simplified from original)
        # If Volatility is Extreme (> 3 sigma) AND Kurtosis is High (Fat Tails),
        # The market is prone to Black Swans.
        
        risk_score = 0.0
        
        # Z-Score of current volatility
        # We need a baseline. Moving Average of Volatility?
        rolling_vol = returns.rolling(20).std()
        current_vol = rolling_vol.iloc[-1]
        mean_vol = rolling_vol.mean()
        
        if mean_vol == 0: return None
        
        z_score = (current_vol - mean_vol) / mean_vol
        
        reason = []
        if z_score > 3.0:
            risk_score += 50
            reason.append(f"Extreme Volatility (Z:{z_score:.2f})")
            
        if kurt > 3.0: # Leptokurtic
            risk_score += 30
            reason.append(f"Fat Tails (Kurt:{kurt:.2f})")
            
        # 3. Decision
        if risk_score > 70:
            return SwarmSignal(
                source=self.name,
                signal_type="VETO",
                confidence=95.0, # High authority
                timestamp=time.time(),
                meta_data={'risk_score': risk_score, 'reason': ", ".join(reason)}
            )
            
        return None
