
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.black_swan_adversary import BlackSwanAdversary # Reuse logic

logger = logging.getLogger("VetoSwarm")

class VetoSwarm(SubconsciousUnit):
    """
    The Censor. 
    A Swarm of 'No-sayers' that must be silenced for a trade to pass.
    """
    def __init__(self):
        super().__init__("Veto_Swarm")
        self.adversary = BlackSwanAdversary()

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        # VetoSwarm runs slightly differently. It inspects the 'Proposed Decision' if available.
        # But 'process' is usually run in parallel before decision. 
        # So we VETO based on Market State, effectively flagging "Toxic Conditions".
        
        df_m5 = context.get('df_m5')
        market_state = context.get('market_state', {})
        
        if df_m5 is None: return None
        
        votes = []
        
        # 1. Stress Test Agent (Black Swan Monitor)
        # If market volatility is crazy high, pre-emptively VETO.
        # 1. Stress Test Agent (Black Swan Monitor)
        # If market volatility is crazy high, pre-emptively VETO.
        
        # Calculate ATR if missing
        if 'ATR' not in df_m5.columns:
            high_low = df_m5['high'] - df_m5['low']
            high_close = np.abs(df_m5['high'] - df_m5['close'].shift())
            low_close = np.abs(df_m5['low'] - df_m5['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            current_atr = atr.iloc[-1]
        else:
            current_atr = df_m5.iloc[-1]['ATR']
            
        atr_pct = (current_atr / df_m5.iloc[-1]['close']) * 100
        if atr_pct > 0.5: # Extreme Volatility (>0.5% M5 range)
             votes.append("VETO: Extreme Volatility (Black Swan Risk)")
             
        # 2. Reality Agent (Physics Check)
        close = df_m5['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if loss > 0:
            rsi = 100 - (100 / (1 + (gain/loss)))
            if rsi > 85: votes.append("VETO: Reality Check (RSI > 85 Overbought)")
            if rsi < 15: votes.append("VETO: Reality Check (RSI < 15 Oversold)")
            
        # 3. Trap Agent (Volume Disconnect)
        # Price Up, Volume Down = Trap
        price_trend = df_m5['close'].iloc[-1] - df_m5['close'].iloc[-5]
        vol_trend = df_m5['volume'].iloc[-1] - df_m5['volume'].iloc[-5]
        
        # Simple divergence check
        if abs(price_trend) > 0 and (np.sign(price_trend) != np.sign(vol_trend)):
             # Rising Price, Falling Volume?
             if price_trend > 0 and vol_trend < 0:
                 pass # Warning, but maybe not hard veto unless extreme
                 
        # 4. Weekend Guard (Dynamic)
        # Check if today is Saturday(5) or Sunday(6)
        current_time = pd.Timestamp.now()
        day_of_week = current_time.dayofweek # 0=Mon, 6=Sun
        
        # Check Config for "Weekend Mode" implied by Profile
        # If Virtual SL is wide (>15) it's likely Crypto/Weekend profile.
        # Or check spread_limit (0.05 vs 0.02).
        config = context.get('config', {})
        is_crypto_profile = config.get('spread_limit', 0.0) >= 0.04
        
        if day_of_week >= 5: # Saturday or Sunday
            if not is_crypto_profile:
                 votes.append(f"VETO: Market Closed (Weekend) for {symbol} (Profile: Forex)")
            else:
                 # It is Crypto Profile, allow it.
                 pass
                 
        if votes:
            reason = " | ".join(votes)
            logger.info(f"VETO SWARM BLOCK: {reason}")
            return SwarmSignal(
                source=self.name,
                signal_type="VETO",
                confidence=100.0,
                timestamp=0,
                meta_data={'reason': reason}
            )
            
        return None
