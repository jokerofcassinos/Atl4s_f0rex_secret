
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.black_swan_adversary import BlackSwanAdversary  # Reuse logic
from core.agi.thought_tree import ThoughtTree
from core.agi.decision_memory import ModuleDecisionMemory
from core.agi.swarm_thought_adapter import AGISwarmAdapter, SwarmThoughtResult

from core.agi.metacognition import RecursiveReflectionLoop

logger = logging.getLogger("VetoSwarm")

class VetoSwarm(SubconsciousUnit):
    """
    The Censor. 
    A Swarm of 'No-sayers' that must be silenced for a trade to pass.
    
    Phase 6: Enhanced with recursive thinking and decision memory.
    Phase 135: Integrated System #5 (Recursive Meta-Critic).
    """
    def __init__(self):
        super().__init__("Veto_Swarm")
        self.adversary = BlackSwanAdversary()
        
        # Phase 6: Recursive Thinking
        self.thought_tree = ThoughtTree("Veto_Swarm", max_depth=5)
        self.decision_memory = ModuleDecisionMemory("Veto_Swarm", max_memory=500)

        # Phase 9: Swarm-level AGI integration
        self.agi_adapter = AGISwarmAdapter("Veto_Swarm")
        
        # Phase 135: The Meta-Critic (System #5)
        self.meta_critic = RecursiveReflectionLoop()

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        # VetoSwarm runs slightly differently. It inspects the 'Proposed Decision' if available.
        # But 'process' is usually run in parallel before decision. 
        # So we VETO based on Market State, effectively flagging "Toxic Conditions".
        
        df_m5 = context.get('df_m5')
        market_state = context.get('market_state', {})
        
        if df_m5 is None: return None
        
        votes = []
        
        # 1. Stress Test Agent (Black Swan Reference)
        # Dynamic Regime Logic (System #23):
        # Instead of static 0.5%, we check for RELATIVE volatility expansion.
        
        # Calculate ATR if missing
        if 'ATR' not in df_m5.columns:
            high_low = df_m5['high'] - df_m5['low']
            high_close = np.abs(df_m5['high'] - df_m5['close'].shift())
            low_close = np.abs(df_m5['low'] - df_m5['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            current_atr = atr.iloc[-1]
            mean_atr = atr.rolling(100).mean().iloc[-1] # Long term mean
        else:
            current_atr = df_m5.iloc[-1]['ATR']
            mean_atr = df_m5['ATR'].rolling(100).mean().iloc[-1]
            
        atr_pct = (current_atr / df_m5.iloc[-1]['close']) * 100
        
        # Heuristic: If Mean ATR unavailable, fallback to static.
        if np.isnan(mean_atr) or mean_atr == 0:
             dynamic_threshold = 0.5
        else:
             # Allow up to 3x normal volatility before panic
             mean_atr_pct = (mean_atr / df_m5.iloc[-1]['close']) * 100
             dynamic_threshold = max(0.5, mean_atr_pct * 3.0)
        
        # Context Awareness: If News is moving the market, relax veto
        news_bias = context.get('news_bias', 0.0)
        if abs(news_bias) > 0.5:
            dynamic_threshold *= 1.5 # Allow 50% more chaos during news

        if atr_pct > dynamic_threshold: # Extreme Volatility
             votes.append(f"VETO: Extreme Volatility ({atr_pct:.2f}% > {dynamic_threshold:.2f}%)")
             
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
                 symbol = context.get('symbol', 'UNKNOWN')
                 votes.append(f"VETO: Market Closed (Weekend) for {symbol} (Profile: Forex)")
            else:
                 # It is Crypto Profile, allow it.
                 pass
                 
        # --- AGI INTERVENTION (META-CRITIC) ---
        if votes:
            reason = " | ".join(votes)
            
            # Ask the Meta-Critic: "Am I being paranoid?"
            reflection = self.meta_critic.reflect(
                decision="VETO",
                confidence=100.0,
                meta_data={'reason': reason},
                context=context
            )
            
            # If Meta-Critic significantly drops valid confidence, we suppress the veto
            # e.g. "Damped Overconfidence" for Veto
            if reflection['adjusted_confidence'] < 80.0:
                 logger.info(f"VETO SUPPRESSED by Meta-Critic: {reason} (Confidence {reflection['adjusted_confidence']:.1f}%)")
                 # We simply return None (No Veto Signal)
                 return None
            
            # Otherwise, proceed with Veto
            logger.info(f"VETO CONFIRMED: {reason}")
            
            return SwarmSignal(
                source=self.name,
                signal_type="VETO",
                confidence=reflection['adjusted_confidence'],
                timestamp=0,
                meta_data={
                    'reason': reason,
                    'meta_reflection': reflection
                }
            )
        
        return None
