
import logging
import numpy as np
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("BayesianSwarm")

class BayesianSwarm(SubconsciousUnit):
    """
    The Bayesian Seer.
    Phase 37 Innovation.
    Logic:
    1. Problem: Signals conflict. How to combine them?
    2. Solution: Bayesian Inference.
    3. Formula: Posterior = (Likelihood * Prior) / Evidence.
    4. Components:
       - Prior: The base probability of Up/Down (derived from Daily Trend).
       - Likelihood: The probability of seeing specific M5 signals GIVEN the Trend is Real.
    """
    def __init__(self):
        super().__init__("BayesianSwarm")
        # Initial Belief (Flat)
        self.belief_bull = 0.5

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        df_d1 = context.get('data_map', {}).get('D1')
        df_m5 = context.get('df_m5')
        
        # 1. Establish Prior (The Base Trend)
        # If D1 is bullish, our Prior belief in a Buy is > 0.5.
        prior_bull = 0.5
        if df_d1 is not None and len(df_d1) > 20:
            ma20 = df_d1['close'].rolling(20).mean().iloc[-1]
            close = df_d1['close'].iloc[-1]
            if close > ma20:
                prior_bull = 0.60 # Slight bias
            else:
                prior_bull = 0.40
        
        # 2. Update with Evidence (The M5 Signals)
        # We look at simple heuristic evidence from the M5 dataframe.
        # Check standard indicators as "Evidence".
        
        # Evidence A: RSI > 50
        # P(RSI>50 | Bull) = 0.7 (Likelihood)
        # P(RSI>50 | Bear) = 0.3
        
        if df_m5 is None or len(df_m5) < 20: return None
        
        # Calculate naive RSI Evidence
        delta = df_m5['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        evidence_is_bullish = (rsi > 50)
        
        # Likelihoods (Hardcoded for now, ideal RL would tune these)
        p_evidence_given_bull = 0.70 if evidence_is_bullish else 0.30
        p_evidence_given_bear = 0.30 if evidence_is_bullish else 0.70
        
        # Total Probability of Evidence
        # P(E) = P(E|Bull)*P(Bull) + P(E|Bear)*P(Bear)
        prior_bear = 1.0 - prior_bull
        p_evidence = (p_evidence_given_bull * prior_bull) + (p_evidence_given_bear * prior_bear)
        
        # 3. Calculate Posterior (The Update)
        # P(Bull | Evidence)
        posterior_bull = (p_evidence_given_bull * prior_bull) / p_evidence
        
        # 4. Sequential Update (Chain of Reasoning)
        # Incorporate Momentum
        # Evidence B: Close > MA20 (M5)
        ma20_m5 = df_m5['close'].rolling(20).mean().iloc[-1]
        evidence_b_bull = (df_m5['close'].iloc[-1] > ma20_m5)
        
        # Use previous Posterior as new Prior
        prior_bull_2 = posterior_bull
        prior_bear_2 = 1.0 - prior_bull_2
        
        p_e2_given_bull = 0.65 if evidence_b_bull else 0.35
        p_e2_given_bear = 0.35 if evidence_b_bull else 0.65
        
        p_evidence_2 = (p_e2_given_bull * prior_bull_2) + (p_e2_given_bear * prior_bear_2)
        
        final_posterior_bull = (p_e2_given_bull * prior_bull_2) / p_evidence_2
        
        # 5. Output
        signal = "WAIT"
        confidence = 0.0
        
        # Bayesian Threshold > 85% certainty
        if final_posterior_bull > 0.85:
            signal = "BUY"
            confidence = final_posterior_bull * 100
        elif final_posterior_bull < 0.15: # < 15% Bull means > 85% Bear
            signal = "SELL"
            confidence = (1.0 - final_posterior_bull) * 100
            
        if signal != "WAIT":
             return SwarmSignal(
                source="BayesianSwarm",
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={
                    "prior": prior_bull,
                    "posterior": final_posterior_bull,
                    "logic": "Bayesian Inference Chain"
                }
            )
            
        return None
