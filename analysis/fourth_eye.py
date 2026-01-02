import logging
import time
import config
import numpy as np

logger = logging.getLogger("Atl4s-FourthEye")

class FourthEye:
    """
    The Quantum Whale (System IV).
    
    A Multi-Dimensional Decision Engine that aggregates:
    1. Consensus Score (The Hive Mind)
    2. Smart Money Concepts (Order Blocks & FVG)
    3. Hyper-Dimensional Geometry (reality_state)
    4. Market Entropy (Volatility)
    
    Logic:
    - It searches for 'Confluence' across dimensions.
    - If Consensus says BUY but Order Block says SELL -> VETO.
    - If Consensus says BUY and Reality says BUY -> SUPERSIZE.
    """
    def __init__(self):
        self.last_candle_time = None
        self.trade_executed_this_candle = False
        self.threshold = 30.0 # Slightly lower base threshold because we have filters
        self.last_trade_time = 0
        self.cooldown_seconds = 300 # 5 Minutes Hard Cooldown
        
    def process_tick(self, tick, df_m5, consensus_score, smc_score, reality_state, volatility_score):
        """
        Calculates Quantum Whale Entry.
        Args:
            tick: Live tick data
            df_m5: M5 DataFrame
            consensus_score: Base score from Consensus Engine (-100 to 100)
            smc_score: Score from SmartMoneyEngine (-100 to 100)
            reality_state: String state from HyperDimension (e.g., "DIMENSIONAL_TREND_BUY")
            volatility_score: Volatility metric (0-100)
            
        Returns: (Action, Reason, DynamicLots)
        """
        # 1. Candle Check
        if df_m5.empty: return None, None, 0
        
        latest_candle_time = df_m5.index[-1]
        if self.last_candle_time != latest_candle_time:
            self.last_candle_time = latest_candle_time
            self.trade_executed_this_candle = False
            
        if self.trade_executed_this_candle:
            return None, "Candle Limit", 0
            
        # Hard Cooldown Check
        if time.time() - self.last_trade_time < self.cooldown_seconds:
             return None, "Cooldown", 0
            
        # 2. QUANTUM WEIGHTING MATRIX
        # We normalize everything to a -100 to 100 scale and sum them up with weights.
        
        # A. Reality Weight (Geometry)
        # "The chart doesn't lie, but it can bend."
        reality_weight = 0
        if "BUY" in reality_state: reality_weight = 40
        elif "SELL" in reality_state: reality_weight = -40
        
        # B. Smart Money Weight (Structure)
        # "Follow the banks."
        # smc_score comes in as -100 to 100 already.
        # We dampen it slightly to avoid over-trading on small FVGs.
        smc_weight = smc_score * 0.5 
        
        # C. Consensus Weight (The Crowd/Math)
        consensus_weight = consensus_score 
        
        # D. Volatility Multiplier (Entropy)
        # High Volatility = Higher Risk but Higher conviction if aligned.
        # Low Volatility = Noise.
        entropy_factor = 1.0
        if volatility_score > 70: entropy_factor = 1.2 # Boost score in high energy
        elif volatility_score < 20: entropy_factor = 0.5 # Dampen in dead market
        
        # 3. FINAL SYNTHESIS
        # Total Quantum Score = (Consensus + SMC + Reality) * Entropy
        quantum_score = (consensus_weight + smc_weight + reality_weight) * entropy_factor
        
        action = None
        reason = None
        
        # 4. DECISION LOGIC (The Singularity)
        # We need a STRONG signal to move the Whale.
        
        whale_threshold = 45.0 # High bar for entry
        
        if quantum_score > whale_threshold:
            action = "BUY"
            reason = f"Q-Matrix BUY (Score: {quantum_score:.1f} | C:{consensus_score:.0f} S:{smc_score:.0f} R:{reality_state})"
        elif quantum_score < -whale_threshold:
            action = "SELL"
            reason = f"Q-Matrix SELL (Score: {quantum_score:.1f} | C:{consensus_score:.0f} S:{smc_score:.0f} R:{reality_state})"
            
        # 5. VETO POWER (The Safety Valve)
        # Even if score is high, check for HARD contradictors
        
        # Veto 1: Buying into a Bearish Order Block?
        if action == "BUY" and smc_score < -50:
            return None, "VETO: Buying into Bearish OB", 0
            
        # Veto 2: Selling into a Bullish Order Block?
        if action == "SELL" and smc_score > 50:
             return None, "VETO: Selling into Bullish OB", 0
             
        # Veto 3: Reality Collapse (Reversal Mode)
        if action == "BUY" and "SELL_REVERSAL" in reality_state:
             return None, "VETO: Dimensional Reversal Detected", 0
             
        if action == "SELL" and "BUY_REVERSAL" in reality_state:
             return None, "VETO: Dimensional Reversal Detected", 0



        if action:
            # 6. Dynamic Sizing (Quantum Scale)
            # Base 0.01
            # Scale with Quantum Score Excess
            excess = abs(quantum_score) - whale_threshold
            extra_lots = (excess // 10) * 0.01 # Less aggressive scaling than before
            
            dynamic_lots = round(0.01 + extra_lots, 2)
            if dynamic_lots > 0.20: dynamic_lots = 0.20 # Cap
            
            self.trade_executed_this_candle = True
            self.last_trade_time = time.time()
            return action, reason, dynamic_lots
            
        return None, None, 0
