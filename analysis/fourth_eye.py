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
        self.cooldown_seconds = 60 # 1 Minute Cooldown for Testing
        
    def process_tick(self, tick, df_m5, consensus_score, smc_score, reality_state, volatility_score, base_lots=0.01):
        """
        Calculates Quantum Whale Entry.
        Args:
            tick: Live tick data
            df_m5: M5 DataFrame
            consensus_score: Base score from Consensus Engine (-100 to 100)
            smc_score: Score from SmartMoneyEngine (-100 to 100)
            reality_state: String state from HyperDimension (e.g., "DIMENSIONAL_TREND_BUY")
            volatility_score: Volatility metric (0-100)
            base_lots: Dynamic base lot size calculated from risk manager
            
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
        # User Request: Boost weight to 0.6
        smc_weight = smc_score * 0.6
        
        # C. Consensus Weight (The Crowd/Math)
        consensus_weight = consensus_score 
        
        # D. Volatility Multiplier (Entropy)
        # High Volatility = Higher Risk but Higher conviction if aligned.
        # Low Volatility = Noise.
        # E. Iceberg Weight (Hidden Volume) - NEW v3.1
        iceberg_score, iceberg_dir = self.analyze_icebergs(df_m5)
        iceberg_weight = iceberg_score * iceberg_dir * 0.8 # Strong weight
        
        entropy_factor = 1.0
        if volatility_score > 70: entropy_factor = 1.2 # Boost score in high energy
        elif volatility_score < 10: entropy_factor = 0.8 # Only penalize dead markets (was < 20)
        
        # 3. FINAL SYNTHESIS
        # Total Quantum Score = (Consensus + SMC + Reality + Iceberg) * Entropy
        quantum_score = (consensus_weight + smc_weight + reality_weight + iceberg_weight) * entropy_factor
        
        action = None
        reason = None
        
        # 4. DECISION LOGIC (The Singularity)
        # We need a STRONG signal to move the Whale.
        
        whale_threshold = 30.0 # Lowered from 40.0 for more activity (User Request)
        
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
             
        # Veto 4: Extension Limits (The "Buy the Top" Preventer)
        # Calculate lightweight RSI & Bollinger
        closes = df_m5['close']
        if len(closes) > 20:
            # RSI 14
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Bollinger Bands (20, 2.5) - Slightly wider to allow trends but catch extremities
            ma20 = closes.rolling(20).mean().iloc[-1]
            std20 = closes.rolling(20).std().iloc[-1]
            upper_band = ma20 + (2.5 * std20)
            lower_band = ma20 - (2.5 * std20)
            current_price = closes.iloc[-1]
            
            # Logic: Don't Buy if we are screamingly overbought
            if action == "BUY":
                if rsi > 75:
                    return None, f"VETO: Overbought (RSI {rsi:.1f})", 0
                if current_price > upper_band:
                    # Exception: If we have SUPER high momentum (SMC > 80), maybe allow it?
                    # No, user specifically halted on this. Block it.
                    return None, "VETO: Price Extension (Above Upper Band)", 0
                    
            if action == "SELL":
                if rsi < 25:
                    return None, f"VETO: Oversold (RSI {rsi:.1f})", 0
                if current_price < lower_band:
                    return None, "VETO: Price Extension (Below Lower Band)", 0

        if action:
            # 6. Dynamic Sizing (Quantum Scale)
            # Base = base_lots (from Risk Manager)
            # Scale with Quantum Score Excess
            excess = abs(quantum_score) - whale_threshold
            # 0.01 per 10 points of excess is aggressive. 
            # We scale relative to base_lots.
            # E.g. Excess 20 -> 2 extra increments.
            
            # If base_lots is 0.02 (small account), increments should be small (e.g. 0.01)
            # If base_lots is 1.0 (large account), increments should be large (e.g. 0.1)
            
            scaling_factor = max(0.01, base_lots * 0.5) 
            extra_lots = (excess // 10) * scaling_factor
            
            dynamic_lots = round(base_lots + extra_lots, 2)
            
            # Hard Safety Cap (e.g. 5x base)
            max_lots = base_lots * 5
            if dynamic_lots > max_lots: dynamic_lots = max_lots
            
            self.trade_executed_this_candle = True
            self.last_trade_time = time.time()
            return action, reason, dynamic_lots
            
        return None, None, 0

    def analyze_icebergs(self, df):
        """
        Detects Hidden Volume (Icebergs).
        Logic: High Volume + Low Range = Absorption.
        Returns: (Score, Direction)
        """
        if len(df) < 20: return 0, 0
        
        # 1. Prepare Data
        # Calculate Range first so it's in 'last'
        df['range'] = df['high'] - df['low']
        
        last = df.iloc[-1]
        
        # 2. Relative Volume
        # Handle different column names (tick_volume vs Volume)
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'Volume' if 'Volume' in df.columns else 'volume'
        
        if vol_col not in df.columns:
             # No volume data available
             return 0, 0
             
        avg_vol = df[vol_col].rolling(20).mean().iloc[-1]
        curr_vol = last[vol_col]
        
        # 3. Relative Range
        avg_range = df['range'].rolling(20).mean().iloc[-1]
        curr_range = last['range']
        
        if avg_range == 0: return 0, 0
        
        # ICEBERG RATIO: (Vol / AvgVol) / (Range / AvgRange)
        # If Vol is 2.0x and Range is 0.5x, Ratio is 4.0 (Huge Absorption)
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0
        range_ratio = curr_range / avg_range if avg_range > 0 else 1.0
        
        iceberg_score = 0
        direction = 0
        
        if vol_ratio > 1.5 and range_ratio < 0.7:
             # Absorption Detected
             # Direction? Look at Close vs Open
             # If Close > Open, absorption was likely Buying (Demand absorbing Supply)
             # Actually, if price didn't move much, we look at the 'Close' relative to 'Low'/'High'
             # Pinbar logic basically
             
             iceberg_score = 50 * (vol_ratio / range_ratio) # Dynamic score
             iceberg_score = min(iceberg_score, 100)
             
             # Determine nature of absorption
             if last['close'] > last['open']:
                 direction = 1 # Buying Absorption
                 logger.info(f"WHALE: Iceberg Buying Detected (Ratio {vol_ratio/range_ratio:.1f})")
             else:
                 direction = -1 # Selling Absorption (Blocking)
                 logger.info(f"WHALE: Iceberg Selling Detected (Ratio {vol_ratio/range_ratio:.1f})")
                 
        return iceberg_score, direction
