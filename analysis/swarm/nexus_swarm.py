
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from core.interfaces import SwarmSignal, SubconsciousUnit

logger = logging.getLogger("NexusSwarm")

class NexusSwarm(SubconsciousUnit):
    """
    The Nexus (Inter-Market Intelligence).
    Phase 29 Innovation.
    Logic:
    1. Reads Global Basket (BTC, ETH, XAU, EUR).
    2. Calculates Correlation Matrix with current asset.
    3. Determines "Global Risk Sentiment".
       - Example: If Crypto + Gold + Stocks all Up -> RISK_ON.
       - If only USD Up -> RISK_OFF.
    4. Provides "NEXUS_BIAS".
    """
    def __init__(self):
        super().__init__("NexusSwarm")

    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        data_map = context.get('data_map', {})
        global_basket = data_map.get('global_basket', {})
        df_h1 = data_map.get('H1')
        tick = context.get('tick')
        
        if df_h1 is None or len(df_h1) < 24: return None
        if not global_basket: return None
        
        # 1. Prepare returns Dataframe
        returns_df = pd.DataFrame()
        # Current asset returns
        returns_df['SELF'] = df_h1['close'].pct_change()
        
        # Correlated assets
        for symbol, df in global_basket.items():
            if df is not None and not df.empty:
                # Reindex to match timescales (approx)
                # For simplicity in this HFT context, we take tail(24) and align index roughly
                series = df['close'].pct_change()
                # We need to align timestamps. 
                # For this prototype, we assume roughly synced H1 bars from Yahoo.
                # We rename to avoid collision
                returns_df[symbol] = series

        returns_df = returns_df.tail(24).dropna()
        if len(returns_df) < 10: return None
        
        # 2. Correlation Matrix
        corr_matrix = returns_df.corr()
        self_corr = corr_matrix['SELF']
        
        # 3. Analyze Global Drift (Risk Sentiment)
        # Sum of correlations * Performance
        # If I am positively correlated with BTC, and BTC is dumping, I should probably dump.
        
        nexus_score = 0.0
        details = []
        
        for symbol in global_basket.keys():
            if symbol not in returns_df.columns: continue
            
            # Correlation with me
            r = self_corr[symbol]
            
            # Current performance (Last 4 hours momentum)
            # 1.0 = Up, -1.0 = Down
            recent_perf = returns_df[symbol].tail(4).sum()
            
            # Impact: If I correlate (r=0.8) and it went up (perf=0.02) -> Score += 0.016
            impact = r * recent_perf
            nexus_score += impact
            details.append(f"{symbol}(r={r:.2f},p={recent_perf:.3f})")
            
        # 4. Interpret Nexus Score
        # Positive Score = The web of assets pulls us UP.
        # Negative Score = The web pulls us DOWN.
        
        signal = "WAIT"
        confidence = 0.0
        bias = "NEUTRAL"
        
        # Thresholds need calibration. 0.01 is actually significant for sum of returns.
        if nexus_score > 0.005: 
            signal = "BUY"
            bias = "RISK_ON"
            confidence = 75.0 
        elif nexus_score < -0.005:
            signal = "SELL"
            bias = "RISK_OFF"
            confidence = 75.0
            
        reason = f"Nexus Score: {nexus_score:.5f} [{bias}] | " + ", ".join(details[:3])
        
        return SwarmSignal(
            source="NexusSwarm",
            signal_type=signal,
            confidence=confidence,
            timestamp=0,
            meta_data={"reason": reason, "nexus_score": nexus_score, "bias": bias}
        )
