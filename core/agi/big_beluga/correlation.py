
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("CorrelationSynapse")

class CorrelationSynapse:
    """
    Analyzes Inter-Market Correlations to validate trade logic.
    "The Web of Causality"
    """
    def __init__(self):
        self.rolling_window = 50
        self.risk_assets = ['AUDJPY', 'BTCUSD', 'SPX500', 'NAS100']
        self.safe_havens = ['USDCHF', 'XAUUSD', 'DXY']
        
    def analyze_correlations(self, target_symbol: str, basket_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Computes correlation matrix and determines Risk Sentiment.
        """
        if not basket_data or target_symbol not in basket_data:
            return {"sentiment": "NEUTRAL", "warning": "Insufficient Basket Data"}

        target_df = basket_data[target_symbol]
        
        # 1. Build Alignment Table
        correlations = {}
        for asset, df in basket_data.items():
            if asset == target_symbol: continue
            
            # Align timestamps (inner join)
            common = target_df['close'].align(df['close'], join='inner')
            if len(common[0]) < self.rolling_window: continue
            
            # Compute Rolling Correlation (last value)
            corr = common[0].rolling(self.rolling_window).corr(common[1]).iloc[-1]
            correlations[asset] = corr

        # 2. Risk Sentiment Analysis
        # Risk-On if AUDJPY is rising up and CHF is weak
        sentiment = "NEUTRAL"
        risk_score = 0.0
        
        # Simple Proxy: Check recent returns of Risk Basket
        for risk_asset in self.risk_assets:
            if risk_asset in basket_data:
                df = basket_data[risk_asset]
                if not df.empty:
                    ret = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
                    if ret > 0.001: risk_score += 1
                    elif ret < -0.001: risk_score -= 1
                    
        for safe_asset in self.safe_havens:
            if safe_asset in basket_data:
                df = basket_data[safe_asset]
                if not df.empty:
                    ret = df['close'].iloc[-1] / df['close'].iloc[-5] - 1
                    if ret > 0.001: risk_score -= 1 # Safe haven up = Risk Off
                    elif ret < -0.001: risk_score += 1
        
        if risk_score >= 2: sentiment = "RISK_ON"
        elif risk_score <= -2: sentiment = "RISK_OFF"
        
        # 3. DXY Check (The Dollar King)
        # Assuming DXY is present or proxied by USDCHF/EURUSD inverse
        dxy_check = "NEUTRAL"
        # If we trade EURUSD, we expect negative correlation with USDCHF or DXY
        if "USD" in target_symbol:
             # Basic check to see if we are decoupled from the basket
             pass
             
        return {
            "correlations": correlations,
            "global_risk_sentiment": sentiment,
            "risk_score": risk_score,
            "primary_correlation": correlations
        }
