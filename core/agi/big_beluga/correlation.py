
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

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
        
        # Hardcoded Correlation Matrix (General Guidelines)
        # 1.0 = Moves Together, -1.0 = Moves Opposite
        self.static_correlations = {
            'EURUSD': {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCHF': -0.9, 'USDJPY': -0.4, 'XAUUSD': 0.5},
            'GBPUSD': {'EURUSD': 0.85, 'AUDUSD': 0.65, 'USDCHF': -0.8, 'USDJPY': -0.3},
            'USDCHF': {'EURUSD': -0.9, 'GBPUSD': -0.8, 'USDJPY': 0.6, 'XAUUSD': -0.7, 'USDCAD': 0.7},
            'USDJPY': {'USDCHF': 0.6, 'EURUSD': -0.4, 'XAUUSD': -0.2, 'USDCAD': 0.5},
            'XAUUSD': {'EURUSD': 0.5, 'USDCHF': -0.7, 'Silver': 0.9, 'BTCUSD': 0.3},
            'USDCAD': {'AUDUSD': -0.7, 'Oil': -0.8, 'USDCHF': 0.7, 'EURUSD': -0.6, 'GBPUSD': -0.5}
        }
        
    def check_correlation_conflict(self, signal_symbol: str, signal_type: str, open_positions: List[Dict]) -> Tuple[bool, str]:
        """
        Checks if the proposed signal conflicts with existing open positions.
        
        Rule 1: Inverse Correlation Conflict
        - We want to BUY A.
        - We hold BUY B.
        - A and B are negatively correlated (-0.8).
        - Result: VETO. We are essentially hedging/neutralizing our exposure.
        
        Rule 2: Positive Correlation Overexposure
        - We want to BUY A.
        - We hold BUY B.
        - A and B are very highly correlated (>0.9).
        - Result: WARN (or Limit Risk). We are doubling down on the same move.
        """
        if not open_positions: return False, ""
        
        matrix = self.static_correlations.get(signal_symbol, {})
        if not matrix: return False, "" # No data
        
        for pos in open_positions:
            pos_symbol = pos['symbol']
            pos_type = pos['type'] # "BUY" or "SELL"
            
            correlation = matrix.get(pos_symbol, 0.0)
            
            if correlation == 0.0: continue
            
            # --- CONFLICT LOGIC ---
            # Scenario 1: Negative Correlation (e.g. EURUSD vs USDCHF)
            # If Corr is Negative, they should go OPPOSITE ways.
            # So BUY A and BUY B is BAD.
            # BUY A and SELL B is GOOD.
            if correlation < -0.7:
                 if signal_type == pos_type:
                      return True, f"Correlation Conflict: {signal_symbol} vs {pos_symbol} (Corr {correlation}). Both {signal_type} is dangerous."
                      
            # Scenario 2: Positive Correlation (e.g. EURUSD vs GBPUSD)
            # If Corr is Positive, they should go SAME way.
            # So BUY A and SELL B is BAD (Fighting the trend).
            if correlation > 0.7:
                 if signal_type != pos_type:
                      return True, f"Correlation Conflict: {signal_symbol} vs {pos_symbol} (Corr {correlation}). Divergent trades."
                      
        return False, ""

    def scan_sympathy_opportunities(self, signal_symbol: str, open_positions: List[Dict]) -> Tuple[bool, str]:
        """
        Checks if we can play a 'Sympathy Move' based on profitable existing positions.
        """
        if not open_positions: return False, ""
        
        matrix = self.static_correlations.get(signal_symbol, {})
        if not matrix: return False, ""
        
        for pos in open_positions:
            # Check only WINNING positions
            if pos.get('profit', 0) <= 0: continue
            
            pos_symbol = pos['symbol']
            pos_type = pos['type']
            correlation = matrix.get(pos_symbol, 0.0)
            
            # 1. Positive Sympathy (Follow the Leader)
            # Existing Winner: BUY GBPUSD
            # New Signal: BUY EURUSD (Corr 0.85)
            # Result: BOOST
            if correlation > 0.8 and  pos.get('profit', 0) > 1.0: # Ensure valid profit
                 return True, f"Sympathy Play: Following Winner {pos_symbol} (Corr {correlation})"
                 
            # 2. Inverse Sympathy (Fade the Loser proxy)
            # Existing Winner: BUY USDCHF
            # New Signal: SELL EURUSD (Corr -0.9)
            # Result: BOOST
            if correlation < -0.8 and pos.get('profit', 0) > 1.0:
                 return True, f"Sympathy Play: Inverse of Winner {pos_symbol} (Corr {correlation})"
                 
        return False, ""

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
