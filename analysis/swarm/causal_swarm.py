import numpy as np
import pandas as pd
from typing import Dict, Any, List
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

class CausalSwarm(SubconsciousUnit):
    """
    Phase 105: The Causal Nexus.
    
    Analyzes the Cross-Asset Topology of the market.
    Determines Lead-Lag relationships to identify the 'Dominant Driver'.
    
    Logic:
    1. Ingests price data for ALL active symbols in the basket.
    2. Computes Rolling Correlation Matrix.
    3. Detects 'Driver' (Asset with highest mean correlation and valid directional movement).
    4. Signals based on the Driver's trend, overriding local noise.
    """

    def __init__(self, name: str = "Causal_Nexus", lookback: int = 50):
        super().__init__(name)
        self.lookback = lookback
        self.driver_symbol = None
        
    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        """
        Context must contain 'basket_data': Dict[symbol, DataFrame]
        """
        # We need access to the broader market view.
        # This might need to be injected into context by OpportunityFlowManager.
        basket_data = context.get('basket_data', {})
        
        # If we only have local data, we can't do causal analysis.
        if not basket_data or len(basket_data) < 2:
            return None
            
        # 1. Align Data
        # We need closing prices for all assets on the same timeframe (e.g. M5)
        aligned_prices = pd.DataFrame()
        
        for sym, df in basket_data.items():
            if df is not None and not df.empty:
                # Use the last N candles
                series = df['close'].iloc[-self.lookback:]
                
                # Check length match roughly (simple truncation)
                if len(aligned_prices) > 0:
                     min_len = min(len(aligned_prices), len(series))
                     aligned_prices = aligned_prices.iloc[-min_len:]
                     series = series.iloc[-min_len:]
                     
                aligned_prices[sym] = series.values
                
        if aligned_prices.empty or len(aligned_prices.columns) < 2:
             return None

        # 2. Compute Correlation Matrix
        try:
            corr_matrix = aligned_prices.corr()
        except:
            return None
        
        # 3. Identify Driver (Centrality)
        # The asset with the highest sum of absolute correlations is the most connected.
        # But we also care about movement magnitude (Volatility).
        
        # Simple Centrality: Sum of |Correlation|
        centrality = corr_matrix.abs().sum().sort_values(ascending=False)
        top_driver = centrality.index[0]
        
        self.driver_symbol = top_driver
        
        # 4. Analyze Driver Direction
        driver_price = aligned_prices[top_driver]
        
        if driver_price.empty: return None
        
        # Simple Trend: ROC of Driver
        start = driver_price.iloc[0]
        end = driver_price.iloc[-1]
        
        if start == 0: return None
        
        roc = (end - start) / start * 100
        
        # 5. Generate Signal for CURRENT Symbol (context('symbol') is usually passed in args, 
        # but context is a dict. The target symbol usually comes from context['symbol'] if available.
        # If not, we check if we can infer it.)
        
        # Swarm Orchestrator context composition:
        # process_tick(self, tick, context, config)
        # context generally includes 'symbol' if we added it?
        # Typically the 'tick' has 'symbol'.
        
        target_symbol = context.get('symbol')
        if not target_symbol:
             # Try getting from tick
             tick = context.get('tick')
             if tick: target_symbol = tick.get('symbol')
             
        if not target_symbol: return None
        if target_symbol not in corr_matrix.columns: return None
        
        # Calculate Correlation between Driver and Target
        correlation = corr_matrix.loc[top_driver, target_symbol]
        
        signal = "WAIT"
        conf = 0.0
        meta = {
            'driver': top_driver,
            'correlation': correlation,
            'driver_roc': roc
        }
        
        # Logic:
        # If Target is Correlated (> 0.5) with Driver: Follow Driver.
        # If Target is Inverse (< -0.5) with Driver: Fade Driver.
        # If Unrelated: No Signal.
        
        if abs(correlation) > 0.6:
            # Strong Link
            if correlation > 0:
                # Positive Correlation
                if roc > 0.1: # Driver is Pumping
                    signal = "BUY"
                    conf = 85.0 + (abs(roc) * 100) # Boost by strength
                elif roc < -0.1: # Driver is Dumping
                    signal = "SELL"
                    conf = 85.0 + (abs(roc) * 100)
            else:
                # Negative Correlation (Inverse)
                if roc > 0.1: # Driver Pumping -> Target Dumps
                    signal = "SELL"
                    conf = 85.0
                elif roc < -0.1: # Driver Dumping -> Target Pumps
                    signal = "BUY"
                    conf = 85.0
                    
        # Clamp Confidence
        conf = min(99.0, max(0.0, conf))
        
        if signal != "WAIT":
            return SwarmSignal(
                signal_type=signal,
                confidence=conf,
                source=self.name,
                meta_data=meta,
                timestamp=time.time()
            )
            
        return None
