
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal
import time

logger = logging.getLogger("SovereignSwarm")

class SovereignSwarm(SubconsciousUnit):
    """
    The 13th Eye: Sovereign AGI (Meta-Cognition).
    
    Role:
    - Does NOT trade directly.
    - Observes Market Regime (Chaos vs Order).
    - Issues "Executive Orders" (Weight Vectors) to the Cortex.
    - Dynamically amplifies or suppresses other Swarms based on the environment.
    """
    def __init__(self):
        super().__init__("Sovereign_Swarm")
        
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Classifies the market into a Regime.
        """
        if df is None or len(df) < 50: return "UNCERTAIN"
        
        # 1. Calculate Volatility (ATR-like)
        high_low = df['high'] - df['low']
        volatility = high_low.rolling(14).mean().iloc[-1]
        avg_price = df['close'].iloc[-1]
        vol_pct = volatility / avg_price
        
        # 2. Calculate Trend Strength (ADX-like approximation)
        # Using simple SMA slope divergence
        sma20 = df['close'].rolling(20).mean()
        sma50 = df['close'].rolling(50).mean()
        
        trend_strength = abs(sma20.iloc[-1] - sma50.iloc[-1]) / avg_price
        
        # 3. Calculate Entropy (Noise)
        # Ratio of body range to total range
        body_size = abs(df['close'] - df['open'])
        candle_range = df['high'] - df['low']
        efficiency = (body_size / candle_range).rolling(10).mean().iloc[-1]
        
        # Classification Logic
        if vol_pct > 0.002 and efficiency < 0.4:
            return "CHAOS" # High Vol, Wicks, No Direction
            
        if trend_strength > 0.005 and efficiency > 0.6:
            return "TRENDING" # Strong Moves, Clean Candles
            
        if vol_pct < 0.0005:
            return "STAGNANT" # Dead Market
            
        return "RANGING" # Default
        
    def get_weights(self, regime: str) -> Dict[str, float]:
        """
        Returns a weight vector for other swarms based on the regime.
        Default weight is 1.0.
        """
        weights = {}
        
        if regime == "CHAOS":
            # Trust Physics & Entropy. Distrust Technicals.
            logger.info("SOVEREIGN: Regime = CHAOS. Amplifying Entropy Engines.")
            weights["ThermodynamicSwarm"] = 2.5
            weights["ChaosSwarm"] = 2.5
            weights["Entropy_Swarm"] = 2.5
            weights["Vortex_Swarm"] = 2.0
            
            weights["TechnicalSwarm"] = 0.2 # Failed in chaos
            weights["TrendingSwarm"] = 0.2
            weights["SMC_Swarm"] = 1.5 # Liquidity grabs happen in chaos
            
        elif regime == "TRENDING":
            # Trust Trend Followers & Institutional Flow
            logger.info("SOVEREIGN: Regime = TRENDING. Amplifying Directional Logic.")
            weights["TrendingSwarm"] = 3.0
            weights["TechnicalSwarm"] = 2.0
            weights["OrderFlowSwarm"] = 2.0
            weights["SmartMoney_Swarm"] = 2.0
            
            weights["RSI_Swarm"] = 0.1 # Oscillators fail in strong trends
            weights["MeanReversion_Swarm"] = 0.1
            
        elif regime == "RANGING":
            # Trust Oscillators & SMC
            logger.info("SOVEREIGN: Regime = RANGING. Amplifying Reversion Logic.")
            weights["RSI_Swarm"] = 2.0
            weights["TechnicalSwarm"] = 1.5
            weights["SmartMoney_Swarm"] = 2.5 # Ping pong between FVGs
            weights["LiquidityMapSwarm"] = 2.0
            
            weights["TrendingSwarm"] = 0.1 # Do not trend trade a range
            weights["Breakout_Swarm"] = 0.5
            
        return weights

    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        df = context.get('df_m5')
        if df is None or len(df) < 50: return None
        
        regime = self.detect_regime(df)
        weights = self.get_weights(regime)
        
        # Meta-Data includes the "Executive Order"
        return SwarmSignal(
            source=self.name,
            signal_type="META_INFO", # Special signal type, ignored by voting summation but read by Orchestrator
            confidence=100.0,
            timestamp=time.time(),
            meta_data={
                'regime': regime,
                'weight_vector': weights
            }
        )
