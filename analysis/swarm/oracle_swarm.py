
import logging
from core.interfaces import SubconsciousUnit, SwarmSignal
from analysis.monte_carlo_fractal import FractalMonteCarlo

logger = logging.getLogger("OracleSwarm")

class OracleSwarm(SubconsciousUnit):
    """
    The Prophet.
    Simulates thousands of futures to determine the most probable path.
    """
    def __init__(self):
        super().__init__("Oracle_Swarm")
        self.oracle = FractalMonteCarlo()

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 100: return None
        
        # Run Simulation
        # This can be CPU intensive. We should consider only running it 
        # on new candles or significant price moves.
        # Ideally, this runs async in a separate thread/process, but here we await.
        
        # We assume oracle.generate_projection is optimized
        forecast = self.oracle.generate_projection(df_m5) 
        
        if not forecast: return None
        
        # Forecast structure: {'bullish_prob': 0.7, 'bearish_prob': 0.3}
        bull_prob = forecast.get('bullish_prob', 0.5)
        bear_prob = forecast.get('bearish_prob', 0.5)
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        if bull_prob > 0.75:
            signal = "BUY"
            confidence = (bull_prob - 0.5) * 200 # 0.75 -> 50% conf? No. 0.75 -> 50. 0.8 -> 60.
            # Rework confidence scaling
            # 0.5 = 0 conf. 1.0 = 100 conf.
            confidence = (bull_prob - 0.5) * 2 * 100
            reason = f"Oracle Prophecy: {bull_prob:.1%} Bullish Probability"
            
        elif bear_prob > 0.75:
            signal = "SELL"
            confidence = (bear_prob - 0.5) * 2 * 100
            reason = f"Oracle Prophecy: {bear_prob:.1%} Bearish Probability"
            
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'bull_prob': bull_prob, 'bear_prob': bear_prob}
            )
            
        return None
