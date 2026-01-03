
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("ChaosSwarm")

class ChaosSwarm(SubconsciousUnit):
    """
    The Physicist.
    Measures Order vs Chaos.
    1. Lyapunov Exponent (Stability)
    2. Shannon Entropy (Information Density)
    """
    def __init__(self):
        super().__init__("Chaos_Swarm")
        self.embedding_dim = 3
        self.lag = 2

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 100: return None
        
        # 1. Calculate Physics Metrics
        lyapunov = self._calculate_lyapunov(df_m5)
        # Simple Entropy estimate using Price Probability
        entropy = self._calculate_entropy(df_m5)
        
        # Interpret
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        # State: ORDERED (Low Lyapunov)
        if lyapunov < 0.005: 
            # System is Stable. Trends are persistent.
            # We check the Trend direction (simple check here, or rely on Orchestrator fusion)
            # ChaosSwarm primarily acts as a CONFIDENCE MODIFIER.
            # But if asked to Signal, it signals based on persistence.
            
            # Simple Trend check
            closes = df_m5['close']
            trend = closes.iloc[-1] - closes.iloc[-20]
            
            if abs(trend) > 0:
                direction = "BUY" if trend > 0 else "SELL"
                # If Entropy is also low -> Laminar Flow (Super Strong)
                if entropy < 0.5:
                    confidence = 90
                    reason = f"Laminar Flow: Stable (L={lyapunov:.4f}) + Low Entropy ({entropy:.2f})"
                else:
                    confidence = 60
                    reason = f"Stable Trend: (L={lyapunov:.4f})"
                signal = direction
                
        # State: CHAOTIC (High Lyapunov)
        elif lyapunov > 0.02:
            # System is diverging fast. Breakout or Crash.
            # Usually implies high volatility.
            # If Entropy is High -> Just Noise (Veto).
            # If Entropy is Low (Structured Chaos) -> Fractal Breakout.
            
            if entropy < 0.6:
                 # Structured Breakout
                 # We need instantaneous momentum
                 mom = df_m5['close'].diff().iloc[-1]
                 if mom != 0:
                     direction = "BUY" if mom > 0 else "SELL"
                     confidence = 75
                     reason = f"Fractal Breakout: High Chaos (L={lyapunov:.4f}) but Structure holds"
                     signal = direction
            else:
                 # Pure Noise
                 # Veto logic usually, but here we just return Low Confidence WAIT
                 pass
                 
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'lyapunov': lyapunov, 'entropy': entropy, 'reason': reason}
            )
            
        return None

    def _calculate_lyapunov(self, df):
        # Accelerated Rosenstein
        try:
             # Prepare Data
            data = np.log(df['close'].values[-200:]) # Last 200 points for better phase space
            N = len(data)
            M = N - (self.embedding_dim - 1) * self.lag
            
            if M < 20: return 0.0
            
            # Construct Orbit for JIT
            orbit = np.zeros((M, self.embedding_dim))
            for i in range(M):
                for j in range(self.embedding_dim):
                    orbit[i, j] = data[i + j*self.lag]
                    
            from core.acceleration import jit_lyapunov_search
            lyapunov = jit_lyapunov_search(orbit, M, self.lag, self.embedding_dim)
            return lyapunov
            
        except Exception as e:
             # logger.error(f"Lyapunov JIT Error: {e}")
             return 0.0

    def _calculate_entropy(self, df):
        # Shannon Entropy of Returns Distribution
        returns = df['close'].pct_change().dropna()[-50:]
        if len(returns) < 10: return 1.0
        
        # Histogram
        counts, _ = np.histogram(returns, bins=10, density=True)
        # Normalize
        probs = counts / np.sum(counts)
        probs = probs[probs > 0] # Remove zeros
        
        entropy = -np.sum(probs * np.log2(probs))
        
        # Max Entropy for 10 bins = log2(10) ~ 3.32
        # Normalize 0-1
        norm_entropy = entropy / 3.32
        return norm_entropy
