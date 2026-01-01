import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from src.macro_math import MacroMath
from analysis.cortex_memory import CortexMemory

logger = logging.getLogger("Atl4s-SeventhEye")

class SeventhEye:
    """
    The Overlord (Synthesis System).
    Highest-level reasoning module.
    - Latent Space Projection (Pattern Recognition vs History)
    - Global Liquidity Map (Stop-Hunt & Trap Detection)
    - Information Geometry (Phase Transitions)
    - Meta-Weighting (Authority Control)
    """
    def __init__(self):
        self.memory = CortexMemory()
        self.atlas = [] # Historical high-performance latent vectors
        self.load_historical_atlas()
        
    def load_historical_atlas(self):
        """Loads a pre-calculated atlas of 'Golden Setup' signatures."""
        # In a real scenario, this would load from a file.
        # Here we initialize with some logic to populate it if empty.
        pass

    def project_latent_space(self, df):
        """
        Projects current market state into a high-dimensional latent vector
        and compares it with the 'Historical Atlas' using Cosine Similarity.
        """
        if df is None or len(df) < 30:
            return 0, 0
            
        # Extract features (price action, volatility, volume, kinematics)
        # CortexMemory current extracts 4 features: [f_rsi, f_vol, f_roc, hour]
        raw_features = self.memory.extract_features(df)
        if not raw_features: return 0, 0
        
        # We need a consistent 5-dim vector for our comparison: [Trend, Vol, Mom, Delta, Entropy]
        # We'll map the available features and fill the rest
        
        # Normalized Delta (Close - Open)
        delta_norm = (df['close'].iloc[-1] - df['open'].iloc[-1]) / (df['high'].iloc[-1] - df['low'].iloc[-1] + 1e-9)
        
        # Entropy (from QuantumMath or approximated if needed, here we use a placeholder or calculate)
        # For efficiency, we'll use ROC as a proxy for Momentum and Vol for Vol.
        
        current_vec = np.array([
            raw_features[2], # Trend/Momentum Proxy (ROC)
            raw_features[1], # Volatility Proxy
            raw_features[0], # RSI Proxy
            delta_norm,      # Real-time Delta
            0.1              # Baseline Entropy
        ])
        
        # Mock 'Golden' Vectors
        golden_long = np.array([0.8, 0.2, 0.7, 0.9, 0.1])
        golden_short = np.array([-0.8, 0.2, -0.7, -0.9, 0.1])
        
        curr_norm = np.linalg.norm(current_vec)
        if curr_norm == 0: return 0, 0
        
        sim_long = np.dot(current_vec, golden_long) / (curr_norm * np.linalg.norm(golden_long))
        sim_short = np.dot(current_vec, golden_short) / (curr_norm * np.linalg.norm(golden_short))
        
        return sim_long, sim_short

    def map_global_liquidity(self, df):
        """
        Detects 'Liquidity Pools' and potential 'Stop Runs'.
        Identifies areas where large orders are likely clustered.
        """
        if df is None or len(df) < 100:
            return 0, "NEUTRAL"
            
        recent_highs = df['high'].rolling(50).max().iloc[-1]
        recent_lows = df['low'].rolling(50).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Proximity to major 'Swing Levels'
        dist_high = (recent_highs - current_price) / current_price
        dist_low = (current_price - recent_lows) / current_price
        
        # If we are very close to a major high/low but haven't broken it, 
        # there is high 'Magnetism' (Informed capital targeting stops).
        magnetism = 0
        bias = "NEUTRAL"
        
        threshold = 0.001 # 0.1% for Gold is significant
        
        if dist_high < threshold:
            magnetism = (threshold - dist_high) / threshold
            bias = "LIQUIDITY_HUNT_UP"
        elif dist_low < threshold:
            magnetism = (threshold - dist_low) / threshold
            bias = "LIQUIDITY_HUNT_DOWN"
            
        return magnetism, bias

    def analyze_phase_transition(self, df):
        """
        Uses Fisher Information Curvature to detect 'Phase Transitions'.
        High curvature spike often precedes large directional moves.
        """
        from src.quantum_math import QuantumMath
        if df is None or len(df) < 50:
            return 0
            
        try:
            curvature = QuantumMath.fisher_information_curvature(df['close'], last_only=True)
            return curvature
        except:
            return 0

    def deliberate(self, data_map):
        """
        Synthesizes all sensors into a 'Master Directive'.
        """
        df_m5 = data_map.get('M5')
        if df_m5 is None: return {'decision': 'WAIT', 'score': 0, 'reason': 'No Data'}
        
        # 1. Analogical Reasoning (Comparison to History)
        sim_long, sim_short = self.project_latent_space(df_m5)
        
        # 2. Liquidity Awareness
        mag, liq_bias = self.map_global_liquidity(df_m5)
        
        # 3. Structural Explosion Potential
        curvature = self.analyze_phase_transition(df_m5)
        
        # 4. Master Logic
        score = 0
        reason = "Synthesis"
        
        # If High Similarity to 'Golden Long' and no bearish liquidity traps
        if sim_long > 0.8:
            score += 40
            reason = "Latent Golden Long"
        elif sim_short > 0.8:
            score -= 40
            reason = "Latent Golden Short"
            
        # Liquidity Boost (Magnetism)
        if liq_bias == "LIQUIDITY_HUNT_UP":
            score += 20 * mag
            reason += " + Liquidity Magnet Up"
        elif liq_bias == "LIQUIDITY_HUNT_DOWN":
            score -= 20 * mag
            reason += " + Liquidity Magnet Down"
            
        # Phase Transition Multiplier
        # If curvature is high, the market is 'ready' to move. 
        # It amplifies existing bias.
        if curvature > 2.0:
            score *= 1.5
            reason += " [VOLATILITY EXPLOSION IMMINENT]"
            
        decision = "WAIT"
        if score > 50: decision = "BUY"
        elif score < -50: decision = "SELL"
        
        return {
            'decision': decision,
            'score': score,
            'reason': reason,
            'metrics': {
                'latent_sim': max(sim_long, sim_short),
                'magnetism': mag,
                'curvature': curvature
            }
        }
