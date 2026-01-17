import logging
import numpy as np
import pandas as pd
from analysis.topology_engine import TopologyEngine

logger = logging.getLogger("Atl4s-NinthEye")

class NinthEye:
    """
    The Singularity (Geometric System).
    Analyzes Information Geometry and Manifold Folding.
    - Geometric Flow Detection
    - Topological Hole Persistence (Extreme Order)
    - Probability Density Collapse
    """
    def __init__(self):
        self.topology = TopologyEngine()

    def analyze_manifold_geometry(self, df):
        """
        Detects if the market manifold is 'folding' into a Singularity.
        A state where multiple high-dimensional paths converge into one.
        """
        if df is None or len(df) < 50:
            return 0
            
        prices = df['close'].values
        # Using TopologyEngine to find 'Loops' vs 'Lines' in phase space
        loop_score, betti_1 = self.topology.analyze_persistence(prices)
        
        # If loop_score is extremely high (> 10), the orbit is perfectly periodic.
        # If betti_1 is 0 but loop_score is decent (> 2), we have high curvature.
        
        singularity_factor = 0
        if loop_score > 8.0:
            # Perfection recurrence - Singularity approaching (Cycle completion)
            singularity_factor = 0.8
        elif loop_score > 4.0:
            singularity_factor = 0.4
            
        return singularity_factor

    def probability_collapse(self, df):
        """
        Measures the concentration of Price-Volume density.
        If the density 'collapses' into a narrow range, an explosion is likely.
        """
        if df is None or len(df) < 30:
            return 0
            
        # Histogram of recent prices weighted by volume
        prices = df['close'].values
        volumes = df['volume'].values
        
        try:
            hist, bin_edges = np.histogram(prices, bins=10, weights=volumes, density=False)
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum
                # Concentration = Peak value vs average value
                concentration = np.max(hist) / (np.mean(hist) + 1e-9)
            else:
                concentration = 0.0
        except Exception:
            concentration = 0.0
        
        # Normalized concentrate (0 to 1)
        # Threshold: > 5.0 is extreme concentration
        return min(1.0, concentration / 5.0)

    def deliberate(self, data_map):
        """
        Checks for Singularity Alignment.
        """
        df_m5 = data_map.get('M5')
        if df_m5 is None: return {'decision': 'WAIT', 'score': 0}
        
        geo_flow = self.analyze_manifold_geometry(df_m5)
        prob_density = self.probability_collapse(df_m5)
        
        # Mastery logic
        # A Singularity occurs when geometry and density both align
        score = (geo_flow * 0.6 + prob_density * 0.4) * 100
        
        direction = "WAIT"
        # The Singularity eye represents ABSOLUTE CERTAINTY of a regime shift.
        # It doesn't decide direction alone, it reinforces the 'State of Execution'.
        if score > 70:
            direction = "SINGULARITY_REACHED"
            
        return {
            'decision': direction,
            'score': score,
            'geometry': geo_flow,
            'density': prob_density
        }
