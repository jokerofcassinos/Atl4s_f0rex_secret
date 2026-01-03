
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger("ManifoldSwarm")

class ManifoldSwarm(SubconsciousUnit):
    """
    The Cartographer.
    Maps the curvature of the Market Manifold using Geodesic Distances.
    """
    def __init__(self):
        super().__init__("Manifold_Swarm")

    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Construct High-Dim Space
        # 4D: Price, Volume, Spread, Volatility (Normed)
        data = df_m5[['close', 'volume', 'high', 'low']].iloc[-40:].copy()
        for c in data.columns:
            data[c] = (data[c] - data[c].mean()) / (data[c].std() + 1e-5)
            
        points = data.values
        
        # 2. Euclidean Distance Matrix
        euclidean_dist = squareform(pdist(points))
        
        # 3. K-Nearest Neighbor Graph (for Geodesic Approx)
        # We connect each point to its k=5 neighbors
        k = 5
        n = len(points)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            dists = euclidean_dist[i]
            idx = np.argsort(dists)
            # Connect k neighbors
            for j in idx[1:k+1]:
                adj_matrix[i, j] = dists[j]
                adj_matrix[j, i] = dists[j] # Sym
                
        # 4. Geodesic Distance (Shortest Path on Graph)
        geodesic_dist = shortest_path(adj_matrix, directed=False)
        
        # 5. Measure Curvature
        # Compare Geodesic End-to-End vs Euclidean End-to-End (Start to Now)
        geo_len = geodesic_dist[0, -1] # Path from Start(0) to End(-1)
        euclid_len = euclidean_dist[0, -1]
        
        # Curvature Ratio
        curvature = 0
        if euclid_len > 0:
            curvature = geo_len / euclid_len
            
        # Interpretation:
        # Curvature ~ 1.0 => Flat Space (Linear Trend).
        # Curvature >> 1.0 => Wrapped/Folded Space (Chaos/Compression).
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        # Logic
        if curvature < 1.5:
            # Linear Regime. Trust the Trend.
            trend = points[-1, 0] - points[0, 0] # Close change
            if trend > 0:
                signal = "BUY"
                confidence = 75
                reason = f"Flat Manifold (Linear Trend, Curv {curvature:.2f})"
            else:
                signal = "SELL"
                confidence = 75
                reason = f"Flat Manifold (Linear Trend, Curv {curvature:.2f})"
        else:
            # High Curvature. Market is twisting.
            # High risk of snaps.
            # We effectively WAIT or trade mean reversion if extreme.
            if curvature > 4.0:
                 # Extreme compression/folding
                 pass
                 
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'curvature': curvature}
            )
            
        return None
