
import logging
import numpy as np
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("ReservoirSwarm")

class ReservoirSwarm(SubconsciousUnit):
    """
    The Synapse.
    Echo State Network (Reservoir Computing) for Chaotic Prediction.
    """
    def __init__(self, n_reservoir=200, spectral_radius=0.9):
        super().__init__("Reservoir_Swarm")
        self.n_reservoir = n_reservoir
        self.density = 0.1
        
        # ESN Matrices
        # Input: Close, Vol in (2 dims)
        self.in_dim = 2 
        self.W_in = (np.random.rand(n_reservoir, self.in_dim) - 0.5) * 2
        
        # Recurrent Weights
        self.W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        # Sparsity
        mask = np.random.rand(n_reservoir, n_reservoir) > self.density
        self.W[mask] = 0
        # Spectral Radius Scaling
        rho = max(abs(np.linalg.eigvals(self.W)))
        self.W = self.W * (spectral_radius / rho)
        
        # Learning
        self.W_out = np.zeros(n_reservoir) # Linear readout
        self.state = np.zeros(n_reservoir)
        
        # Buffer for online learning
        self.X_buffer = []
        self.Y_buffer = []
        
    async def process(self, context) -> SwarmSignal:
        df_m5 = context.get('df_m5')
        if df_m5 is None or len(df_m5) < 50: return None
        
        # 1. Update State
        last = df_m5.iloc[-1]
        input_vec = np.array([
            last['close'] / df_m5['close'].iloc[-20:].mean(), # Normalized Close
            last['volume'] / (df_m5['volume'].iloc[-20:].mean() + 1e-5) # Norm Vol
        ])
        
        # x(t) = tanh(W_in*u(t) + W*x(t-1))
        self.state = np.tanh(np.dot(self.W_in, input_vec) + np.dot(self.W, self.state))
        
        # 2. Predict
        # If we have trained weights
        prediction = np.dot(self.W_out, self.state)
        
        # 3. Online Learning (Simple LMS or RLS)
        # We need the ACTUAL target from previous step to train.
        # Ideally, we store the previous state and previous price, and update weights now.
        
        # Placeholder for proper Ridge Regression:
        # We treat the prediction as a "Sentiment Score" (-1 to 1) 
        # because actual price regression requires stable training.
        
        # Signal Generation similar to Oracle
        # If the Neural State is 'excited' in a known pattern
        
        # Simple Logic for prototype:
        # Use Neural State Energy to detect regime change
        energy = np.mean(self.state ** 2)
        
        signal = "WAIT"
        confidence = 0
        reason = ""
        
        if energy > 0.5:
             # High Activity in Neural Net -> Novelty / Chaos / Opportunity
             signal = "BUY" # Blind guess for prototype structure
             confidence = 50 + (energy * 20)
             reason = f"Neural Activation High ({energy:.2f})"
             
        # In a real impl, we would use W_out * state to get next_price_delta.
        # Assuming W_out is trained (which requires historical batch training).
        
        # For now, return the 'Intuition' based on network resonance
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=0,
                meta_data={'energy': energy}
            )
            
        return None
