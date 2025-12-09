import numpy as np
import logging

logger = logging.getLogger("Atl4s-Quantum")

class QuantumCore:
    def __init__(self):
        self.planck_const = 1.0 # Normalized h_bar
        self.mass = 1.0 # Normalized price mass

    def calculate_potential_energy(self, price, levels):
        """
        Calculates Potential Energy V(x) based on Support/Resistance.
        """
        V = 0.0
        sigma = 5.0 
        for level in levels:
            V += np.exp(-((price - level)**2) / (2 * sigma**2))
        return V * 100 

    def harmonic_oscillator_psi(self, x, n=0):
        """
        Wave function for Quantum Harmonic Oscillator.
        psi_n(x) ~ H_n(x) * exp(-x^2/2)
        We use ground state (n=0) to represent mean reversion equilibrium.
        """
        # H_0(x) = 1
        return np.exp(-(x**2) / 2)

    def probability_density(self, x_dev, std_dev):
        """
        Calculates probability density of price being at deviation x_dev from mean.
        Models price as a particle in a quadratic potential (mean reverting force).
        """
        # Normalize x by standard deviation (characteristic length)
        # alpha = m*omega / h_bar. Let's simplify to standard normal context.
        if std_dev == 0: return 0
        z = x_dev / std_dev
        
        # Ground state probability density |psi_0|^2
        # proportional to exp(-z^2)
        prob = np.exp(-z**2)
        return prob

    def heisenberg_uncertainty(self, volatility, volume_delta):
        """
        Applies Uncertainty Principle: Delta_x * Delta_p >= h_bar / 2
        
        Delta_p (Momentum Uncertainty): Derived from Volume Delta and Volatility (Kinetic Energy).
        Delta_x (Position Uncertainty): The resulting optimal stop loss with buffer.
        
        If Momentum is very precise (Strong Trend), Position is uncertain -> WIDER STOP.
        If Momentum is uncertain (Choppy), Position is known -> TIGHTER STOP (or wait).
        
        Actually, in trading context, usually:
        High Volatility (High Energy) -> High Uncertainty in next position -> Wider Stop.
        """
        # Let's define Delta_p ~ Change in Momentum
        # momentum p = mass * velocity.
        # velocity ~ volatility.
        
        # We model this as: The more 'explosive' the market (High Energy), 
        # the larger the positional uncertainty.
        
        energy = volatility * (1 + abs(volume_delta))
        
        # K is our scaling constant
        K = 1.5 
        
        # Delta_x >= K * Energy
        # This is a heuristic application.
        uncertainty = K * energy
        return uncertainty

    def tunneling_probability(self, current_price, target_level, kinetic_energy):
        V_barrier = 50.0 
        E = kinetic_energy
        if E > V_barrier: return 1.0
            
        width = abs(target_level - current_price)
        if V_barrier <= E: return 1.0
            
        exponent = -2 * width * np.sqrt(2 * self.mass * (V_barrier - E)) / self.planck_const
        exponent = max(exponent, -100)
        return np.exp(exponent)

    def analyze(self, df, levels):
        """
        Quantum Analysis.
        Returns tunneling probability and Harmonic Oscillator state.
        """
        if df is None or df.empty:
            return {'tunneling_prob': 0.0, 'state': 'COLLAPSED', 'uncertainty': 0.0}
            
        current_price = df.iloc[-1]['close']
        
        # Kinetic Energy
        atr = df.iloc[-1].get('ATR', 1.0)
        
        # Calculate Kinetic Energy from recent momentum
        if len(df) > 5:
            roc = df['close'].diff(3).abs().mean()
        else:
            roc = 0
            
        kinetic_energy = (roc * 10) + atr 
        
        # Tunneling
        if not levels:
            t_prob = 1.0
        else:
            nearest_level = min(levels, key=lambda x: abs(x - current_price))
            t_prob = self.tunneling_probability(current_price, nearest_level, kinetic_energy)
            
        # Harmonic Oscillator State (Mean Reversion)
        # Calculate deviation from Moving Average
        if len(df) > 20:
            ma_20 = df['close'].rolling(20).mean().iloc[-1]
            std_20 = df['close'].rolling(20).std().iloc[-1]
        else:
            ma_20 = current_price
            std_20 = 1.0
        
        deviation = current_price - ma_20
        ground_state_prob = self.probability_density(deviation, std_20)
        
        # Excited State? (Low probability of being here if in equilibrium)
        is_excited = ground_state_prob < 0.05 # > 2 Sigma roughly
        
        # Heisenberg Uncertainty for Stops
        try:
             vol_delta = df['volume'].pct_change().iloc[-1]
        except:
             vol_delta = 0
             
        uncertainty = self.heisenberg_uncertainty(atr, vol_delta)

        return {
            'tunneling_prob': t_prob,
            'ground_state_prob': ground_state_prob,
            'is_excited': is_excited,
            'uncertainty': uncertainty,
            'kinetic_energy': kinetic_energy
        }

    def collapse_state(self, current_price, volume_profile):
        if not volume_profile: return current_price
        nearest = min(volume_profile, key=lambda x: abs(x - current_price))
        return nearest
