
import logging
from typing import Dict

logger = logging.getLogger("Neuroplasticity")

class NeuroplasticityLoop:
    """
    The Adaptation Engine.
    Adjusts Synaptic Weights based on Agent Accuracy.
    """
    def __init__(self):
        # Base Weights
        self.weights = {
            'Trending_Swarm': 1.0,
            'Sniper_Swarm': 1.0,
            'Quant_Swarm': 1.0,
            'Kinematic_Swarm': 2.0, # High Authority (Physics > Opinion)
            'Veto_Swarm': 2.0, # High authority
            'Whale_Swarm': 1.5,
            'Quantum_Grid_Swarm': 1.2,
            'Game_Swarm': 1.0,
            'Chaos_Swarm': 0.8,
            'Macro_Swarm': 1.0,
            'Fractal_Vision_Swarm': 1.0,
            'Oracle_Swarm': 1.8, # High trust in Monte Carlo
            'Reservoir_Swarm': 1.2, # Neural intuition
            'Sentiment_Swarm': 1.3, # Psychology
            'Order_Flow_Swarm': 1.5, # The Tape
            'Liquidity_Map_Swarm': 1.6, # Stop Hunts
            'Causal_Graph_Swarm': 1.4, # The Logic
            'Counterfactual_Engine': 1.2, # The Imagination
            'Spectral_Swarm': 1.3, # The Cycles
            'Wavelet_Swarm': 1.5, # The Filter
            'Topological_Swarm': 1.4, # The Geometer
            'Manifold_Swarm': 1.4, # The Cartographer
            'Associative_Swarm': 1.3, # The Librarian
            'Path_Integral_Swarm': 1.6, # The Physicist
            'Singularity_Swarm': 2.5, # THE UNIFIED FIELD (Max Authority)
            'Singularity_Swarm': 2.5, # THE UNIFIED FIELD (Max Authority)
            'Hyperdimensional_Swarm': 2.2, # CHAOS DIVERGENCE (Safety Hedge)
            'Vortex_Swarm': 2.1, # TRAP DETECTOR (Wick Rejection)
            'DNA_Swarm': 2.0, # EVOLUTIONARY MEMORY (Genetic Match)
            'Schrodinger_Swarm': 1.9, # QUANTUM TUNNELING (Breakout Physics)
            'Antimatter_Swarm': 1.5, # PARITY CHECK (Symmetry Validator)
            'Heisenberg_Swarm': 1.7, # UNCERTAINTY PRINCIPLE (Volatility Squeeze)
            'Navier_Stokes_Swarm': 1.6, # FLUID DYNAMICS (Laminar vs Turbulent)
            'Dark_Matter_Swarm': 1.8, # MASS-TO-LIGHT RATIO (Invisible Liquidity)
            'Holographic_Swarm': 1.7, # ENTROPY (AdS/CFT)
            'Superluminal_Swarm': 2.1, # FFT PRECOGNITION (Tachyonic Antitelephone)
            'Event_Horizon_Swarm': 1.8, # GENERAL RELATIVITY (Gravity Well)
            'Lorentz_Swarm': 2.2, # CHAOS THEORY (Bifurcation Detector)
            'Minkowski_Swarm': 1.9, # SPECIAL RELATIVITY (Spacetime Interval)
            'Higgs_Swarm': 2.0, # PARTICLE PHYSICS (Mass Generation / Liquidity Drag)
            'Boltzmann_Swarm': 1.8, # THERMODYNAMICS (Free Energy)
            'Fermi_Swarm': 2.1, # NUCLEAR PHYSICS (Decay Half-Life)
            'Bose_Einstein_Swarm': 2.3, # QUANTUM COHERENCE (The Laser Beam)
            'Schrodinger_Newton_Swarm': 1.9, # SELF-GRAVITATING WAVEFUNCTION (Price Collapse)
            'Tachyon_Swarm': 2.5, # RETROCAUSAL TRAP DETECTOR (Imaginary Mass)
            'Feynman_Swarm': 2.4, # PATH INTEGRAL (Least Action Exit)
            'Maxwell_Swarm': 2.2, # ELECTRODYNAMICS (Back-EMF Rejection)
            'Heisenberg_Swarm': 2.6, # UNCERTAINTY PRINCIPLE (Particle vs Wave State)
            'Riemann_Swarm': 2.7, # DIFFERENTIAL GEOMETRY (Metric Curvature)
            'Penrose_Swarm': 2.8, # TWISTOR THEORY (Aeon Transition)
            'Game_Swarm': 2.9, # EVOLUTIONARY GAME THEORY (Replicator Dynamics)
            'Godel_Swarm': 3.0 # META-LOGIC (Incompleteness Veto)
        }
        
        # Performance Trackers (Rolling Accuracy)
        self.performance = {k: 0.5 for k in self.weights.keys()} # Start at 50%
        self.history = {k: [] for k in self.weights.keys()}

    def register_outcome(self, signals: Dict[str, str], actual_move: float):
        """
        After a trade (or candle close), rate the agents.
        signals: {'Trending_Swarm': 'BUY', 'Sniper_Swarm': 'WAIT', ...}
        actual_move: +10.5 (price delta)
        """
        direction = 1 if actual_move > 0 else -1
        
        for agent_name, signal in signals.items():
            if agent_name not in self.weights: continue
            
            score = 0
            if signal == "BUY":
                score = 1 if direction == 1 else 0
            elif signal == "SELL":
                score = 1 if direction == -1 else 0
            else:
                continue # WAIT signals are neutral, or maybe penalized if opportunity missed?
                
            # Update History
            self.history[agent_name].append(score)
            if len(self.history[agent_name]) > 50: # Keep last 50 decisions
                self.history[agent_name].pop(0)
                
            # Recalculate Performance
            avg_acc = sum(self.history[agent_name]) / len(self.history[agent_name])
            self.performance[agent_name] = avg_acc
            
            # Adjust Weight: Plasticity Formula
            # Weight = Base * (Accuracy / 0.5) 
            # If Acc = 0.5 (Random) -> Weight = Base
            # If Acc = 0.8 -> Weight = Base * 1.6
            # If Acc = 0.2 -> Weight = Base * 0.4
            
            # Dampening factor to prevent runaways
            plasticity = pow(avg_acc / 0.5, 2) # Quadratic reward for high excellence
            
            # Update
            # We don't change 'base', we assume current self.weights IS the dynamic weight? 
            # Or we store base separately? For now, adapt the running weight.
            # Actually better to keep Base and Dynamic separated.
            # But for simplicity, let's just return the Modifier.
            
    def get_dynamic_weights(self) -> Dict[str, float]:
        """
        Returns the current optimized weights.
        """
        dynamic_weights = {}
        for name, base_w in self.weights.items():
            acc = self.performance.get(name, 0.5)
            # Safe clamping
            acc = max(0.1, min(0.9, acc))
            
            modifier = (acc / 0.5)
            dynamic_weights[name] = base_w * modifier
            
        return dynamic_weights
