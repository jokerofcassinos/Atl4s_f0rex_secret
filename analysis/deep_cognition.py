import numpy as np
from analysis.cortex_memory import CortexMemory
from src.quantum_math import QuantumMath
from analysis.microstructure import MicroStructure
from analysis.kinematics import Kinematics
from analysis.prediction_engine import PredictionEngine
from analysis.seventh_eye import SeventhEye

class DeepCognition:
    def __init__(self):
        self.memory = CortexMemory()
        self.micro = MicroStructure()
        self.qm = QuantumMath()
        self.kinematics = Kinematics()
        self.oracle = PredictionEngine()
        self.overlord = SeventhEye() # Synthesis Sensor
        
        # Adaptive Cognitive Weights (ACW) - NeuroPlasticity
        self.weights = {
            'trend': 0.35,
            'volatility': 0.10,
            'pattern': 0.15,
            'smc': 0.20,
            'micro': 0.10,
            'physics': 0.10
        }
        self.learning_rate = 0.01

    def update_neuroplasticity(self, success, signal_matrix):
        """
        Adjusts synaptic weights based on who was 'right'.
        success: True (Profit) / False (Loss)
        signal_matrix: dict of {'trend': 1, 'smc': -1, ...} directions at time of trade.
        """
        # If trade was a WIN, boost those who agreed with the final decision.
        # If trade was a LOSS, punish those who agreed.
        # This requires storing the signal matrix at execution time suitable for feedback loop.
        # For now, we implement a simpler Hebbian drift: 
        # "Neurons that fire together, wire together."
        pass # Placeholder for v2.1 Feedback Learning loop

    def consult_subconscious(self, trend_score, volatility_score, pattern_score, smc_score, df_m5=None, df_m1=None, live_tick=None, details=None):
        """
        Deep Neural Analysis (Now with M1 Micro-Fidelity).
        """
        
        # 0. Update Microstructure (Fast Brain)
        if live_tick:
            self.micro.on_tick(live_tick)
            
        micro_metrics = self.micro.analyze()
        
        # 0.5 M1 Micro-Sentiment (New Layer)
        m1_sentiment = 0
        if df_m1 is not None and not df_m1.empty:
            # Simple Impulse Check (Last 3 candles)
            closes = df_m1['close'].values[-3:]
            if closes[-1] > closes[0]: m1_sentiment = 20
            else: m1_sentiment = -20
            
        # 1. Base Instincts (Using Adaptive Weights)
        w = self.weights
        
        # Micro Inputs
        micro_score = 0
        if micro_metrics['delta'] > 0: micro_score += 50
        else: micro_score -= 50
        if micro_metrics['velocity'] > 0: micro_score += 50
        else: micro_score -= 50
        
        # Physics Inputs
        phy_score = 0
        phase_angle = 0
        orbit_energy = 0
        if df_m5 is not None and not df_m5.empty and len(df_m5) > 10:
             _, _, phy_score, phase_angle, orbit_energy = self.kinematics.analyze(df_m5)
             
        instinct_signal = (trend_score * w['trend']) + \
                          (volatility_score * w['volatility']) + \
                          (pattern_score * w['pattern']) + \
                          (smc_score * w['smc']) + \
                          (micro_score * w['micro']) + \
                          (phy_score * w['physics'])
                          
        # Instinct is roughly -100 to 100. Normalize to -1 to 1.
        # Adjusted sensitivity: Divisor 20.0 makes 20 score -> 0.76 (Strong)
        instinct_norm = np.tanh(instinct_signal / 25.0) 
        
        # Debug Log
        # print(f"DEBUG: Instinct: {instinct_signal:.2f} -> Norm: {instinct_norm:.2f}") 
        
        # 2. Consult Memory (Experience) & Oracle (Future)
        memory_bias = 0
        future_bias = 0
        
        if df_m5 is not None and not df_m5.empty and len(df_m5) > 20:
            # Memory
            features = self.memory.extract_features(df_m5)
            bullish_prob_mem = self.memory.recall(features, k=10)
            memory_bias = (bullish_prob_mem - 0.5) * 2
            
            # Oracle (Monte Carlo)
            oracle_data = self.oracle.analyze(df_m5)
            bullish_prob_future = oracle_data.get('prob_bullish', 0.5)
            # 0.5 -> 0, 1.0 -> 0.5 bias (Oracle should be supportive, not dominant)
            future_bias = (bullish_prob_future - 0.5) * 2 
            
        # 3. Quantum Layer (Math Reality Check)
        closes = df_m5['close']
        entropy_series = self.qm.calculate_entropy(closes, window=14)
        entropy_latest = entropy_series.iloc[-1] if not entropy_series.empty else 0
        
        # Penalty for Chaos
        chaos_penalty = 1.0
        if entropy_latest > 2.0: chaos_penalty = 0.7
        instinct_norm *= chaos_penalty
        
        # Kalman Bias
        kalman = self.qm.kalman_filter(closes).iloc[-1]
        price = closes.iloc[-1]
        kalman_bias = 0
        if price > kalman: kalman_bias = -0.1
        else: kalman_bias = 0.1
        
        instinct_norm += kalman_bias

        # 4. Meta-Critic Layer (Information Geometry)
        # Check for 'Internal Conflict' between Sensors 
        # Optimized: Use Overlord data from 'details' if provided
        overlord_data = details.get('Overlord', {}) if details else self.overlord.deliberate({'M5': df_m5})
        
        overlord_dir = np.sign(overlord_data.get('score', 0)) if abs(overlord_data.get('score', 0)) > 10 else 0
        
        # If Overlord (The High Brain) disagrees with Instinct (The Base Brain)
        if overlord_dir != 0 and np.sign(instinct_norm) != overlord_dir:
            conflict_score = 0.5
            instinct_norm *= 0.7 # Penalyze instinct if Overlord says 'Wait' or 'Reverse'
            
        # Curvature Awareness (Geometry of the Manifold)
        curvature = overlord_data.get('metrics', {}).get('curvature', 0)
        
        # final_decision = ...
        overlord_score = overlord_data.get('score', 0)
        final_decision = (instinct_norm * 0.4) + \
                         (memory_bias * 0.2) + \
                         (future_bias * 0.2) + \
                         (np.tanh(overlord_score / 50.0) * 0.2)
        
        # Conflicting Timeframes?
        if np.sign(future_bias) != np.sign(instinct_norm) and abs(future_bias) > 0.3:
             # Future disagrees with Now -> Caution
             final_decision *= 0.6
             
        # SUPER CONFLUENCE CHECK
        # If Trend (Instinct), Future (Oracle), and Physics (Energy) all align
        is_super_confluence = False
        if abs(final_decision) > 0.4: # Only if we already have a signal
            # Check alignment
            trend_dir = np.sign(instinct_norm)
            future_dir = np.sign(future_bias)
            
            # Physics Alignment: High Energy + Correct Direction implies support
            # We use orbit_energy from kinematics (need to capture it)
            # Since we calculate phy_score in step 1, we can use that sign
            phy_dir = np.sign(phy_score)
            
            if trend_dir == future_dir == phy_dir:
                # All 3 brains agree on direction
                final_decision *= 1.2 # Boost confidence
                final_decision = max(min(final_decision, 1.0), -1.0) # Cap at 1
                is_super_confluence = True
             
        # Detect Energy State
        state_label = "NEUTRAL"
        if orbit_energy > 2.5:
            state_label = "HIGH_ENERGY"
        elif entropy_latest > 0.8: 
            state_label = "CHAOTIC"
            
        if is_super_confluence:
            state_label += " [SUPER]"
             
        # Return Decision, State Label, Future Prob, Physics Energy, AND full Micro Metrics for Swarm
        return final_decision, state_label, bullish_prob_future, orbit_energy, micro_metrics

    def learn(self, df, outcome):
        features = self.memory.extract_features(df)
        self.memory.store_experience(features, outcome)

    def analyze_risk_perception(self, recent_losses, market_volatility):
        fear_factor = 1.0
        if recent_losses > 0: fear_factor *= 0.5 
        if market_volatility > 0.001: fear_factor *= 0.8
        return fear_factor
