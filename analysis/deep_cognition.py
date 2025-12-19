import numpy as np
from analysis.cortex_memory import CortexMemory
from src.quantum_math import QuantumMath
from analysis.microstructure import MicroStructure
from analysis.kinematics import Kinematics
from analysis.prediction_engine import PredictionEngine

class DeepCognition:
    def __init__(self):
        self.memory = CortexMemory()
        self.micro = MicroStructure()
        self.qm = QuantumMath()
        self.kinematics = Kinematics()
        self.oracle = PredictionEngine()
        
    def consult_subconscious(self, trend_score, volatility_score, pattern_score, smc_score, df_m5=None, live_tick=None):
        """
        Weighted consensus of different 'brain' parts, augmented by:
        - Cortex (Memory)
        - Quantum (Entropy/Kalman)
        - Microstructure (Flow)
        - Kinematics (Physical Force)
        - Oracle (Pre-Cognition)
        """
        
        # 0. Update Microstructure (Fast Brain)
        if live_tick:
            self.micro.on_tick(live_tick)
            
        micro_metrics = self.micro.analyze()
        
        # 1. Base Instincts
        # User Feedback: Trend was correct while Physics was too cautious.
        # Adjustment: Boost Trend Weight to trust Technicals more.
        w_trend = 0.35 
        w_vol = 0.1
        w_pat = 0.15
        w_smc = 0.20
        w_micro = 0.10 
        w_phy = 0.10 # Reduced Physics weight to avoid over-caution
        
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
             
        instinct_signal = (trend_score * w_trend) + \
                          (volatility_score * w_vol) + \
                          (pattern_score * w_pat) + \
                          (smc_score * w_smc) + \
                          (micro_score * w_micro) + \
                          (phy_score * w_phy)
                          
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
            entropy_latest = self.qm.calculate_entropy(closes, window=14).iloc[-1]
            
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

        # 4. Final Consensus
        # Mix: Instinct (Current) + Memory (Past) + Oracle (Future)
        final_decision = (instinct_norm * 0.5) + (memory_bias * 0.3) + (future_bias * 0.2)
        
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
             
        # Detect Energy
        state_label = "NEUTRAL"
        if orbit_energy > 2.5:
            state_label = "HIGH_ENERGY"
        elif entropy_latest > 0.8: # Using entropy if defined
            state_label = "CHAOTIC"
            
        if is_super_confluence:
            state_label += " [SUPER]"
             
        return final_decision, state_label, bullish_prob_future

    def learn(self, df, outcome):
        features = self.memory.extract_features(df)
        self.memory.store_experience(features, outcome)

    def analyze_risk_perception(self, recent_losses, market_volatility):
        fear_factor = 1.0
        if recent_losses > 0: fear_factor *= 0.5 
        if market_volatility > 0.001: fear_factor *= 0.8
        return fear_factor
