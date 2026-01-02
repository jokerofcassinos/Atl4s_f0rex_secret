import logging
import numpy as np
import pandas as pd
import os
import json
import config

logger = logging.getLogger("Atl4s-Architect")

class TenthEye:
    """
    The Holographic Architect (System X).
    
    A Strategic Commander that oversees the entire bot ecosystem.
    Functions:
    1. Strategic Directives: Sets global mode (AGGRESSIVE, DEFENSIVE, SURVIVAL) based on Chaos/Hurst.
    2. Authority Allocation: Dynamically re-weights Eyes based on the Directive.
    3. Mental Sandbox: Simulates conflicting signals to find the 'True Path'.
    4. System Health: Monitoring reliability of all sub-systems.
    """
    def __init__(self):
        self.stats_file = os.path.join(config.CACHE_DIR, "eye_reliability.json")
        self.eye_weights = {f"Eye{i}": 1.0 for i in range(1, 10)}
        self.current_directive = "NEUTRAL"
        self.load_stats()
        
    def load_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.eye_weights = json.load(f)
            except:
                pass

    def save_stats(self):
        try:
             with open(self.stats_file, 'w') as f:
                 json.dump(self.eye_weights, f)
        except:
             pass

    def determine_directive(self, market_regime, volatility_score, hurst, lyapunov):
        """
        Calculates the Strategic Directive for the next tick.
        """
        # Default
        directive = "BALANCED"
        
        # 1. Chaos Check (Lyapunov)
        if lyapunov > 1.0:
            directive = "DEFENSIVE" # High Chaos -> Protect Capital
            if volatility_score > 80:
                directive = "SURVIVAL" # Extreme Volatility -> Shutdown/Min Size
        
        # 2. Trend Check (Hurst)
        elif hurst > 0.65:
            directive = "AGGRESSIVE_TREND" # Smooth Trend -> Maximize Drift
            
        # 3. Mean Reversion Check
        elif hurst < 0.35:
            directive = "SNIPER_AMBUSH" # Choppy -> Prioritize Snipers
            
        return directive

    def allocate_authority(self, directive):
        """
        Re-weights the Eyes based on the Directive.
        """
        weights = self.eye_weights.copy()
        
        if directive == "AGGRESSIVE_TREND":
            # Boost Trend Eyes (Whale, Trend Architect, Swarm)
            weights['Eye1'] *= 1.5 # Swarm
            weights['Eye3'] *= 1.2 # Trend
            weights['Eye4'] *= 1.5 # Whale
            
        elif directive == "SNIPER_AMBUSH":
            # Boost Reversion Eyes (Sniper, Oscillator, Cycle)
            weights['Eye2'] *= 1.5 # Sniper
            weights['Eye6'] *= 1.3 # Quant
            weights['Eye9'] *= 1.3 # Cycle
            
        elif directive == "DEFENSIVE":
            # Boost Risk & Logic, Penalize Aggression
            weights['Eye1'] *= 0.5
            weights['Eye2'] *= 0.5
            weights['Eye5'] *= 1.5 # Fortress (Support/Resistance)
            
        elif directive == "SURVIVAL":
            # Shutdown mostly
            for k in weights: weights[k] *= 0.1
            
        return weights

    def record_outcome(self, eye_decisions, actual_outcome):
        """
        Reinforcement Learning: Update base weights based on success.
        """
        for eye, decision in eye_decisions.items():
            if eye not in self.eye_weights: continue
            
            if decision == actual_outcome:
                self.eye_weights[eye] = min(2.0, self.eye_weights[eye] + 0.05)
            else:
                self.eye_weights[eye] = max(0.1, self.eye_weights[eye] - 0.10)
        
        self.save_stats()

    def deliberate(self, results_map, market_state):
        """
        The Architect's Final Judgement.
        Args:
            results_map: Output from all eyes.
            market_state: Dict with 'hurst', 'lyapunov', 'volatility', etc.
        """
        # 1. Determine Directive
        hurst = market_state.get('hurst', 0.5)
        lya = market_state.get('lyapunov', 0.0)
        vol = market_state.get('volatility', 0)
        
        self.current_directive = self.determine_directive("UNKNOWN", vol, hurst, lya)
        
        # 2. Allocate Temporary Authority
        active_weights = self.allocate_authority(self.current_directive)
        
        # 3. Calculate Coherence (Weighted)
        weighted_votes = []
        total_weight = 0
        
        for name, res in results_map.items():
            if name not in active_weights: continue
            
            w = active_weights[name]
            
            if isinstance(res, dict) and 'decision' in res:
                dec = res['decision']
                val = 0
                if "BUY" in dec: val = 1
                elif "SELL" in dec: val = -1
                
                weighted_votes.append(val * w)
                total_weight += w
                
        coherence = 0
        if total_weight > 0:
            # Coherence is the magnitude of the weighted mean vector
            net_vote = sum(weighted_votes)
            coherence = abs(net_vote / total_weight)
            
        # 4. Sandbox Veto
        # If Directive is SURVIVAL, we Veto everything unless Coherence is perfect
        veto = False
        if self.current_directive == "SURVIVAL" and coherence < 0.9:
            veto = True
            
        # 5. Health Score
        health_score = int(coherence * 100)
        
        return {
            'status': "OPERATIONAL",
            'directive': self.current_directive,
            'coherence': coherence,
            'health_score': health_score,
            'veto': veto,
            'eye_authorities': active_weights
        }
