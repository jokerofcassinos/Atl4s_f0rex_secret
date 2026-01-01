import logging
import numpy as np
import pandas as pd
import os
import json
import config

logger = logging.getLogger("Atl4s-Architect")

class TenthEye:
    """
    The Architect (Meta-System).
    Oversees the health and weighting of all other Analytic Eyes.
    - Eye Performance Tracking (Reliability)
    - Dynamic Authority Adjustment
    - System Coherence Veto
    """
    def __init__(self):
        self.stats_file = os.path.join(config.CACHE_DIR, "eye_reliability.json")
        self.eye_weights = {f"Eye{i}": 1.0 for i in range(1, 10)}
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

    def record_outcome(self, eye_decisions, actual_outcome):
        """
        Updates reliability weights based on whether an Eye was right about the direction.
        Called after a trade is closed or at fixed intervals.
        """
        for eye, decision in eye_decisions.items():
            if eye not in self.eye_weights: continue
            
            # Simple reinforcement learning
            if decision == actual_outcome:
                self.eye_weights[eye] = min(2.0, self.eye_weights[eye] + 0.05)
            else:
                self.eye_weights[eye] = max(0.1, self.eye_weights[eye] - 0.10)
        
        self.save_stats()

    def calculate_coherence(self, results_map):
        """
        Analyzes the internal conflict of the system.
        If too many eyes disagree, the Architect triggers a Global Veto.
        """
        direct_votes = []
        for name, res in results_map.items():
            if isinstance(res, dict) and 'decision' in res:
                dec = res['decision']
                if "BUY" in dec: direct_votes.append(1)
                elif "SELL" in dec: direct_votes.append(-1)
                
        if not direct_votes: return 1.0
        
        # Coherence = Mean of absolute votes (1.0 means unanimous, 0.0 means perfect conflict)
        coherence = abs(np.mean(direct_votes))
        return coherence

    def calculate_health_score(self, coherence, status):
        """
        Converts internal coherence and status into a user-facing health score (0-100).
        """
        base_score = coherence * 100
        
        if status == "SYSTEM_CONFLICT":
            base_score = max(0, base_score - 20)
        elif status == "HIGH_COHERENCE":
             base_score = min(100, base_score + 10)
             
        # Penalty for low eye weights (degraded reliability)
        avg_weight = np.mean(list(self.eye_weights.values()))
        if avg_weight < 0.8:
            base_score -= 10
            
        return int(max(0, min(100, base_score)))

    def deliberate(self, results_map):
        """
        Final System Audit.
        """
        coherence = self.calculate_coherence(results_map)
        
        # Meta-Directives
        status = "OPERATIONAL"
        veto = False
        
        if coherence < 0.3: # Major Conflict
            status = "SYSTEM_CONFLICT"
            veto = True
        elif coherence > 0.8:
            status = "HIGH_COHERENCE"
            
        health_score = self.calculate_health_score(coherence, status)
            
        return {
            'status': status,
            'coherence': coherence,
            'health_score': health_score,
            'veto': veto,
            'eye_authorities': self.eye_weights
        }
