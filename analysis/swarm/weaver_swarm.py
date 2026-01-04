
import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("WeaverSwarm")

class WeaverSwarm(SubconsciousUnit):
    """
    Phase 97: The Weaver Swarm (Meta-Causal Entanglement).
    
    analyzes the 'Social Network' of the Swarms.
    Constructs a dynamic graph where:
    - Nodes: Active Swarm Agents.
    - Edges: Correlation of their recent signals.
    
    Goal:
    - Detect 'Singularity': High Coherence (All agents agree). strong signal.
    - Detect 'Civil War': High Entropy (Agents split 50/50). WAIT signal.
    - Detect 'Schism': Physics Agents vs Statistical Agents.
    """
    def __init__(self, bus):
        super().__init__("Weaver_Swarm")
        self.bus = bus # Access to the Quantum Bus to read other thoughts
        self.history = {} # {agent_name: [last_5_signals]}
        
    async def process(self, context: Dict[str, Any]) -> Optional[SwarmSignal]:
        # 1. Gather Intelligence
        # We look at recent thoughts. The Bus flushes them on call.
        thoughts = self.bus.get_recent_thoughts()
        
        if not thoughts: return None
        
        # 2. Build Alignment Matrix
        # Map: Agent -> Current Stance (1=Buy, -1=Sell, 0=Wait)
        stance_map = {}
        
        # We only care about the LATEST thought from each agent in the window
        # Thoughts are usually chronological, so usually the last one for each agent
        for t in thoughts:
            val = 0
            if t.signal_type == "BUY": val = 1
            elif t.signal_type == "SELL": val = -1
            
            # Confidence weighting?
            # val *= (t.confidence / 100.0)
            
            stance_map[t.source] = val
            
        if len(stance_map) < 3: return None # Need a crowd to have a consensus
        
        # 3. Calculate Network Entropy (Coherence)
        values = list(stance_map.values())
        mean_stance = np.mean(values)
        
        # Coherence = abs(mean). 1.0 = Perfect Unanimity. 0.0 = Civil War.
        coherence = abs(mean_stance)
        
        # 4. Graph Theory Analysis (Simplified)
        # Count formatting cliques
        buyers = [k for k,v in stance_map.items() if v > 0]
        sellers = [k for k,v in stance_map.items() if v < 0]
        
        n_buy = len(buyers)
        n_sell = len(sellers)
        total = n_buy + n_sell
        
        if total == 0: return None
        
        # ENTROPY CALCULATION
        # Shannon Entropy of the Buy/Sell distribution
        # p_buy = n_buy / total
        # p_sell = n_sell / total
        # H = - (p_buy * log2(p_buy) + p_sell * log2(p_sell))
        # Range 0 (Unanimous) to 1 (50/50 split)
        
        decision = "WAIT"
        conf = 0.0
        meta = {}
        
        # If Coherence is High (>0.7), we join the winner.
        if coherence > 0.7:
            if n_buy > n_sell:
                decision = "BUY"
                conf = 85.0 + (coherence * 10) # Boost confidence based on unity
                meta['reason'] = f"Swarm Singularity (Coherence {coherence:.2f})"
            else:
                decision = "SELL"
                conf = 85.0 + (coherence * 10)
                meta['reason'] = f"Swarm Singularity (Coherence {coherence:.2f})"
                
        # If Coherence is Low (<0.3), we detect Civil War.
        elif coherence < 0.3:
            # We explicitly signal WAIT to dampen the noise
            # or we signal "VETO" if we have authority?
            # Let's just signal WAIT but with High Confidence to drag the average down?
            # Weaver doesn't override, it votes.
            decision = "WAIT" 
            conf = 0.0 # Wait signals usually have 0 conf, but maybe we want to log the event.
            meta['reason'] = f"Civil War Detected (Buyers: {n_buy}, Sellers: {n_sell})"
            
            # Special Case: If Physics Agents (Kinematic) are all on one side, but ignored?
            # Creating a 'Meta-Veto' might be useful later.
            return SwarmSignal(self.name, "VETO", 90.0, time.time(), meta)

        if decision == "WAIT": return None
        
        return SwarmSignal(
            source=self.name,
            signal_type=decision,
            confidence=min(conf, 99.9),
            timestamp=time.time(),
            meta_data=meta
        )
