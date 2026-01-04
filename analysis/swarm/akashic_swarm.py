
import logging
import numpy as np
import time
from typing import Dict, Any, List, Tuple
import json
import os
from core.interfaces import SubconsciousUnit, SwarmSignal

logger = logging.getLogger("AkashicSwarm")

class AkashicRecords:
    """
    The Hall of Records.
    Stores market states (DNA) and their outcomes.
    Allows the Swarm to 'recall' past lives (Market Regimes).
    """
    def __init__(self, db_path="akashic_records.json"):
        self.db_path = db_path
        self.memory = [] # List of vectors
        self.limit = 10000
        self.load()
        
    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.memory = json.load(f)
                logger.info(f"Akashic Records loaded: {len(self.memory)} memories.")
            except:
                self.memory = []
        
    def save(self):
        # Async save ideally, but simple for now
        with open(self.db_path, 'w') as f:
            json.dump(self.memory[-self.limit:], f)
            
    def store_memory(self, state_vector: List[float], outcome: float, context: str):
        """
        Record a moment in time.
        state_vector: [RSI, Volatility, Spread, Flux, etc]
        outcome: Profit/Loss delayed.
        """
        entry = {
            'vector': state_vector,
            'outcome': outcome,
            'context': context,
            'weight': 1.0
        }
        self.memory.append(entry)
        if len(self.memory) % 100 == 0:
            self.save()

    def query(self, current_vector: List[float], k=5) -> Tuple[str, float]:
        """
        Finds the 5 most similar past situations.
        Returns weighted consensus of their outcomes.
        """
        if not self.memory: return ("WAIT", 0.0)
        
        # Simple Euclidean Distance (or Cosine Similarity)
        # We assume vectors are normalized or roughly same scale.
        best_matches = []
        target = np.array(current_vector)
        
        for mem in self.memory:
            past_vec = np.array(mem['vector'])
            # Check dimensions match
            if len(past_vec) != len(target): continue
            
            dist = np.linalg.norm(target - past_vec)
            best_matches.append((dist, mem))
            
        # Sort by distance (asc)
        best_matches.sort(key=lambda x: x[0])
        nearest = best_matches[:k]
        
        if not nearest: return ("WAIT", 0.0)
        
        # Consensus
        total_outcome = 0.0
        total_weight = 0.0
        
        for dist, mem in nearest:
            # Weight = 1 / (1 + dist)
            w = 1.0 / (1.0 + dist)
            total_outcome += mem['outcome'] * w
            total_weight += w
            
        if total_weight == 0: return ("WAIT", 0.0)
        
        avg_outcome = total_outcome / total_weight
        
        # logger.info(f"Akashic Recall: {len(nearest)} echoes found. Avg Outcome: {avg_outcome:.2f}")
        
        if avg_outcome > 0.5: return ("BUY", min(85.0 + avg_outcome*10, 99.0))
        if avg_outcome < -0.5: return ("SELL", min(85.0 + abs(avg_outcome)*10, 99.0))
        
        return ("WAIT", 0.0)

class AkashicSwarm(SubconsciousUnit):
    """
    The Keeper of Memory.
    Projects current market state into the Akashic Records to foresee the inevitable.
    Phase 96: Ultra Complex Reasoning via Historical Isomorphism.
    """
    def __init__(self):
        super().__init__("Akashic_Swarm")
        self.records = AkashicRecords()
        
    async def process(self, context: Dict[str, Any]) -> SwarmSignal:
        tick = context.get('tick')
        market_state = context.get('market_state', {})
        if not tick: return None

        # 1. Hashing the Reality (Creating Vector)
        # We need a consistent vector representation of the market.
        # [RSI, Volatility, Spread, Entropy]
        
        rsi = market_state.get('rsi', 50)
        volat = market_state.get('volatility', 1.0)
        spread = tick.get('ask',0) - tick.get('bid',0)
        entropy = market_state.get('entropy', 0.5)
        
        vector = [
            (rsi - 50) / 50.0, # -1 to 1
            min(volat / 100.0, 1.0),
            min(spread * 10, 1.0),
            entropy
        ]
        
        # 2. Consult the Records
        decision, confidence = self.records.query(vector)
        
        return SwarmSignal(
            source=self.name,
            signal_type=decision,
            confidence=confidence,
            timestamp=time.time(), # Added Timestamp
            meta_data={'vector_hash': str(vector[:3])}
        )
