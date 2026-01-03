
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from core.interfaces import SubconsciousUnit, SwarmSignal
import time
from collections import deque

logger = logging.getLogger("DNASwarm")

class DNASwarm(SubconsciousUnit):
    """
    Phase 68: The DNA Swarm (Biological Sequence Mining).
    
    Treats market data not as numbers, but as Genetic Code.
    Encodes price action into DNA Strands (ACGT).
    Scans the 'Evolutionary Record' (History) for matching genes.
    
    Encoding:
    - A (Adenine): Strong Impulse UP
    - C (Cytosine): Drift UP
    - G (Guanine): Drift DOWN
    - T (Thymine): Strong Impulse DOWN
    
    If 'Current Strand' matches 'Ancient Strand', we predict the 'Evolutionary Outcome'.
    """
    def __init__(self):
        super().__init__("DNA_Swarm")
        self.genome_vault = {} # Hash map of Sequence -> Outcome
        self.sequence_len = 8 # Length of gene to match
        self.learning_rate = 0.1 
        self.min_samples = 100
        
        # We need a rolling buffer to build the vault
        self.history_buffer = []

    def _encode_candle(self, open_p, close_p, atr) -> str:
        delta = close_p - open_p
        threshold = atr * 0.5
        
        if delta > threshold: return 'A' # Strong Up
        elif delta > 0: return 'C'       # Weak Up
        elif delta < -threshold: return 'T' # Strong Down
        else: return 'G'                 # Weak Down

    async def process(self, context) -> SwarmSignal:
        df_m1 = context.get('df_m1')
        if df_m1 is None or len(df_m1) < 200: return None
        
        # 1. Calculate ATR for dynamic thresholding
        # Simple approximation
        high = df_m1['high'].values
        low = df_m1['low'].values
        close = df_m1['close'].values
        open_p = df_m1['open'].values
        
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        atr = np.mean(tr[-14:])
        
        # 2. Sequence the Genome (Recent History)
        # We encode the last 500 candles to build our 'Vault' of known species
        # In production, this vault would be persistent. Here we rebuild/cache.
        
        full_sequence = []
        for i in range(len(df_m1)):
            code = self._encode_candle(open_p[i], close[i], atr)
            full_sequence.append(code)
            
        # 3. Extract Current Strand (The active gene)
        current_strand_list = full_sequence[-self.sequence_len:]
        current_strand = "".join(current_strand_list)
        
        # 4. Search for Homology (BLAST Search)
        # We look backwards in time for this exact sequence
        # We start from -sequence_len (current) and go back
        
        matches = []
        search_limit = min(2000, len(full_sequence) - self.sequence_len)
        
        for i in range(search_limit):
            # Index from end: -8 (current) -> need to check -9, -10...
            # seq_start = -(i + self.sequence_len * 2) 
            # This logic is complex. Let's use forward indexing on the historical list.
            
            # Look at segment [j : j+8]
            j = len(full_sequence) - self.sequence_len - 1 - i
            if j < 0: break
            
            historical_strand = "".join(full_sequence[j : j + self.sequence_len])
            
            if historical_strand == current_strand:
                # MATCH FOUND!
                # What happened NEXT? (The Mutation)
                # The next gene is at j + sequence_len
                next_gene_idx = j + self.sequence_len
                if next_gene_idx < len(full_sequence):
                    outcome = full_sequence[next_gene_idx]
                    matches.append(outcome)
                    
        # 5. Analyze Expression
        if not matches: return None
        
        count = len(matches)
        a_count = matches.count('A')
        c_count = matches.count('C')
        g_count = matches.count('G')
        t_count = matches.count('T')
        
        bullish = a_count + c_count
        bearish = t_count + g_count
        
        strong_bull = a_count
        strong_bear = t_count
        
        signal = "WAIT"
        confidence = 0.0
        reason = ""
        
        # Bias calculation
        total = count
        bull_ratio = bullish / total
        bear_ratio = bearish / total
        
        if total < 3: 
            return None # Insufficient genetic data
            
        # If 90% of ancestors with this gene went UP
        if bull_ratio > 0.8:
            signal = "BUY"
            confidence = 80.0 + (bull_ratio * 10)
            reason = f"DNA: Genetic Match ({current_strand}) found {total} times. 🧬 80%+ Evolved Bullish."
            
        elif bear_ratio > 0.8:
            signal = "SELL"
            confidence = 80.0 + (bear_ratio * 10)
            reason = f"DNA: Genetic Match ({current_strand}) found {total} times. 🧬 80%+ Evolved Bearish."
            
        if signal != "WAIT":
            return SwarmSignal(
                source=self.name,
                signal_type=signal,
                confidence=confidence,
                timestamp=time.time(),
                meta_data={'gene': current_strand, 'matches': total, 'reason': reason}
            )
            
        return None
