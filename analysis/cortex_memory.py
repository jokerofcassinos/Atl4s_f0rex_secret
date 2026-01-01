import pandas as pd
import numpy as np
import logging
import json
import os
import config

logger = logging.getLogger("Atl4s-Cortex")

class CortexMemory:
    def __init__(self):
        self.memory_file = os.path.join(config.CACHE_DIR, "cortex_memory.json")
        self.memories = [] # List of {features: [], outcome: val}
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memories = json.load(f)
                logger.info(f"Cortex loaded {len(self.memories)} memories.")
            except Exception as e:
                logger.error(f"Failed to load Cortex Memory: {e}")
                self.memories = []

    def save_memory(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f)
        except Exception as e:
            logger.error(f"Failed to save Cortex Memory: {e}")

    def extract_features(self, df):
        """
        Extracts a Holographic Feature Vector (10-dim).
        Market State Snapshot.
        """
        if df is None or df.empty:
            return []
            
        last = df.iloc[-1]
        
        rsi = last.get('RSI', 50)
        atr = last.get('ATR', 1.0)
        
        f_rsi = rsi / 100.0
        f_vol = (atr / last['close']) * 1000
        
        roc = (last['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close'] if len(df) > 5 else 0
        f_roc = roc * 100
        
        hour = pd.Timestamp.now().hour / 24.0
        
        features = [f_rsi, f_vol, f_roc, hour]
        return features

    def recall(self, current_features, k=5):
        """
        Holographic Recall using Cosine Similarity.
        """
        if not self.memories or not current_features:
            return 0.5 
            
        curr_vec = np.array(current_features)
        curr_norm = np.linalg.norm(curr_vec)
        if curr_norm == 0: return 0.5
        
        similarities = []
        for mem in self.memories:
            mem_vec = np.array(mem['features'])
            mem_norm = np.linalg.norm(mem_vec)
            
            if len(mem_vec) != len(curr_vec) or mem_norm == 0:
                continue
            
            sim = np.dot(curr_vec, mem_vec) / (curr_norm * mem_norm)
            similarities.append((sim, mem['outcome']))
            
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        top_k = similarities[:k]
        if not top_k: return 0.5
        
        total_weight = sum([x[0] for x in top_k])
        if total_weight == 0: return 0.5
        
        weighted_outcome = sum([x[0] * x[1] for x in top_k]) / total_weight
        
        prob_bullish = (weighted_outcome + 1) / 2
        return prob_bullish

    def store_experience(self, features, outcome):
        """
        Stores a new memory.
        outcome: 1 (Win/Up), -1 (Loss/Down)
        """
        if features is None:
            return
            
        self.memories.append({
            'features': features,
            'outcome': outcome,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        if len(self.memories) > 10000:
            self.memories.pop(0)
            
        self.save_memory()
