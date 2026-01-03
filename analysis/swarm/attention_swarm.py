
import logging
import numpy as np
from typing import List, Dict
from core.interfaces import SubconsciousUnit, SwarmSignal
from core.transformer_lite import TransformerBlock

logger = logging.getLogger("AttentionSwarm")

class AttentionSwarm(SubconsciousUnit):
    """
    The Synaptic Web.
    Uses Self-Attention to synthesize disparate Swarm Signals into a unified context.
    """
    def __init__(self, num_agents=20):
        super().__init__("Attention_Swarm")
        self.embed_dim = 16 # Small embedding
        self.head_dim = 16
        self.transformer = TransformerBlock(self.embed_dim, self.head_dim)
        
        # Embedding Lookups (Simplified)
        # 0: WAIT, 1: BUY, 2: SELL
        # + Normalized Confidence
        
    def _embed_signals(self, signals: List[SwarmSignal]) -> np.ndarray:
        """
        Converts List of Signals into (N, embed_dim) matrix.
        """
        batch = []
        for sig in signals:
            # Create a vector representation
            # [Type(-1,0,1), Conf(0-1), 0, 0... noise]
            vec = np.random.rand(self.embed_dim) * 0.01 # Init with noise
            
            val_type = 0
            if sig.signal_type == 'BUY': val_type = 1
            elif sig.signal_type == 'SELL': val_type = -1
            
            vec[0] = val_type
            vec[1] = sig.confidence / 100.0
            
            # Agent ID Hashing (Simple)
            h = hash(sig.source) % 10
            vec[2] = h / 10.0
            
            batch.append(vec)
            
        return np.array(batch)

    def synthesize(self, signals: List[SwarmSignal]):
        """
        Run the Attention Mechanism.
        """
        if not signals: return None
        
        # 1. Embed
        x = self._embed_signals(signals)
        
        # Calculate Raw Average Confidence (Pre-Transformer)
        # We trust the raw confidence of agents more than the transformer's vector magnitude for this dimension
        raw_confidences = [sig.confidence for sig in signals]
        avg_raw_conf = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0.0
        
        # 2. Transformer Pass
        # context_matrix: (N, embed_dim) - Each agent's signal updated by context of others
        # weights: (N, N) - Who is paying attention to whom
        context_matrix, weights = self.transformer.forward(x)
        
        # 3. Decode Consensus
        # We average the context-aware vectors to get the "Swarm State"
        mean_vector = np.mean(context_matrix, axis=0)
        
        # Decode Output
        # Index 0 is direction
        direction_score = mean_vector[0]
        
        # REMOVED: Agreement Penalty. 
        # Reason: In a committee of 40 agents, 100% agreement is impossible.
        # A score of 0.15 is significant enough to act if the Raw Confidence is high.
        
        # New Logic: Trust the Raw Confidence of the Agents.
        final_conf = avg_raw_conf
        
        # Minor Boost for strong consensus, but NO PENALTY for weak consensus.
        if abs(direction_score) > 0.3:
            final_conf *= 1.2
        
        # Cap at 100
        final_conf = min(100.0, final_conf)
        
        logger.info(f"Attention Consensus: Score {direction_score:.3f} | Conf {final_conf:.1f} (Raw: {avg_raw_conf:.1f})")
        
        # Lowered Threshold to 0.1 to be very sensitive to "Trend"
        if abs(direction_score) > 0.1: 
            final_sig = "BUY" if direction_score > 0 else "SELL"
            # Return a synthetic signal
            return SwarmSignal(
                source="Attention_Core",
                signal_type=final_sig,
                confidence=final_conf,
                timestamp=0,
                meta_data={'attention_weights': weights.tolist()}
            )
            
        return None

    async def process(self, context) -> SwarmSignal:
        # AttentionSwarm is a Meta-Swarm, usually called explicitly by Orchestrator
        # But if called normally, it returns None
        return None
