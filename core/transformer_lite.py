
import numpy as np
import logging

# Try to use our JIT acceleration if available
try:
    from core.acceleration import njit
except ImportError:
    # Fallback mock
    def njit(f): return f

logger = logging.getLogger("TransformerLite")

@njit
def softmax(x):
    """Numerically stable softmax with NaN protection"""
    # Replace NaNs with -inf to ignore them (prob -> 0)
    x = np.nan_to_num(x, nan=-1e9)
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-9) # Epsilon div by zero

@njit
def scaled_dot_product_attention(Q, K, V, dk):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) * V
    """
    # Q: (n_seq, d_k)
    # K: (n_seq, d_k)
    # V: (n_seq, d_v)
    
    matmul_qk = np.dot(Q, K.T) # (n_seq, n_seq)
    
    # Scale
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    # Softmax
    attention_weights = softmax(scaled_attention_logits)
    
    # Output
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

class TransformerLite:
    """
    Simplified Single-Head Attention Block for Swarm Consensus.
    """
    def __init__(self, embed_dim, head_dim):
        self.d_model = embed_dim
        self.d_k = head_dim
        
        # Weights (Random Init)
        self.W_q = np.random.rand(self.d_model, self.d_k) - 0.5
        self.W_k = np.random.rand(self.d_model, self.d_k) - 0.5
        self.W_v = np.random.rand(self.d_model, self.d_k) - 0.5
        self.W_o = np.random.rand(self.d_k, self.d_model) - 0.5
        
    def forward(self, x):
        """
        x: Input tensor (n_agents, embed_dim)
        """
        # Linear Projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Attention
        attn_out, weights = scaled_dot_product_attention(Q, K, V, self.d_k)
        
        # Output Linear
        output = np.dot(attn_out, self.W_o)
        
        # Residual Connection + Norm (Simplified)
        output = output + x
        output = output / (np.linalg.norm(output, axis=1, keepdims=True) + 1e-6)
        
        return output, weights
