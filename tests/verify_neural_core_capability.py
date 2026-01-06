
import torch
import torch.nn as nn
import math

class ReasoningTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_layers=8, n_heads=8, d_ff=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.pos_encoder, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, 1) # Truth Score

    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer_encoder(x)
        return self.output_head(x.mean(dim=1))

def verify_parameters():
    print("--- VERIFYING NEURAL CORE CAPACITY ---")
    try:
        model = ReasoningTransformer()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Architecture: 8 Layers, 512 Hidden, 32k Vocab")
        print(f"Total Parameters: {total_params:,}")
        
        if total_params > 30_000_000:
             print("SUCCESS: Model exceeds 30 Million Parameters requirement.")
        else:
             print("FAILURE: Model is too small.")
             
    except Exception as e:
        print(f"Error initializing model: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA Available: Yes")
    else:
        print("CUDA Available: No (Running on CPU)")
    
    verify_parameters()
