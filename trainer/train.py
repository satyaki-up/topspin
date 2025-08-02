import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import argparse
from typing import Optional, Tuple
import os

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x shape: (batch, heads, seq_len, head_dim)
    # freqs_cis shape: (seq_len, head_dim//2)
    
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    
    freqs_cis_expanded = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    freqs_cis_expanded = freqs_cis_expanded.expand(x.shape[0], x.shape[1], -1, -1)
    
    result_complex = x_complex * freqs_cis_expanded
    
    result_real = torch.view_as_real(result_complex)
    return result_real.reshape(*x.shape)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.hidden_size = config['hidden_size']
        
        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, config['max_seq_len'])
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rotary_emb(q, self.freqs_cis[:T])
        k = apply_rotary_emb(k, self.freqs_cis[:T])
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config['hidden_size'], config['ffn_size'], bias=False)
        self.w2 = nn.Linear(config['ffn_size'], config['hidden_size'], bias=False)
        self.w3 = nn.Linear(config['hidden_size'], config['ffn_size'], bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config['hidden_size'])
        self.ffn_norm = RMSNorm(config['hidden_size'])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class LLaMAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layers'])
        ])
        
        self.norm = RMSNorm(config['hidden_size'])
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.output.weight = self.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        
        if attention_mask is None:
            mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0)
        else:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask = mask.expand(-1, -1, T, -1)  # (B, 1, T, T)
            causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device))
            mask = mask * causal_mask.unsqueeze(0).unsqueeze(0)

        x = self.tok_embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        logits = self.output(x)
        return logits

def load_model_config(config_path: str = "configs/model.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_path: str) -> dict:
    data = torch.load(data_path)
    return data

def create_dataloader(data: dict, batch_size: int = 4, seq_len: int = 512):
    num_samples, max_seq_len = data['data'].shape
    
    def get_batch():
        sample_indices = torch.randint(0, num_samples, (batch_size,))
        start_positions = torch.randint(0, max_seq_len - seq_len - 1, (batch_size,))
        
        x = torch.stack([data['data'][i, start:start + seq_len] for i, start in zip(sample_indices, start_positions)])
        y = torch.stack([data['data'][i, start + 1:start + seq_len + 1] for i, start in zip(sample_indices, start_positions)])
        
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        return x, y, mask
    
    return get_batch

def train_model(model: nn.Module, dataloader, num_epochs: int = 10, lr: float = 1e-4):
    """Train the model."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx in range(100):  # 100 batches per epoch
            x, y, mask = dataloader()
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            logits = model(x, mask)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train LLaMA-style model")
    parser.add_argument("--data_path", default="data/minipile_tokenized.pt", help="Path to tokenized data")
    parser.add_argument("--config", default="configs/model.yaml", help="Path to model config")
    
    args = parser.parse_args()
    
    print("Loading model configuration...")
    config = load_model_config(args.config)
    
    print("Loading data...")
    data = load_data(args.data_path)
    print(f"Data loaded: {data['data'].shape}")
    
    config['model']['vocab_size'] = data['vocab_size']
    print(f"Model config: {config['model']}")
    
    print("Creating model...")
    model = LLaMAModel(config['model'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    dataloader = create_dataloader(data, config['training']['batch_size'], config['training']['seq_len'])
    
    print("Starting training...")
    lr = float(config['training']['learning_rate'])
    train_model(model, dataloader, config['training']['num_epochs'], lr)
    
    save_path = config['output']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config['model'],
        'vocab_size': data['vocab_size']
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main() 