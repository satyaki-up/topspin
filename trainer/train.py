import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import argparse
from typing import Optional, Tuple
import os
import pandas as pd
import json
import glob

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

def load_sharded_data(data_dir: str) -> dict:
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    shard_pattern = os.path.join(data_dir, 'shard_*.parquet')
    shard_files = sorted(glob.glob(shard_pattern))
    
    print(f"Found {len(shard_files)} shard files")
    
    all_tokens = []
    all_sequence_lengths = []
    
    for i, shard_file in enumerate(shard_files):
        if i % 10 == 0 or i == len(shard_files) - 1:
            print(f"Loading shard {i+1}/{len(shard_files)}...")
        
        df = pd.read_parquet(shard_file)
        all_tokens.extend(df['tokens'].tolist())
        all_sequence_lengths.extend(df['sequence_length'].tolist())
    
    max_len = max(all_sequence_lengths)
    num_samples = len(all_tokens)
    
    print(f"Creating padded tensor: {num_samples} x {max_len}")
    padded_data = torch.zeros(num_samples, max_len, dtype=torch.long)
    attention_mask = torch.zeros(num_samples, max_len, dtype=torch.bool)
    
    print(f"Padded data tensor size: {padded_data.shape}, memory: {padded_data.element_size() * padded_data.nelement() / 1024**2:.2f} MB")
    print(f"Attention mask tensor size: {attention_mask.shape}, memory: {attention_mask.element_size() * attention_mask.nelement() / 1024**2:.2f} MB")
    
    for i, tokens in enumerate(all_tokens):
        padded_data[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :len(tokens)] = True
    
    return {
        'data': padded_data,
        'attention_mask': attention_mask,
        'sequence_lengths': all_sequence_lengths,
        'vocab_size': metadata['vocab_size'],
        'num_samples': num_samples
    }

def load_data(data_path: str) -> dict:
    if data_path.endswith('.pt'):
        data = torch.load(data_path)
        return data
    else:
        return load_sharded_data(data_path)

def create_dataloader(data: dict, batch_size: int = 4, seq_len: int = 512):
    num_samples, max_seq_len = data['data'].shape
    print(f"Creating dataloader: {num_samples} samples, max_seq_len: {max_seq_len}, batch_size: {batch_size}, seq_len: {seq_len}")
    
    def get_batch():
        sample_indices = torch.randint(0, num_samples, (batch_size,))
        start_positions = torch.randint(0, max_seq_len - seq_len - 1, (batch_size,))
        
        x = torch.stack([data['data'][i, start:start + seq_len] for i, start in zip(sample_indices, start_positions)])
        y = torch.stack([data['data'][i, start + 1:start + seq_len + 1] for i, start in zip(sample_indices, start_positions)])
        
        # Convert tokens to long dtype
        x = x.long()
        y = y.long()
        
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        print(f"Batch tensors - x: {x.shape}, y: {y.shape}, mask: {mask.shape}")
        print(f"Batch memory - x: {x.element_size() * x.nelement() / 1024**2:.2f} MB, y: {y.element_size() * y.nelement() / 1024**2:.2f} MB")
        
        return x, y, mask
    
    return get_batch

def train_model(model: nn.Module, dataloader, num_epochs: int = 10, lr: float = 1e-4):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model memory: {sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2:.2f} MB")
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx in range(100):  # 100 batches per epoch
            x, y, mask = dataloader()
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            if batch_idx == 0:  # Log first batch of each epoch
                print(f"Epoch {epoch+1} - Device tensors - x: {x.shape}, y: {y.shape}, mask: {mask.shape}")
                print(f"Device memory - x: {x.element_size() * x.nelement() / 1024**2:.2f} MB, y: {y.element_size() * y.nelement() / 1024**2:.2f} MB")
            
            optimizer.zero_grad()
            
            logits = model(x, mask)
            print(f"Logits shape: {logits.shape}, memory: {logits.element_size() * logits.nelement() / 1024**2:.2f} MB")
            
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            print(f"Reshaped - logits: {logits.shape}, y: {y.shape}")
            
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
    parser.add_argument("--data_path", default="data", help="Path to data directory (for sharded) or .pt file (for legacy)")
    parser.add_argument("--config", default="configs/model.yaml", help="Path to model config")
    
    args = parser.parse_args()
    
    print("Loading model configuration...")
    config = load_model_config(args.config)
    
    print("Loading data...")
    data = load_data(args.data_path)
    print(f"Data loaded: {data['data'].shape}")
    
    # Calculate and print number of tokens
    total_tokens = data['data'].numel()
    print(f"Total tokens: {total_tokens:,}")
    
    config['model']['vocab_size'] = data['vocab_size']
    print(f"Model config: {config['model']}")
    
    print("Creating model...")
    model = LLaMAModel(config['model'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model moved to device, total model memory: {sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2:.2f} MB")
    
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