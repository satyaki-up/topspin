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
import wandb

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
    
    print(f"Found {len(shard_files)} shard files for streaming")
    print(f"Total samples: {metadata['total_samples']:,}")
    print(f"Vocab size: {metadata['vocab_size']}")
    print(f"Max sequence length: {metadata['max_sequence_length']}")
    
    return {
        'shard_files': shard_files,
        'metadata': metadata,
        'vocab_size': metadata['vocab_size'],
        'num_samples': metadata['total_samples']
    }

def load_data(data_path: str) -> dict:
    return load_sharded_data(data_path)

def create_streaming_dataloader(data: dict, batch_size: int = 4, seq_len: int = 512):
    shard_files = data['shard_files']
    metadata = data['metadata']
    total_samples = metadata['total_samples']
    
    print(f"Creating streaming dataloader: {total_samples} samples, batch_size: {batch_size}, seq_len: {seq_len}")
    print(f"Will load shards on-demand during training")
    
    print("Loading shard metadata...")
    shard_data = []
    for i, shard_file in enumerate(shard_files):
        if i % 10 == 0 or i == len(shard_files) - 1:
            print(f"Loading shard metadata {i+1}/{len(shard_files)}...")
        df = pd.read_parquet(shard_file)
        shard_data.append(df)
    
    print(f"All shards loaded in memory (as DataFrames)")
    
    def get_batch():
        batch_tokens = []
        batch_lengths = []
        
        for _ in range(batch_size):
            shard_idx = torch.randint(0, len(shard_data), (1,)).item()
            shard_df = shard_data[shard_idx]
            
            sample_idx = torch.randint(0, len(shard_df), (1,)).item()
            tokens = shard_df.iloc[sample_idx]['tokens']
            seq_len_actual = shard_df.iloc[sample_idx]['sequence_length']
            
            batch_tokens.append(tokens)
            batch_lengths.append(seq_len_actual)
        
        x = torch.zeros(batch_size, seq_len, dtype=torch.long)
        y = torch.zeros(batch_size, seq_len, dtype=torch.long)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for i, (tokens, length) in enumerate(zip(batch_tokens, batch_lengths)):
            if length > seq_len:
                start_pos = torch.randint(0, length - seq_len, (1,)).item()
                end_pos = start_pos + seq_len
                x[i] = torch.tensor(tokens[start_pos:end_pos], dtype=torch.long)
                y[i] = torch.tensor(tokens[start_pos + 1:end_pos + 1], dtype=torch.long)
                mask[i] = True
            else:
                x[i, :length-1] = torch.tensor(tokens[:-1], dtype=torch.long)
                y[i, :length-1] = torch.tensor(tokens[1:], dtype=torch.long)
                mask[i, :length-1] = True
        
        return x, y, mask
    
    return get_batch

def create_dataloader(data: dict, batch_size: int = 4, seq_len: int = 512):
    return create_streaming_dataloader(data, batch_size, seq_len)

def train_model(model: nn.Module, dataloader, num_epochs: int = 10, lr: float = 1e-4, config: dict = None, data: dict = None):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model memory: {sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2:.2f} MB")
    
    wandb_config = {
        "model": config['model'] if config else {},
        "training": {
            "learning_rate": lr,
            "num_epochs": num_epochs,
            "batch_size": 4,
            "seq_len": 512,
        },
        "device": str(device),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    if data and 'metadata' in data:
        wandb_config["data"] = {
            "num_samples": data['num_samples'],
            "total_tokens": data['metadata']['total_tokens'],
            "vocab_size": data['vocab_size'],
            "max_sequence_length": data['metadata']['max_sequence_length'],
            "num_shards": data['metadata']['num_shards']
        }
    
    wandb.init(
        project="topspin",
        config=wandb_config
    )
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        epoch_losses = []
        
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
            
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            epoch_losses.append(loss_value)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss_value:.4f}")
                wandb.log({
                    "batch_loss": loss_value,
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": avg_loss,
            "epoch_loss_std": torch.tensor(epoch_losses).std().item(),
            "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train LLaMA-style model")
    parser.add_argument("--data_path", default="data", help="Path to data directory with sharded parquet files")
    parser.add_argument("--config", default="configs/model.yaml", help="Path to model config")
    
    args = parser.parse_args()
    
    print("Loading model configuration...")
    config = load_model_config(args.config)
    
    print("Loading data...")
    data = load_data(args.data_path)
    
    print(f"Streaming data loaded: {data['num_samples']:,} samples")
    print(f"Total tokens: {data['metadata']['total_tokens']:,}")
    
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
    train_model(model, dataloader, config['training']['num_epochs'], lr, config, data)
    
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