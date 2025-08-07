import torch
import torch.nn as nn
import yaml
import argparse
import os
import wandb
import time

from llama_model import LLaMAModel
from dataprep import load_sharded_data, create_dataloader
from checkpoints import save_checkpoint, clean_up_old_checkpoints

def load_model_config(config_path: str = "configs/model_50m.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(model: nn.Module, dataloader, batch_size: int = 10, lr: float = 1e-4, config: dict = None, data: dict = None):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model memory: {sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2:.2f} MB")
    
    total_samples = data['num_samples'] if data else 0
    total_batches = total_samples // batch_size
    
    wandb_config = {
        "model": config['model'] if config else {},
        "training": {
            "learning_rate": lr,
            "batch_size": batch_size,
            "seq_len": 1024,
            "total_batches": total_batches,
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
    
    total_loss = 0
    batch_losses = []
    
    print(f"Training for {total_batches} batches ({total_samples} samples with batch_size {batch_size})")
    
    for batch_idx in range(total_batches):
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
        batch_losses.append(loss_value)
        
        checkpoint_interval = total_batches // 10
        if checkpoint_interval > 0 and (batch_idx + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, batch_idx + 1, total_batches, loss_value, config, data)
        
        log_interval = total_batches // 100
        if log_interval > 0 and batch_idx % log_interval == 0:
            print(f"Batch {batch_idx}/{total_batches}, Loss: {loss_value:.4f}")
            wandb.log({
                "batch_loss": loss_value,
                "batch": batch_idx,
                "progress": batch_idx / total_batches,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    avg_loss = total_loss / len(batch_losses)
    print(f"Training completed. Average loss: {avg_loss:.4f}")
    
    checkpoint_interval = total_batches // 10
    final_checkpoint_num = 10
    final_checkpoint_path = os.path.join("models", f"checkpoint_{final_checkpoint_num:02d}.pt")
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_idx': total_batches,
        'total_batches': total_batches,
        'checkpoint_num': final_checkpoint_num,
        'loss': avg_loss,
        'config': config,
        'data_vocab_size': data['vocab_size'] if data else None,
        'timestamp': torch.tensor(time.time())
    }
    
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Saved final checkpoint {final_checkpoint_num} to {final_checkpoint_path}")
    
    clean_up_old_checkpoints("models")
        
    wandb.log({
        "final_loss": avg_loss,
        "loss_std": torch.tensor(batch_losses).std().item(),
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train LLaMA-style model")
    parser.add_argument("--data_path", default="data", help="Path to data directory with sharded parquet files")
    parser.add_argument("--config", default="configs/model_50m.yaml", help="Path to model config")
    
    args = parser.parse_args()
    
    print("Loading model configuration...")
    config = load_model_config(args.config)
    
    print("Loading data...")
    data = load_sharded_data(args.data_path)
    
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
    batch_size = config['training']['batch_size']
    train_model(model, dataloader, batch_size, lr, config, data)

if __name__ == "__main__":
    main() 