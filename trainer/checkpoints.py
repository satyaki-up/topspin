import os
import time
import torch

def clean_up_old_checkpoints(checkpoint_dir: str = "models"):
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".pt"):
            try:
                checkpoint_num = int(filename.split("_")[1].split(".")[0])
                checkpoint_files.append((checkpoint_num, filename))
            except ValueError:
                continue
    
    checkpoint_files.sort()
    
    if len(checkpoint_files) > 2:
        files_to_delete = checkpoint_files[:-2]
        for _, filename in files_to_delete:
            filepath = os.path.join(checkpoint_dir, filename)
            os.remove(filepath)
            print(f"Deleted old checkpoint: {filename}")

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_idx: int, total_batches: int, 
                   loss: float, config: dict, data: dict, checkpoint_dir: str = "models"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_interval = total_batches // 10
    checkpoint_num = batch_idx // checkpoint_interval
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num:02d}.pt")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_idx': batch_idx,
        'total_batches': total_batches,
        'checkpoint_num': checkpoint_num,
        'loss': loss,
        'config': config,
        'data_vocab_size': data['vocab_size'] if data else None,
        'timestamp': torch.tensor(time.time())
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint {checkpoint_num} to {checkpoint_path}")
    
    clean_up_old_checkpoints(checkpoint_dir) 