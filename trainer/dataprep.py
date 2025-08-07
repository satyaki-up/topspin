import os
import json
import pandas as pd
import glob
import torch

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

def create_streaming_dataloader(data: dict, batch_size: int = 4, seq_len: int = 1024):
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