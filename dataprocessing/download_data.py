import os
import json
import torch
import tiktoken
import yaml
from datasets import load_dataset
from typing import List, Dict, Any
import argparse
import pandas as pd
import psutil

def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)

def load_config(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_memory_usage() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def tokenize_text(text: str, tokenizer: tiktoken.Encoding) -> List[int]:
    return tokenizer.encode(text, disallowed_special=())

def process_dataset_in_shards(config: Dict[str, Any]) -> None:
    dataset_name = config['dataset']['name']
    split = config['dataset']['split']
    max_samples = config['dataset']['max_samples']
    shard_size = config.get('shard_size', 10000)
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    total_samples = len(dataset)
    if max_samples:
        total_samples = min(max_samples, total_samples)
    
    print(f"Dataset has {len(dataset)} samples, processing {total_samples}")
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    tokenizer_name = config['tokenizer']['name']
    tokenizer = get_tokenizer(tokenizer_name)
    print(f"Using tokenizer: {tokenizer.name}")
    
    output_dir = config['output']['directory']
    os.makedirs(output_dir, exist_ok=True)
    
    num_shards = (total_samples + shard_size - 1) // shard_size
    print(f"Processing in {num_shards} shards of {shard_size} samples each")
    
    all_sequence_lengths = []
    total_tokens = 0
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, total_samples)
        
        if shard_idx % 10 == 0 or shard_idx == num_shards - 1:
            print(f"Processing shard {shard_idx + 1}/{num_shards}...")
        
        shard_dataset = dataset.select(range(start_idx, end_idx))
        
        shard_tokens = []
        for i, example in enumerate(shard_dataset):
            text = example.get('text', example.get('content', str(example)))
            if isinstance(text, list):
                text = ' '.join(text)
            
            tokens = tokenize_text(text, tokenizer)
            shard_tokens.append(tokens)
            all_sequence_lengths.append(len(tokens))
            total_tokens += len(tokens)
        
        shard_filename = f"shard_{shard_idx:04d}.parquet"
        shard_path = os.path.join(output_dir, shard_filename)
        
        shard_data = {
            'tokens': shard_tokens,
            'sequence_length': [len(tokens) for tokens in shard_tokens]
        }
        df = pd.DataFrame(shard_data)
        df.to_parquet(shard_path, index=False)
        
        del shard_dataset, shard_tokens, shard_data, df
    
    metadata = {
        'num_shards': num_shards,
        'total_samples': total_samples,
        'shard_size': shard_size,
        'vocab_size': config['tokenizer']['vocab_size'],
        'all_sequence_lengths': all_sequence_lengths,
        'total_tokens': total_tokens,
        'max_sequence_length': max(all_sequence_lengths) if all_sequence_lengths else 0
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processing complete!")
    print(f"Total samples: {total_samples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Max sequence length: {metadata['max_sequence_length']}")
    print(f"Shards saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

def save_tokenized_data(tokenized_data: List[List[int]], 
                       config: Dict[str, Any]):
    output_dir = config['output']['directory']
    filename = config['output']['filename']
    vocab_size = config['tokenizer']['vocab_size']
    
    os.makedirs(output_dir, exist_ok=True)
    
    max_len = max(len(seq) for seq in tokenized_data)
    print(f"Maximum sequence length: {max_len}")
    
    padded_data = torch.zeros(len(tokenized_data), max_len, dtype=torch.int32)
    attention_mask = torch.zeros(len(tokenized_data), max_len, dtype=torch.bool)
    
    for i, tokens in enumerate(tokenized_data):
        padded_data[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        attention_mask[i, :len(tokens)] = True
    
    output_path = os.path.join(output_dir, filename)
    torch.save({
        'data': padded_data,
        'attention_mask': attention_mask,
        'sequence_lengths': [len(seq) for seq in tokenized_data],
        'vocab_size': vocab_size,
        'num_samples': len(tokenized_data)
    }, output_path)
    
    print(f"Saved tokenized data to: {output_path}")
    print(f"Data shape: {padded_data.shape}")
    print(f"Memory usage: {padded_data.element_size() * padded_data.nelement() / 1024**2:.2f} MB")

def load_tokenized_data(data_path: str) -> Dict[str, Any]:
    data = torch.load(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"Data shape: {data['data'].shape}")
    print(f"Number of samples: {data['num_samples']}")
    return data

def main():
    parser = argparse.ArgumentParser(description="Download and tokenize minipile dataset")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config file")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max_samples from config")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of samples per shard")
    parser.add_argument("--load_only", action="store_true", help="Only load existing data")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.max_samples is not None:
        config['dataset']['max_samples'] = args.max_samples
    
    if args.shard_size is not None:
        config['shard_size'] = args.shard_size
    
    if args.load_only:
        data_path = os.path.join(config['output']['directory'], config['output']['filename'])
        if os.path.exists(data_path):
            load_tokenized_data(data_path)
        else:
            print(f"Data file not found: {data_path}")
    else:
        process_dataset_in_shards(config)

if __name__ == "__main__":
    main()
