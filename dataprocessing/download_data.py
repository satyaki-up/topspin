import os
import json
import torch
import tiktoken
import yaml
from datasets import load_dataset
from typing import List, Dict, Any
import argparse

def get_tokenizer(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken tokenizer."""
    return tiktoken.get_encoding(encoding_name)

def load_config(config_path: str = "configs/data.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def tokenize_text(text: str, tokenizer: tiktoken.Encoding) -> List[int]:
    """Tokenize a single text string."""
    # Handle special tokens by allowing them to be encoded as normal text
    return tokenizer.encode(text, disallowed_special=())

def process_dataset(config: Dict[str, Any]) -> List[List[int]]:
    """Download and tokenize the dataset."""
    dataset_name = config['dataset']['name']
    split = config['dataset']['split']
    max_samples = config['dataset']['max_samples']
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Get tokenizer
    tokenizer_name = config['tokenizer']['name']
    tokenizer = get_tokenizer(tokenizer_name)
    print(f"Using tokenizer: {tokenizer.name}")
    
    # Tokenize all texts
    tokenized_data = []
    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(dataset)}")
        
        # Get the text content (adjust field name if needed)
        text = example.get('text', example.get('content', str(example)))
        if isinstance(text, list):
            text = ' '.join(text)
        
        # Tokenize
        tokens = tokenize_text(text, tokenizer)
        tokenized_data.append(tokens)
    
    print(f"Tokenization complete. Total samples: {len(tokenized_data)}")
    return tokenized_data

def save_tokenized_data(tokenized_data: List[List[int]], 
                       config: Dict[str, Any]):
    """Save tokenized data in PyTorch format."""
    output_dir = config['output']['directory']
    filename = config['output']['filename']
    vocab_size = config['tokenizer']['vocab_size']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to PyTorch tensors
    # Pad sequences to the same length for efficient batching
    max_len = max(len(seq) for seq in tokenized_data)
    print(f"Maximum sequence length: {max_len}")
    
    # Create padded tensor
    padded_data = torch.zeros(len(tokenized_data), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(tokenized_data), max_len, dtype=torch.bool)
    
    for i, tokens in enumerate(tokenized_data):
        padded_data[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :len(tokens)] = True
    
    # Save data
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
    """Load tokenized data from disk."""
    data = torch.load(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"Data shape: {data['data'].shape}")
    print(f"Number of samples: {data['num_samples']}")
    return data

def main():
    parser = argparse.ArgumentParser(description="Download and tokenize minipile dataset")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to config file")
    parser.add_argument("--max_samples", type=int, default=None, help="Override max_samples from config")
    parser.add_argument("--load_only", action="store_true", help="Only load existing data")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.max_samples is not None:
        config['dataset']['max_samples'] = args.max_samples
    
    if args.load_only:
        data_path = os.path.join(config['output']['directory'], config['output']['filename'])
        if os.path.exists(data_path):
            load_tokenized_data(data_path)
        else:
            print(f"Data file not found: {data_path}")
    else:
        # Download and tokenize
        tokenized_data = process_dataset(config)
        
        # Save to disk
        save_tokenized_data(tokenized_data, config)

if __name__ == "__main__":
    main()
