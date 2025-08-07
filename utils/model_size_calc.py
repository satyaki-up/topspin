import yaml
import argparse
import sys
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'trainer'))
from llama_model import LLaMAModel

def load_model_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_parameters_theoretical(config: dict) -> int:
    model_config = config['model']
    
    n_layers = model_config['n_layers']
    hidden_size = model_config['hidden_size']
    n_heads = model_config['n_heads']
    head_dim = model_config['head_dim']
    ffn_size = model_config['ffn_size']
    vocab_size = model_config['vocab_size']
    
    embedding_params = vocab_size * hidden_size
    
    attention_params_per_layer = (
        4 * hidden_size * (n_heads * head_dim) +  # wq, wk, wv, wo
        0  # no bias terms
    )
    
    ffn_params_per_layer = (
        hidden_size * ffn_size +  # w1
        ffn_size * hidden_size +  # w2
        hidden_size * ffn_size    # w3
    )
    
    rms_norm_params_per_layer = hidden_size * 2  # 2 RMSNorm layers per transformer block
    
    transformer_block_params = attention_params_per_layer + ffn_params_per_layer + rms_norm_params_per_layer
    
    total_params = (
        embedding_params +  # token embeddings (shared with output)
        n_layers * transformer_block_params +  # transformer layers
        hidden_size  # final RMSNorm
        # Note: output layer shares weights with embeddings, so no additional parameters
    )
    
    return total_params

def calculate_parameters_actual(config: dict) -> int:
    model = LLaMAModel(config['model'])
    return sum(p.numel() for p in model.parameters())

def main():
    parser = argparse.ArgumentParser(description="Calculate model parameters from config")
    parser.add_argument("--config", default="configs/model_50m.yaml", help="Path to model config file")
    
    args = parser.parse_args()
    
    try:
        config = load_model_config(args.config)
        
        theoretical_params = calculate_parameters_theoretical(config)
        actual_params = calculate_parameters_actual(config)
        
        theoretical_millions = theoretical_params / 1_000_000
        actual_millions = actual_params / 1_000_000
        
        print(f"Model configuration: {config['model']}")
        print(f"Theoretical parameters: {theoretical_params:,} ({theoretical_millions:.2f}M)")
        print(f"Actual parameters: {actual_params:,} ({actual_millions:.2f}M)")
        
        if theoretical_params != actual_params:
            diff = abs(theoretical_params - actual_params)
            diff_millions = diff / 1_000_000
            print(f"Difference: {diff:,} parameters ({diff_millions:.2f}M)")
            print("Warning: Theoretical and actual parameter counts differ!")
        else:
            print("✓ Theoretical and actual parameter counts match!")
        
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
