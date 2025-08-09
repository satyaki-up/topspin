import torch
import argparse
import sys
import os
import yaml
import tiktoken

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'trainer'))
from llama_model import LLaMAModel

def load_checkpoint(checkpoint_path: str = "models/checkpoint_10.pt", device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = LLaMAModel(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8):
    device = next(model.parameters()).device
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eot_token:
                break
                
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Inference script for trained LLaMA model")
    parser.add_argument("text", help="Input text prompt for the model")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists("models/checkpoint_10.pt"):
        print("Error: Checkpoint not found at models/checkpoint_10.pt")
        sys.exit(1)
    
    model, config = load_checkpoint(device=device)
    tokenizer = get_tokenizer()
    
    response = generate_text(
        model, 
        tokenizer, 
        args.text, 
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(response)

if __name__ == "__main__":
    main()
