## Prediction generator
## Use: python generator.py --model {modelname} "prompt here"

import torch
import torch.nn as nn
import sys
import json
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# =======================================================
# NEW: GENERATION HYPERPARAMETERS (Easily changeable here)
# =======================================================
# Scratch Model
SCRATCH_TEMP = 0.8
SCRATCH_TOP_K = 40
SCRATCH_MAX_LEN = 200

# Pretrained Model
PRETRAINED_TEMP = 0.8
PRETRAINED_TOP_K = 50
PRETRAINED_TOP_P = 0.9
PRETRAINED_MAX_LEN = 100
# =======================================================


# Character-level transformer (same architecture as training)
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_heads=8, n_layers=6, max_seq_len=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb
        
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

def generate_scratch(prompt, model_path='models/model_scratch_best.pt', 
                     max_length=SCRATCH_MAX_LEN, temperature=SCRATCH_TEMP, top_k=SCRATCH_TOP_K):
    """Generate text using the from-scratch model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    with open('models/vocab_scratch.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    char_to_idx = {k: int(v) for k, v in vocab['char_to_idx'].items()}
    idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}
    vocab_size = len(char_to_idx)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = CharTransformer(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Encode prompt
    context = [char_to_idx.get(ch, 0) for ch in prompt]
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = prompt
    
    with torch.no_grad():
        for _ in range(max_length):
            if context.size(1) > config['max_seq_len']:
                context = context[:, -config['max_seq_len']:]
            
            logits = model(context)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            next_char = idx_to_char[next_token.item()]
            generated += next_char
            
            context = torch.cat([context, next_token], dim=1)
            
            # Stop at end of sentence
            if next_char in ['.', '!', '?'] and len(generated) > len(prompt) + 20:
                break
    
    return generated

def generate_pretrained(prompt, model_path='models/model_pretrained_best',
                       max_length=PRETRAINED_MAX_LEN, temperature=PRETRAINED_TEMP, 
                       top_k=PRETRAINED_TOP_K, top_p=PRETRAINED_TOP_P):
    """Generate text using the pretrained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('models/tokenizer_pretrained')
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()
    
    # FIX: Explicitly set pad token to prevent warning when pad and eos are the same
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.size(1) + max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated

def main():
    if len(sys.argv) < 2:
        print("Usage: python generator.py '<prompt>' [--model scratch|pretrained]")
        print("\nExample:")
        print("  python generator.py 'Hola mundo'")
        print("  python generator.py 'Hola mundo' --model pretrained")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    # Check for model type argument
    model_type = 'both'
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_type = sys.argv[idx + 1]
    
    print(f"\nPrompt: '{prompt}'")
    print("="*60)
    
    # Check which models are available
    scratch_available = os.path.exists('models/model_scratch_best.pt')
    pretrained_available = os.path.exists('models/model_pretrained_best')
    
    if model_type in ['scratch', 'both'] and scratch_available:
        print("\n[From-Scratch Model]")
        print("-"*60)
        try:
            # NOTE: Parameters are now defined at the top of the file
            text = generate_scratch(prompt)
            print(text)
        except Exception as e:
            print(f"Error generating with scratch model: {e}")
    elif model_type == 'scratch' and not scratch_available:
        print("\nFrom-scratch model not found. Please train it first with:")
        print("  python train_from_scratch.py")
    
    if model_type in ['pretrained', 'both'] and pretrained_available:
        print("\n[Pretrained Model]")
        print("-"*60)
        try:
            # NOTE: Parameters are now defined at the top of the file
            text = generate_pretrained(prompt)
            print(text)
        except Exception as e:
            print(f"Error generating with pretrained model: {e}")
    elif model_type == 'pretrained' and not pretrained_available:
        print("\nPretrained model not found. Please train it first with:")
        print("  python train_pretrained.py")
    
    if not scratch_available and not pretrained_available:
        print("\nNo trained models found. Please train a model first:")
        print("  python train_from_scratch.py")
        print("  python train_pretrained.py")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()