import os
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# optional: tiktoken if available for GPT-2 BPE
try:
    import tiktoken
    _has_tiktoken = True
except Exception:
    _has_tiktoken = False

# --- Configuration (Must match training script) ---
# Hardcoded to match the original script's settings for loading the checkpoint
# The model config is loaded from the checkpoint, but these are needed for model definition
block_size = 256
dropout = 0.35 # Used in model definition
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device chosen: {device}")
# ------------------------------------------------

# -------------------------
# Tokenizer functions
# -------------------------
if _has_tiktoken:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
    decode = lambda ids: enc.decode(ids)
    vocab_size = enc.n_vocab
else:
    vocab_size = 256
    def encode(s):
        return [c for c in s.encode("utf-8")]
    def decode(ids):
        return bytes(ids).decode("utf-8", errors="replace")

# -------------------------
# Model Architecture (Must match training script)
# -------------------------

# Define the necessary components from the training script
# The model dimensions (n_embd, n_head, n_layer) will be updated after loading the checkpoint config.

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, n_embd, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SmallGPT(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, block_size, vocab_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weights init is skipped for simplicity on load, as we load state_dict

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(pos_ids)
        x = tok + pos
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=None):
        for _ in range(max_new_tokens):
            # Crop context to the model's block_size
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set logits of tokens outside the top_k to -Inf
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# -------------------------
# Main Generator Logic
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text with a SmallGPT model.")
    parser.add_argument("prompt", type=str, help="The starting text/words for generation.")
    parser.add_argument("--model_name", type=str, 
                        default=None,
                        help="The name of the checkpoint file in the 'models' directory (e.g., best_gpt_...pt). If None, tries to find the latest 'best_' file.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff.")
    args = parser.parse_args()

    # --- Find the Checkpoint ---
    model_dir = "models"
    if args.model_name:
        checkpoint_path = os.path.join(model_dir, args.model_name)
    else:
        # Find the latest 'best_' checkpoint
        try:
            best_files = sorted(Path(model_dir).glob("best_*.pt"), 
                                key=os.path.getmtime, reverse=True)
            if not best_files:
                print(f"Error: No 'best_*.pt' checkpoint found in '{model_dir}'.")
                return
            checkpoint_path = str(best_files[0])
            print(f"Loading latest checkpoint: {Path(checkpoint_path).name}")
        except FileNotFoundError:
            print(f"Error: Directory '{model_dir}' not found.")
            return

    # --- Load Checkpoint and Config ---
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
    except Exception as e:
        print(f"Error loading checkpoint or config from {checkpoint_path}: {e}")
        return

    # --- Initialize Model ---
    model = SmallGPT(
        n_embd=config['n_embd'], 
        n_head=config['n_head'], 
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        vocab_size=config['vocab_size'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # --- Prepare Input ---
    start_ids = encode(args.prompt)
    if not start_ids:
        print("Error: Prompt could not be encoded.")
        return
        
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # --- Generate Text ---
    min_tokens, max_tokens = 150, 250
    max_new_tokens = random.randint(min_tokens, max_tokens)
    
    print("-" * 50)
    print(f"Prompt: **{args.prompt}**")
    print(f"Generating {max_new_tokens} tokens (Temp: {args.temperature}, Top-k: {args.top_k})")
    print("-" * 50)

    with torch.no_grad():
        out_ids = model.generate(context, max_new_tokens=max_new_tokens, 
                                temperature=args.temperature, top_k=args.top_k)[0].tolist()
        
    # Remove the input prompt tokens from the output for clean display
    generated_text = decode(out_ids[len(start_ids):])
    
    print(args.prompt + generated_text)
    print("-" * 50)

if __name__ == '__main__':
    main()