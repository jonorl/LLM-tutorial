### Optimized LLM Training from Scratch

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import math
import time
from tqdm import tqdm
import os

# hyperparameters
batch_size = 32  # reduced for memory
block_size = 128  # reduced context length
max_iters = 600  # reduced - stop before overfitting
eval_interval = 50  # more frequent evaluation
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 192  # reduced model size
n_head = 3  # reduced heads
n_layer = 3  # reduced layers
dropout = 0.55  # increased dropout for regularization
warmup_iters = 100
grad_accum_steps = 16  # increased to maintain effective batch size of 128
use_amp = True  # mixed precision training
compile_mode = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"
# ------------

print(f"Using device: {device}")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Better tokenization with tiktoken (BPE)
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab

print(f"Vocabulary size: {vocab_size}")

# Load and tokenize text
with open('./data/clean_whatsapp.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")

# Dataset class for better data loading
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# Create dataloaders
train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)  # pin_memory=False for AMD
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

# Get batch iterator
def get_batch(split):
    loader = train_loader if split == 'train' else val_loader
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yield x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        loader = train_loader if split == 'train' else val_loader
        for k, (X, Y) in enumerate(loader):
            if k >= eval_iters:
                break
            X, Y = X.to(device), Y.to(device)
            with torch.amp.autocast(device_type=device, enabled=use_amp):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled dot-product attention
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ FFN with GELU activation """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Changed from ReLU to GELU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block with Pre-LayerNorm """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Weight initialization
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
model = GPTLanguageModel()
model = model.to(device)

# Compile model for speedup (PyTorch 2.0+)
# Disabled for AMD GPUs due to CUDA graphs compatibility issues
try:
    is_amd = torch.cuda.is_available() and 'amd' in torch.cuda.get_device_name(0).lower()
    
    if is_amd:
        print(f"AMD GPU detected ({torch.cuda.get_device_name(0)})")
        print("Skipping torch.compile() - better compatibility with ROCm/HIP")
        # AMD ROCm has issues with CUDA graphs in torch.compile
        # Training will still be fast without it
    else:
        # NVIDIA: use compilation
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully!")
except Exception as e:
    print(f"torch.compile not available or failed ({e}), skipping compilation")

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.2)

# Learning rate scheduler with warmup
def get_lr(it):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > max_iters:
        return learning_rate * 0.1
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * 0.1 + coeff * (learning_rate - learning_rate * 0.1)

# Mixed precision scaler
scaler = torch.amp.GradScaler(device, enabled=use_amp)

# Create models directory
os.makedirs('models', exist_ok=True)

# Training loop
print("\n" + "="*50)
print("Starting training...")
print("="*50)
train_start_time = time.time()
train_iter = iter(train_loader)

# Progress bar for training
pbar = tqdm(range(max_iters), desc="Training", ncols=100)

best_val_loss = float('inf')
patience = 3
patience_counter = 0

for iter_num in pbar:
    
    # Update learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - train_start_time
        print(f"\nstep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}, time {elapsed:.2f}s")
        
        # Early stopping check
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_num': iter_num,
                'val_loss': best_val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'block_size': block_size,
                    'dropout': dropout
                }
            }, os.path.join('models', 'gpt_model_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping: validation loss hasn't improved for {patience} evaluations")
                break
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f"{losses['train']:.4f}",
            'val_loss': f"{losses['val']:.4f}",
            'best_val': f"{best_val_loss:.4f}",
            'lr': f"{lr:.2e}"
        })
    
    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        
        xb, yb = xb.to(device), yb.to(device)
        
        with torch.amp.autocast(device_type=device, enabled=use_amp):
            logits, loss = model(xb, yb)
            loss = loss / grad_accum_steps  # Scale loss
        
        scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

pbar.close()

# Training complete
total_time = time.time() - train_start_time
print("\n" + "="*50)
print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
print("="*50)

# Save the model
model_path = os.path.join('models', 'gpt_model.pt')
print(f"\nSaving model to {model_path}...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'iter_num': max_iters,
    'config': {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout
    }
}, model_path)
print("Model saved successfully!")

# Generate sample
print("\n" + "="*50)
print("Generating sample text...")
print("="*50)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=200)[0].tolist())
print(generated)