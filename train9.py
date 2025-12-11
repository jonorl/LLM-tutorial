# train_improved.py
"""
Optimized small-dataset GPT-like training script with auto-naming and enhanced features.

Put your training text under ./data/clean_whatsapp.txt (or change DATA_PATH below).
Improvements:
- Automatic timestamped run naming
- Better hyperparameter tuning for ~17k words
- Enhanced logging and monitoring
- Gradient norm tracking
- Perplexity metrics
"""

import os
import math
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# optional: tiktoken if available for GPT-2 BPE
try:
    import tiktoken
    _has_tiktoken = True
except Exception:
    _has_tiktoken = False

# =========================
# User / hyper-parameters
# =========================
DATA_PATH = "./data/clean_whatsapp.txt"
SAVE_DIR = "models"
BASE_LOG_DIR = "runs"  # Base directory for all runs
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device chosen:", device)

# Auto-generate run name with timestamp and key hyperparameters
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"gpt_{timestamp}"  # Will be extended with hyperparams

# Model / training config (OPTIMIZED FOR AMD 6950XT - 16GB VRAM)
# Let's actually USE that VRAM! 🚀
block_size = 256           # 2x context length for better long-range dependencies
n_embd = 512               # 2x embedding size - bigger is better
n_head = 16                # 2x attention heads for richer representations
n_layer = 8                # 2x layers - let's go deeper!
dropout = 0.35             # Lower dropout since bigger model = better regularization
batch_size = 32            # 4x larger batches - utilize that VRAM!
grad_accum_steps = 8       # Effective batch = 32*8 = 256 (huge!)
max_iters = 2000           # More iterations for the bigger model
eval_interval = 100        # Evaluate every 100 iterations
eval_iters = 100           # More thorough evaluation
learning_rate = 3e-4       # Higher LR for larger batches
weight_decay = 0.08        # Stronger weight decay for bigger model
warmup_iters = 200         # Longer warmup for stability
use_amp = True             # Mixed precision - even faster!
use_checkpointing = False  # Turn OFF checkpointing - we have the VRAM!
label_smoothing = 0.10     # Moderate label smoothing
patience = 8               # More patience for bigger model
seed = 42

# Extend run name with key hyperparameters
run_name += f"_lr{learning_rate}_bs{batch_size}x{grad_accum_steps}_drop{dropout}_ls{label_smoothing}"
LOG_DIR = os.path.join(BASE_LOG_DIR, run_name)
# =========================

torch.manual_seed(seed)

# Create save and log dirs
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(LOG_DIR)
print(f"TensorBoard logs: {LOG_DIR}")
print(f"Run name: {run_name}")

# -------------------------
# tokenizer (fallback to simple byte-level)
# -------------------------
if _has_tiktoken:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
    decode = lambda ids: enc.decode(ids)
    vocab_size = enc.n_vocab
    print("Using tiktoken gpt2 encoder, vocab_size:", vocab_size)
else:
    print("tiktoken not available; falling back to byte-level tokenizer.")
    vocab_size = 256
    def encode(s):
        return [c for c in s.encode("utf-8")]
    def decode(ids):
        return bytes(ids).decode("utf-8", errors="replace")

# -------------------------
# load data
# -------------------------
try:
    text = Path(DATA_PATH).read_text(encoding="utf-8")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

tokens = torch.tensor(encode(text), dtype=torch.long)
n_tokens = len(tokens)
print(f"Loaded {n_tokens:,} tokens from {DATA_PATH}")
print(f"Estimated ~{len(text.split()):,} words")

# Log dataset info
writer.add_text("Dataset Info", f"Tokens: {n_tokens:,}, Words: ~{len(text.split()):,}")

# split (90/10)
n_train = int(0.9 * n_tokens)
train_tokens = tokens[:n_train]
val_tokens = tokens[n_train:]
print(f"Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens")

# -------------------------
# Dataset with random windows
# -------------------------
class RandomTextDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        # 10x multiplier for better randomness per epoch
        return max(1, (len(self.data) - self.block_size) * 10)

    def __getitem__(self, idx):
        if len(self.data) <= self.block_size:
            start = 0
        else:
            start = torch.randint(0, len(self.data) - self.block_size, (1,)).item()
        x = self.data[start:start + self.block_size].long()
        y = self.data[start + 1:start + self.block_size + 1].long()
        return x, y

train_dataset = RandomTextDataset(train_tokens, block_size)
val_dataset = RandomTextDataset(val_tokens, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True, 
                          persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True, prefetch_factor=2)

# -------------------------
# Model: small GPT-like Architecture
# -------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
    def __init__(self, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SmallGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
        pos = self.pos_emb(pos_ids)
        x = tok + pos
        
        for block in self.blocks:
            # No checkpointing - we have VRAM to spare!
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=label_smoothing)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

# -------------------------
# model init
# -------------------------
model = SmallGPT().to(device)
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model params: {num_params:.2f}M")

# Log hyperparameters to TensorBoard
hparams = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "grad_accum_steps": grad_accum_steps,
    "dropout": dropout,
    "label_smoothing": label_smoothing,
    "n_layer": n_layer,
    "n_embd": n_embd,
    "n_head": n_head,
    "block_size": block_size,
    "weight_decay": weight_decay,
    "max_iters": max_iters,
}
writer.add_hparams(hparams, {})

# torch.compile check (AMD 6950XT should handle this fine!)
_use_compile = False
try:
    is_amd = False
    if torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0).lower()
            if "amd" in dev_name or "radeon" in dev_name:
                is_amd = True
                print(f"AMD GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception:
            pass
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        is_amd = True
    
    # Try compile even on AMD - modern ROCm should handle it
    # But disable CUDA graphs for gradient accumulation compatibility
    if hasattr(torch, "compile"):
        try:
            print("Attempting torch.compile (this may take a minute)...")
            # Disable CUDA graphs to avoid the overwriting issue with grad accumulation
            model = torch.compile(model, mode="reduce-overhead", 
                                options={"triton.cudagraphs": False})
            _use_compile = True
            print("✓ torch.compile applied successfully (CUDA graphs disabled for stability)!")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
            print("Training will proceed without compilation (still fast!)")
except Exception as e:
    print("Compile check error:", e)

# optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                             betas=(0.9, 0.95), weight_decay=weight_decay)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it >= max_iters:
        return learning_rate * 0.1
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    min_lr = learning_rate * 0.1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -------------------------
# helper: estimate loss & perplexity
# -------------------------
@torch.no_grad()
def estimate_loss(model, iters=eval_iters):
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    for split, loader in (("train", train_loader), ("val", val_loader)):
        total = 0
        running = 0.0
        for i, (xb, yb) in enumerate(loader):
            if i >= iters:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(xb, yb)
            running += float(loss.detach())
            total += 1
        losses[split] = running / max(1, total)
    model.train()
    return losses

# -------------------------
# training loop
# -------------------------
best_val = float("inf")
patience_counter = 0
train_iter = iter(train_loader)
global_step = 0

print("\n" + "="*60)
print("Starting training...")
print(f"Effective Batch Size: {batch_size * grad_accum_steps}")
print(f"Run: {run_name}")
print("="*60)
start_time = time.time()

pbar = tqdm(range(max_iters), desc="Training", ncols=130)
for it in pbar:
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr
    
    writer.add_scalar('Hyperparameters/learning_rate', lr, global_step)

    # Evaluation
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        
        # Calculate perplexity
        train_ppl = math.exp(min(losses['train'], 20))  # Cap for numerical stability
        val_ppl = math.exp(min(losses['val'], 20))
        
        # Log to TensorBoard
        writer.add_scalars('Loss', {'train': losses['train'], 'val': losses['val']}, global_step)
        writer.add_scalars('Perplexity', {'train': train_ppl, 'val': val_ppl}, global_step)

        print(f"\n[{it:4d}] train_loss: {losses['train']:.4f} (ppl: {train_ppl:.2f}) | "
              f"val_loss: {losses['val']:.4f} (ppl: {val_ppl:.2f}) | "
              f"lr: {lr:.2e} | time: {elapsed:.1f}s")

        if losses["val"] < best_val:
            best_val = losses["val"]
            patience_counter = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "iteration": it,
                "val_loss": best_val,
                "train_loss": losses["train"],
                "config": {
                    "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
                    "block_size": block_size, "vocab_size": vocab_size,
                    "dropout": dropout, "label_smoothing": label_smoothing
                }
            }
            torch.save(checkpoint, os.path.join(SAVE_DIR, f"best_{run_name}.pt"))
            print(f"✓ Saved best model (val improved to {best_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping: no improvement for {patience} evaluations")
                break

    # Training step with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    micro_losses = []
    
    for micro in range(grad_accum_steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb = xb.to(device)
        yb = yb.to(device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            _, loss = model(xb, yb)
            loss = loss / grad_accum_steps

        micro_losses.append(loss.detach() * grad_accum_steps)
        scaler.scale(loss).backward()
    
    # Gradient clipping and logging
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    
    # Logging
    mean_loss = torch.stack(micro_losses).mean().item()
    writer.add_scalar('Loss/train_step', mean_loss, global_step)
    writer.add_scalar('Gradients/norm', grad_norm, global_step)
    
    global_step += 1
    pbar.set_postfix({
        "iter": it,
        "loss": f"{mean_loss:.4f}",
        "best_val": f"{best_val:.4f}",
        "lr": f"{lr:.2e}",
        "grad": f"{grad_norm:.2f}"
    })

pbar.close()

# Save final model
final_iter = it if 'it' in locals() else max_iters
final_checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "iteration": final_iter,
    "config": {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "vocab_size": vocab_size,
        "dropout": dropout, "label_smoothing": label_smoothing
    }
}
torch.save(final_checkpoint, os.path.join(SAVE_DIR, f"final_{run_name}.pt"))

# Generate samples
print("\n" + "="*60)
print("Generating samples...")
model.eval()
for temp in [0.7, 0.9, 1.1]:
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        out_ids = model.generate(context, max_new_tokens=300, 
                                temperature=temp, top_k=100)[0].tolist()
        generated = decode(out_ids)
        
        print(f"\n--- Temperature: {temp} ---")
        print(generated[:500])  # Print first 500 chars
        writer.add_text(f"Sample_temp_{temp}", generated, final_iter)

writer.close()
print("\n" + "="*60)
print(f"Training complete! Logs saved to: {LOG_DIR}")
print(f"Best validation loss: {best_val:.4f}")
print("="*60)