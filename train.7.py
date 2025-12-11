# train6.py
"""
Optimized small-dataset GPT-like training script with TensorBoard integration.

Put your training text under ./data/clean_whatsapp.txt (or change DATA_PATH below).
Based on user's successful train5.py, incorporating TensorBoard for monitoring.
"""

import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# TensorBoard
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
LOG_DIR = "runs/small_gpt_run_7" # TensorBoard log directory
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device chosen:", device)

# Model / training config (tuned for small dataset)
block_size = 128           # context length (from train5.py)
n_embd = 256               # embedding size (from train5.py)
n_head = 8                 # attention heads (from train5.py)
n_layer = 4                # small depth (from train5.py)
dropout = 0.50             # Increased dropout slightly from 0.35 (train5) for stronger regularization
batch_size = 8             # micro-batch on GPU
grad_accum_steps = 32      # Increased from 8 (train5) -> Effective Batch Size = 8 * 16 = 128 (Better for stability/gradient flow)
max_iters = 1000           # Increased max iters from 800 (train5) for better convergence
eval_interval = 50
eval_iters = 50
learning_rate = 2e-4       # (from train5.py)
weight_decay = 0.05        # (from train5.py)
warmup_iters = 100         # Increased warmup slightly (from 80)
use_amp = True             # mixed precision
use_checkpointing = True   # memory saving (from train5.py)
label_smoothing = 0.10     # Increased label smoothing from 0.05 (train5) for stronger generalization
patience = 5               # early stopping patience
seed = 42
# =========================

torch.manual_seed(seed)

# Create save and log dirs
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
# Initialize TensorBoard writer
writer = SummaryWriter(LOG_DIR)
print(f"TensorBoard logs being written to: {LOG_DIR}")

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
    # fallback: byte-level vocab (0-255)
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
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure the data directory and file exist.")
    exit()

tokens = torch.tensor(encode(text), dtype=torch.long)
n_tokens = len(tokens)
print(f"Loaded {n_tokens} tokens from {DATA_PATH}")

# split
n_train = int(0.9 * n_tokens)
train_tokens = tokens[:n_train]
val_tokens = tokens[n_train:]
print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

# -------------------------
# Dataset with random windows (from train5.py)
# -------------------------
class RandomTextDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        # Increased sample space for better randomness
        return max(1, (len(self.data) - self.block_size) * 10)

    def __getitem__(self, idx):
        # sample a random start index each call (stochastic)
        if len(self.data) <= self.block_size:
            start = 0
        else:
            start = torch.randint(0, len(self.data) - self.block_size, (1,)).item()
        x = self.data[start:start + self.block_size].long()
        y = self.data[start + 1:start + self.block_size + 1].long()
        return x, y

train_dataset = RandomTextDataset(train_tokens, block_size)
val_dataset = RandomTextDataset(val_tokens, block_size)

# Using more workers is usually better, but keeping 2 from train5.py for compatibility
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=False, drop_last=True)

# -------------------------
# Model: small GPT-like Architecture (from train5.py)
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
        # Pre-LayerNorm (more stable)
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward()

    def forward(self, x):
        # Add residual connection before LayerNorm (GPT-2 style)
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

        # init
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
        # apply checkpointing per block to save memory
        for block in self.blocks:
            if use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # flatten
            logits = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            # label smoothing
            loss = F.cross_entropy(logits, targets_flat, label_smoothing=label_smoothing)
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
# model init & AMD handling (from train5.py)
# -------------------------
model = SmallGPT().to(device)
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model params: {num_params:.2f}M")
writer.add_text("Model Config", f"Params: {num_params:.2f}M, Layers: {n_layer}, Embed: {n_embd}, Block: {block_size}")

# do not use torch.compile on AMD/ROCm (can cause instability)
_use_compile = False
try:
    is_amd = False
    if torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0).lower()
            if "amd" in dev_name or "radeon" in dev_name:
                is_amd = True
        except Exception:
            pass
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        is_amd = True
    if not is_amd and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            _use_compile = True
            print("torch.compile applied")
        except Exception as e:
            print("torch.compile failed or not beneficial:", e)
    elif is_amd:
        print("AMD/ROCm detected — skipping torch.compile() for stability")
except Exception as e:
    print("Device detection/compile check error:", e)

# optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it >= max_iters:
        # Minimum learning rate to avoid complete stop
        return learning_rate * 0.1
    # cosine decay after warmup (min_lr is 10% of base LR)
    decay_ratio = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    min_lr = learning_rate * 0.1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# amp scaler
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -------------------------
# helper: estimate loss (on small eval set)
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
            with torch.cuda.amp.autocast(enabled=use_amp):
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
global_step = 0 # for TensorBoard logging

print("\n" + "="*50)
print("Starting training with TensorBoard logging...")
print("Effective Batch Size:", batch_size * grad_accum_steps)
print("="*50)
start_time = time.time()

pbar = tqdm(range(max_iters), desc="training", ncols=120)
for it in pbar:
    # set lr
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr
    
    # Log learning rate
    writer.add_scalar('Hyperparameters/learning_rate', lr, global_step)

    # periodic eval
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        
        # Log losses to TensorBoard
        writer.add_scalars('Loss/CrossEntropy', {'train': losses['train'], 'val': losses['val']}, global_step)

        print(f"\niter {it:4d}  train_loss {losses['train']:.4f}  val_loss {losses['val']:.4f}  lr {lr:.2e}  elapsed {elapsed:.1f}s")

        if losses["val"] < best_val:
            best_val = losses["val"]
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "it": it,
                "val_loss": best_val,
                "config": {
                    "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
                    "block_size": block_size, "vocab_size": vocab_size
                }
            }, os.path.join(SAVE_DIR, "best.pt"))
            print("Saved best model (val improved).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: no improvement for {patience} evals.")
                break

    # gradient accumulation steps
    optimizer.zero_grad(set_to_none=True)
    # Track the loss over the micro-steps for logging
    micro_step_losses = []
    
    for micro in range(grad_accum_steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb = xb.to(device)
        yb = yb.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(xb, yb)
            loss = loss / grad_accum_steps # Scale loss by accumulation factor

        # Log the unscaled loss per micro-step before scaling for backward
        micro_step_losses.append(loss.detach() * grad_accum_steps)
        scaler.scale(loss).backward()
    
    # Log the mean training loss over the accumulated steps
    mean_micro_loss = torch.stack(micro_step_losses).mean().item()
    writer.add_scalar('Loss/Training_Loss_Accumulated', mean_micro_loss, global_step)

    # gradient clipping and optimizer step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    global_step += 1
    pbar.set_postfix({
        "it": it,
        "lr": f"{lr:.2e}",
        "best_val": f"{best_val:.4f}",
        "loss": f"{mean_micro_loss:.4f}"
    })

pbar.close()
writer.close()

# save final
final_iter = it if 'it' in locals() else max_iters
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "it": final_iter,
    "config": {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "vocab_size": vocab_size
    }
}, os.path.join(SAVE_DIR, "final.pt"))
print("Training finished and model saved.")

# generate sample
model.eval()
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out_ids = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=100)[0].tolist()
    generated_text = decode(out_ids)
    
    # Log sample text to TensorBoard
    writer.add_text("Generated Sample", f"Iteration {final_iter}:\n{generated_text}", final_iter)
    
    print("\n" + "="*50)
    print("=== sample ===")
    print(generated_text)
    print("="*50)