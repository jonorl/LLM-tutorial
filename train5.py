# train_amd_optim.py
"""
AMD-optimized small-dataset GPT-like training script

Put your training text under ./data/input.txt (or change DATA_PATH below).
Designed for small corpora (~17k words). Use with ROCm-enabled PyTorch on AMD 6950XT.
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
device = "cuda" if torch.cuda.is_available() else "cpu"   # ROCm still exposes cuda device
print("Device chosen:", device)
# Model / training config tuned for small dataset and AMD GPU
block_size = 128           # smaller context for small data
n_embd = 256               # medium embedding size
n_head = 8                 # attention heads (n_embd must be divisible by n_head)
n_layer = 4                # small depth (avoid overfitting)
dropout = 0.35             # regularization
batch_size = 8             # micro-batch on GPU; will accumulate
grad_accum_steps = 8       # effective batch size = batch_size * grad_accum_steps
max_iters = 800
eval_interval = 50
eval_iters = 50
learning_rate = 3e-4
weight_decay = 0.01
warmup_iters = 80
use_amp = True             # mixed precision
use_checkpointing = True   # apply torch.utils.checkpoint on transformer blocks to save memory
label_smoothing = 0.05     # helps generalization on small datasets
patience = 5               # early stopping patience (evaluations)
seed = 42
# =========================

torch.manual_seed(seed)

# Create save dir
os.makedirs(SAVE_DIR, exist_ok=True)

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
text = Path(DATA_PATH).read_text(encoding="utf-8")
tokens = torch.tensor(encode(text), dtype=torch.long)
n_tokens = len(tokens)
print(f"Loaded {n_tokens} tokens from {DATA_PATH}")

# split
n_train = int(0.9 * n_tokens)
train_tokens = tokens[:n_train]
val_tokens = tokens[n_train:]

# -------------------------
# Dataset with random windows (reduces overfitting)
# -------------------------
class RandomTextDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        # give many possible samples but not too huge for small data
        return max(1, (len(self.data) - self.block_size) // 1)

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=False, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=False, drop_last=True)

# -------------------------
# Model: small GPT-like
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
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # scaled dot-product
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
        # optionally apply checkpointing per block to save memory
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
            # label smoothing via cross_entropy's label_smoothing (PyTorch >=1.10)
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
# model init & AMD handling
# -------------------------
model = SmallGPT().to(device)
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model params: {num_params:.2f}M")

# do not use torch.compile on AMD/ROCm (can cause instability)
_use_compile = False
try:
    # quick heuristic for AMD: device name containing "amd" OR torch.version.hip is present
    is_amd = False
    if torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0).lower()
            if "amd" in dev_name or "radeon" in dev_name:
                is_amd = True
        except Exception:
            pass
    # check for hip build
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        is_amd = True
    if not is_amd:
        # try to compile for non-AMD (optional)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                _use_compile = True
                print("torch.compile applied")
            except Exception as e:
                print("torch.compile failed or not beneficial:", e)
    else:
        print("AMD/ROCm detected — skipping torch.compile() for stability")
except Exception as e:
    print("Device detection/compile check error:", e)

# optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it >= max_iters:
        return learning_rate * 0.0
    # cosine decay after warmup
    progress = (it - warmup_iters) / max(1, (max_iters - warmup_iters))
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * learning_rate

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

print("Starting training...")
start_time = time.time()

pbar = tqdm(range(max_iters), desc="training", ncols=120)
for it in pbar:
    # set lr
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # periodic eval
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        print(f"\niter {it:4d}  train_loss {losses['train']:.4f}  val_loss {losses['val']:.4f}  lr {lr:.2e}  elapsed {elapsed:.1f}s")

        if losses["val"] < best_val:
            best_val = losses["val"]
            patience_counter = 0
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
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

    # gradient clipping and optimizer step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    pbar.set_postfix({
        "it": it,
        "lr": f"{lr:.2e}",
        "best_val": f"{best_val:.4f}"
    })

pbar.close()

# save final
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "it": it,
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
    out_ids = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=100)[0].tolist()
    print("=== sample ===")
    print(decode(out_ids))
