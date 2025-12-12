## Small GPT2 based model - training from scratch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import json

# Simple character-level transformer model
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
        
        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

class TextDataset(Dataset):
    def __init__(self, text, char_to_idx, seq_len=128):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.data = [char_to_idx[ch] for ch in text]
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 128
    EPOCHS = 50
    LEARNING_RATE = 3e-4
    PATIENCE = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("Loading data...")
    with open('./data/clean_whatsapp.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text)} characters")
    
    # Create vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Save vocabulary
    os.makedirs('models', exist_ok=True)
    with open('models/vocab_scratch.json', 'w', encoding='utf-8') as f:
        json.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f, ensure_ascii=False)
    
    # Create dataset and dataloader
    dataset = TextDataset(text, char_to_idx, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Initialize model
    print("Initializing model...")
    model = CharTransformer(
        vocab_size=vocab_size,
        embed_dim=256,
        n_heads=8,
        n_layers=6,
        max_seq_len=SEQ_LEN
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # TensorBoard
    writer = SummaryWriter('runs/scratch_training')
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting training with early stopping (patience={PATIENCE})...")
    start_time = time.time()
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
        
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}')
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'vocab_size': vocab_size,
                'config': {
                    'embed_dim': 256,
                    'n_heads': 8,
                    'n_layers': 6,
                    'max_seq_len': SEQ_LEN
                }
            }, 'models/model_scratch_best.pt')
            print(f"✓ New best model saved! Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                print(f"Best loss: {best_loss:.4f}")
                break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'vocab_size': vocab_size,
        'config': {
            'embed_dim': 256,
            'n_heads': 8,
            'n_layers': 6,
            'max_seq_len': SEQ_LEN
        }
    }, 'models/model_scratch_final.pt')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: models/model_scratch_best.pt")
    print(f"{'='*50}")
    
    writer.close()

if __name__ == '__main__':
    train()