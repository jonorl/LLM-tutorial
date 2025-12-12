## Small GPT2 based model - training using pre-trained model DeepESP/gpt2-spanish

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import time
import os

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Split text into chunks
        self.examples = []
        tokens = tokenizer.encode(text)
        
        for i in range(0, len(tokens) - max_length, max_length // 2):
            chunk = tokens[i:i + max_length + 1]
            if len(chunk) > 1:
                self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Pad if necessary
        if len(x) < self.max_length:
            padding = self.max_length - len(x)
            x = torch.cat([x, torch.zeros(padding, dtype=torch.long)])
            y = torch.cat([y, torch.full((padding,), -100, dtype=torch.long)])
        
        return x, y

def train():
    # Hyperparameters
    BATCH_SIZE = 8
    MAX_LENGTH = 128
    EPOCHS = 50
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 100
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
    
    # Load pretrained model and tokenizer
    print("Loading pretrained GPT-2 model (multilingual)...")
    model_name = 'DeepESP/gpt2-spanish'  # GPT-2 small works well for Spanish
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 🚨 CRITICAL CHANGE: Tokens now match the raw chat data format (Name:)
    new_tokens = [
    "Leonardo Araya:",
    "Pepe:",
    "Jon:",
    "Erich Orlowski:",
    "Panda:",
    "Alan:"
    ]

    num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # VITAL FIX: Resize the token embeddings to match the new tokenizer size
    new_vocab_size = len(tokenizer)
    print(f"Resizing model embeddings from {model.config.vocab_size} to {new_vocab_size}...")
    model.resize_token_embeddings(new_vocab_size)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save tokenizer
    os.makedirs('models', exist_ok=True)
    tokenizer.save_pretrained('models/tokenizer_pretrained')
    
    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # TensorBoard
    writer = SummaryWriter('runs/pretrained_training')
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    print(f"\nStarting fine-tuning with early stopping (patience={PATIENCE})...")
    start_time = time.time()
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)
        
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        
        print(f'Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}')
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save_pretrained('models/model_pretrained_best')
            print(f"✓ New best model saved! Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                print(f"Best loss: {best_loss:.4f}")
                break
    
    # Save final model
    model.save_pretrained('models/model_pretrained_final')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Fine-tuning completed!")
    print(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: models/model_pretrained_best")
    print(f"{'='*50}")
    
    writer.close()

if __name__ == '__main__':
    train()