import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
import math
from datetime import datetime
from collections import Counter

class SimpleTokenizer:
    """Custom tokenizer built from the dataset"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            words = str(text).lower().split()
            word_counts.update(words)
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        # Add words that meet minimum frequency
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {word_counts.most_common(20)}")
        
    def encode(self, text, max_length=None):
        """Convert text to token indices"""
        words = str(text).lower().split()
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        
        if max_length:
            if len(indices) < max_length:
                indices += [self.word2idx[self.pad_token]] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices):
        """Convert token indices back to text"""
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        # Remove padding and special tokens
        words = [w for w in words if w not in [self.pad_token, self.bos_token, self.eos_token]]
        return ' '.join(words)
    
    def save(self, path):
        """Save tokenizer"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.vocab_size = data['vocab_size']

class SimpsonsDatasetScratch(Dataset):
    def __init__(self, csv_path, tokenizer=None, max_length=128, build_vocab=False):
        self.df = pd.read_csv(csv_path)
        self.max_length = max_length
        
        # Clean data
        self.df = self.df.dropna(subset=['raw_character_text', 'raw_location_text', 'spoken_words'])
        self.df = self.df[self.df['spoken_words'].str.strip() != '']
        
        # Create character and location mappings
        self.characters = sorted(self.df['raw_character_text'].unique())
        self.locations = sorted(self.df['raw_location_text'].unique())
        self.char2idx = {char: idx for idx, char in enumerate(self.characters)}
        self.loc2idx = {loc: idx for idx, loc in enumerate(self.locations)}
        
        print(f"Loaded {len(self.df)} dialogue samples")
        print(f"Unique characters: {len(self.characters)}")
        print(f"Unique locations: {len(self.locations)}")
        
        # Build or use existing tokenizer
        if build_vocab:
            self.tokenizer = SimpleTokenizer()
            all_texts = []
            for _, row in self.df.iterrows():
                spoken = str(row['spoken_words']).strip()
                words = spoken.split()
                prompt_word = words[0] if words else "the"
                text = f"{row['raw_character_text']} {row['raw_location_text']} {prompt_word} {spoken}"
                all_texts.append(text)
            
            self.tokenizer.build_vocab(all_texts, min_freq=3)
        else:
            self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get character and location indices
        char_idx = self.char2idx[row['raw_character_text']]
        loc_idx = self.loc2idx[row['raw_location_text']]
        
        # Get prompt word
        spoken = str(row['spoken_words']).strip()
        words = spoken.split()
        prompt_word = words[0].lower() if words else "the"
        
        # Encode text
        text_indices = self.tokenizer.encode(spoken, max_length=self.max_length)
        
        return {
            'char_idx': torch.tensor(char_idx, dtype=torch.long),
            'loc_idx': torch.tensor(loc_idx, dtype=torch.long),
            'prompt_word': prompt_word,
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'text_length': min(len(words), self.max_length)
        }

class TransformerDialogueModel(nn.Module):
    """Transformer-based dialogue generation model"""
    def __init__(self, vocab_size, num_characters, num_locations, 
                 embed_dim=256, num_heads=8, num_layers=6, 
                 max_seq_length=128, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.char_embedding = nn.Embedding(num_characters, embed_dim)
        self.loc_embedding = nn.Embedding(num_locations, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, text_indices, char_idx, loc_idx):
        batch_size, seq_len = text_indices.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=text_indices.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(text_indices)
        pos_embeds = self.position_embedding(positions)
        
        # Condition embeddings (character + location)
        char_embeds = self.char_embedding(char_idx).unsqueeze(1)
        loc_embeds = self.loc_embedding(loc_idx).unsqueeze(1)
        
        # Combine conditioning
        memory = char_embeds + loc_embeds  # [batch, 1, embed_dim]
        
        # Add embeddings
        x = self.dropout(token_embeds + pos_embeds)
        
        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(text_indices.device)
        
        # Transformer decoder
        x = self.transformer_decoder(x, memory, tgt_mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits

class ScratchTrainer:
    def __init__(self, csv_path='simpsons_data.csv', output_dir='./scratch_model', log_dir='./logs_scratch'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.output_dir = output_dir
        self.log_dir = log_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Load dataset and build vocabulary
        print("Loading dataset and building vocabulary...")
        self.dataset = SimpsonsDatasetScratch(csv_path, build_vocab=True)
        
        # Initialize model
        print("Initializing model...")
        self.model = TransformerDialogueModel(
            vocab_size=self.dataset.tokenizer.vocab_size,
            num_characters=len(self.dataset.characters),
            num_locations=len(self.dataset.locations),
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            max_seq_length=128,
            dropout=0.1
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'{log_dir}/run_{timestamp}')
        
    def train(self, batch_size=32, epochs=10, learning_rate=1e-3, 
              gradient_accumulation_steps=1, warmup_steps=1000):
        
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(dataloader) * epochs // gradient_accumulation_steps
        
        # Warmup scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.tokenizer.word2idx['<PAD>'])
        
        print(f"\nTraining Configuration:")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}\n")
        
        self.model.train()
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                text_indices = batch['text_indices'].to(self.device)
                char_idx = batch['char_idx'].to(self.device)
                loc_idx = batch['loc_idx'].to(self.device)
                
                # Prepare inputs and targets
                input_indices = text_indices[:, :-1]
                target_indices = text_indices[:, 1:]
                
                # Forward pass
                logits = self.model(input_indices, char_idx, loc_idx)
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_indices.reshape(-1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Log to TensorBoard
                    self.writer.add_scalar('Loss/train', loss.item() * gradient_accumulation_steps, global_step)
                    self.writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)
                    self.writer.add_scalar('Perplexity', math.exp(min(loss.item() * gradient_accumulation_steps, 10)), global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'ppl': f'{math.exp(min(loss.item() * gradient_accumulation_steps, 10)):.2f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            # Epoch summary
            avg_loss = epoch_loss / len(dataloader)
            perplexity = math.exp(min(avg_loss, 10))
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}\n")
            
            self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
            self.writer.add_scalar('Perplexity/epoch', perplexity, epoch)
            
            # Save checkpoint if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = f"{self.output_dir}/best_model"
                self.save_model(checkpoint_path)
                print(f"New best model saved! Loss: {best_loss:.4f}\n")
            
            # Save periodic checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"{self.output_dir}/checkpoint_epoch_{epoch+1}"
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}\n")
        
        # Save final model
        final_path = f"{self.output_dir}/final_model"
        self.save_model(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")
        
        self.writer.close()
    
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.dataset.tokenizer.vocab_size,
            'num_characters': len(self.dataset.characters),
            'num_locations': len(self.dataset.locations),
        }, f"{path}/model.pt")
        
        # Save tokenizer
        self.dataset.tokenizer.save(f"{path}/tokenizer.json")
        
        # Save character and location mappings
        with open(f"{path}/mappings.json", 'w') as f:
            json.dump({
                'characters': self.dataset.characters,
                'locations': self.dataset.locations,
                'char2idx': self.dataset.char2idx,
                'loc2idx': self.dataset.loc2idx
            }, f, indent=2)
        
        # Save config
        config = {
            'dataset_size': len(self.dataset),
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def generate_sample(self, character, location, prompt_word, max_length=50, temperature=0.8):
        """Generate a sample dialogue"""
        self.model.eval()
        
        # Get indices
        char_idx = self.dataset.char2idx.get(character, 0)
        loc_idx = self.dataset.loc2idx.get(location, 0)
        
        # Start with prompt word
        tokens = self.dataset.tokenizer.encode(prompt_word, max_length=1)
        generated = tokens.copy()
        
        char_tensor = torch.tensor([char_idx], device=self.device)
        loc_tensor = torch.tensor([loc_idx], device=self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([generated], device=self.device)
                
                # Forward pass
                logits = self.model(input_tensor, char_tensor, loc_tensor)
                
                # Get last token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if EOS or PAD
                if next_token in [self.dataset.tokenizer.word2idx['<EOS>'], 
                                 self.dataset.tokenizer.word2idx['<PAD>']]:
                    break
                
                generated.append(next_token)
        
        generated_text = self.dataset.tokenizer.decode(generated)
        self.model.train()
        
        return generated_text

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "./data/simpsons.csv"
    OUTPUT_DIR = "./simpsons_scratch_model"
    LOG_DIR = "./logs_scratch"
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    GRAD_ACCUM_STEPS = 1
    
    # Initialize trainer
    trainer = ScratchTrainer(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        log_dir=LOG_DIR
    )
    
    # Generate sample before training
    print("\n" + "="*60)
    print("SAMPLE GENERATION BEFORE TRAINING:")
    print("="*60)
    sample = trainer.generate_sample("Homer Simpson", "Simpson Home", "hey", max_length=30)
    print(f"Character: Homer Simpson")
    print(f"Location: Simpson Home")
    print(f"Prompt: hey")
    print(f"Generated: {sample}")
    print("="*60 + "\n")
    
    # Train the model
    trainer.train(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=1000
    )
    
    # Generate samples after training
    print("\n" + "="*60)
    print("SAMPLE GENERATIONS AFTER TRAINING:")
    print("="*60)
    
    test_cases = [
        ("Homer Simpson", "Simpson Home", "hey"),
        ("Marge Simpson", "Kitchen", "oh"),
        ("Bart Simpson", "School", "don't"),
    ]
    
    for char, loc, prompt in test_cases:
        sample = trainer.generate_sample(char, loc, prompt, max_length=30)
        print(f"\nCharacter: {char}")
        print(f"Location: {loc}")
        print(f"Prompt: {prompt}")
        print(f"Generated: {sample}")
    
    print("\n" + "="*60 + "\n")
    
    print("To view training progress, run:")
    print(f"tensorboard --logdir {LOG_DIR}")