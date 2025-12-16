import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
from datetime import datetime

class SimpsonsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Clean data
        self.df = self.df.dropna(subset=['raw_character_text', 'raw_location_text', 'spoken_words'])
        self.df = self.df[self.df['spoken_words'].str.strip() != '']
        
        print(f"Loaded {len(self.df)} dialogue samples")
        print(f"Unique characters: {self.df['raw_character_text'].nunique()}")
        print(f"Unique locations: {self.df['raw_location_text'].nunique()}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Format: [CHAR:Homer][LOC:Power Plant][PROMPT:nuclear] spoken words here
        # Extract first word from spoken_words as prompt
        spoken = str(row['spoken_words']).strip()
        words = spoken.split()
        prompt_word = words[0] if words else "the"
        
        # Create formatted input
        formatted = (
            f"[CHAR:{row['raw_character_text']}]"
            f"[LOC:{row['raw_location_text']}]"
            f"[PROMPT:{prompt_word}] {spoken}"
        )
        
        # Tokenize
        encoding = self.tokenizer(
            formatted,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class SimpsonsTrainer:
    def __init__(self, model_name='gpt2', csv_path='./data/simpsons.csv', 
                 output_dir='./models', log_dir='./logs'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        print("Loading tokenizer and model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Setup directories
        self.output_dir = output_dir
        self.log_dir = log_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = SimpsonsDataset(csv_path, self.tokenizer)
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'{log_dir}/run_{timestamp}')
        
    def train(self, batch_size=8, epochs=3, learning_rate=5e-5, 
              gradient_accumulation_steps=4, warmup_steps=500):
        
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\nTraining Configuration:")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}\n")
        
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / gradient_accumulation_steps
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
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
            
            # Epoch summary
            avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}\n")
            
            self.writer.add_scalar('Loss/epoch', avg_loss, epoch)
            
            # Save checkpoint
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
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training config
        config = {
            'model_name': 'gpt2',
            'dataset_size': len(self.dataset),
            'timestamp': datetime.now().isoformat()
        }
        with open(f"{path}/training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def generate_sample(self, character, location, prompt_word, max_length=100):
        """Generate a sample dialogue for testing"""
        self.model.eval()
        
        input_text = f"[CHAR:{character}][LOC:{location}][PROMPT:{prompt_word}]"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        self.model.train()
        
        return generated_text

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "./data/simpsons.csv"  # Update this path
    OUTPUT_DIR = "./models"
    LOG_DIR = "./runs"
    
    # Training parameters
    BATCH_SIZE = 8          # Adjust based on GPU memory
    EPOCHS = 3              # Start with 3, can increase
    LEARNING_RATE = 5e-5
    GRAD_ACCUM_STEPS = 4    # Effective batch size = 32
    
    # Initialize trainer
    trainer = SimpsonsTrainer(
        model_name='gpt2',
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        log_dir=LOG_DIR
    )
    
    # Generate sample before training
    print("\n" + "="*60)
    print("SAMPLE GENERATION BEFORE TRAINING:")
    print("="*60)
    sample = trainer.generate_sample("Homer Simpson", "Power Plant", "nuclear")
    print(sample)
    print("="*60 + "\n")
    
    # Train the model
    trainer.train(
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=500
    )
    
    # Generate sample after training
    print("\n" + "="*60)
    print("SAMPLE GENERATION AFTER TRAINING:")
    print("="*60)
    sample = trainer.generate_sample("Homer Simpson", "Power Plant", "nuclear")
    print(sample)
    print("="*60 + "\n")
    
    print("To view training progress, run:")
    print(f"tensorboard --logdir {LOG_DIR}")