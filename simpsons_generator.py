import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import os

class SimpsonsGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!\n")
    
    def generate(self, character, location, prompt_word, 
                 max_length=100, temperature=0.8, top_k=50, top_p=0.9,
                 num_return_sequences=1, repetition_penalty=1.2):
        """
        Generate dialogue based on character, location, and prompt word.
        
        Args:
            character: Character name (e.g., "Homer Simpson")
            location: Location/setting (e.g., "Power Plant")
            prompt_word: Starting word for the dialogue
            max_length: Maximum length of generated text (default: 100)
            temperature: Sampling temperature (higher = more random) (default: 0.8)
            top_k: Top-k sampling parameter (default: 50)
            top_p: Nucleus sampling parameter (default: 0.9)
            num_return_sequences: Number of sequences to generate (default: 1)
            repetition_penalty: Penalty for repeating tokens (default: 1.2)
        
        Returns:
            List of generated dialogue strings
        """
        # Format input with special tokens
        input_text = f"[CHAR:{character}][LOC:{location}][PROMPT:{prompt_word}]"
        
        print(f"Input: {input_text}")
        print(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}, max_len={max_length}")
        print("-" * 80)
        
        # Encode input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Avoid repeating 3-grams
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            full_text = self.tokenizer.decode(output, skip_special_tokens=False)
            
            # Extract just the dialogue part (after the prompt)
            # Remove the input prefix and special tokens
            dialogue = full_text.split(f"[PROMPT:{prompt_word}]")[-1].strip()
            dialogue = dialogue.replace('<|endoftext|>', '').strip()
            
            generated_texts.append(dialogue)
        
        return generated_texts
    
    def interactive_mode(self):
        """Interactive mode for generating dialogue"""
        print("\n" + "="*80)
        print("SIMPSONS DIALOGUE GENERATOR - Interactive Mode")
        print("="*80)
        print("\nType 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                # Get character
                character = input("Character name (e.g., Homer Simpson): ").strip()
                if character.lower() in ['quit', 'exit']:
                    break
                if not character:
                    character = "Homer Simpson"
                    print(f"  Using default: {character}")
                
                # Get location
                location = input("Location (e.g., Power Plant): ").strip()
                if location.lower() in ['quit', 'exit']:
                    break
                if not location:
                    location = "Simpson Home"
                    print(f"  Using default: {location}")
                
                # Get prompt word
                prompt_word = input("Prompt word (e.g., nuclear): ").strip()
                if prompt_word.lower() in ['quit', 'exit']:
                    break
                if not prompt_word:
                    prompt_word = "hey"
                    print(f"  Using default: {prompt_word}")
                
                # Get parameters (optional)
                temp_input = input("Temperature [0.8]: ").strip()
                temperature = float(temp_input) if temp_input else 0.8
                
                top_k_input = input("Top-k [50]: ").strip()
                top_k = int(top_k_input) if top_k_input else 50
                
                top_p_input = input("Top-p [0.9]: ").strip()
                top_p = float(top_p_input) if top_p_input else 0.9
                
                max_len_input = input("Max length [100]: ").strip()
                max_length = int(max_len_input) if max_len_input else 100
                
                num_seq_input = input("Number of outputs [1]: ").strip()
                num_sequences = int(num_seq_input) if num_seq_input else 1
                
                print("\nGenerating...\n")
                
                # Generate
                results = self.generate(
                    character=character,
                    location=location,
                    prompt_word=prompt_word,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_length=max_length,
                    num_return_sequences=num_sequences
                )
                
                # Display results
                print("\n" + "="*80)
                for i, text in enumerate(results, 1):
                    print(f"\n[Output {i}]")
                    print(f"{character}: {text}")
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue

def main():
    parser = argparse.ArgumentParser(description='Generate Simpsons dialogue')
    parser.add_argument('--model_path', type=str, 
                       default='./models/simpsons_final_model_from_GPT/',
                       help='Path to the trained model')
    parser.add_argument('--character', type=str, default=None,
                       help='Character name (e.g., "Homer Simpson")')
    parser.add_argument('--location', type=str, default=None,
                       help='Location (e.g., "Power Plant")')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Prompt word to start the dialogue')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling (default: 50)')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling (default: 0.9)')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum length of generated text (default: 100)')
    parser.add_argument('--num_outputs', type=int, default=1,
                       help='Number of outputs to generate (default: 1)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                       help='Repetition penalty (default: 1.2)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist!")
        return
    
    # Initialize generator
    generator = SimpsonsGenerator(args.model_path)
    
    # Interactive mode
    if args.interactive or (not args.character and not args.location and not args.prompt):
        generator.interactive_mode()
    else:
        # Command-line mode
        character = args.character or "Homer Simpson"
        location = args.location or "Simpson Home"
        prompt_word = args.prompt or "hey"
        
        results = generator.generate(
            character=character,
            location=location,
            prompt_word=prompt_word,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=args.max_length,
            num_return_sequences=args.num_outputs,
            repetition_penalty=args.repetition_penalty
        )
        
        # Display results
        print("\n" + "="*80)
        print("GENERATED DIALOGUE:")
        print("="*80)
        for i, text in enumerate(results, 1):
            print(f"\n[Output {i}]")
            print(f"{character}: {text}")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()