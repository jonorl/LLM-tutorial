import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
        
        print("Model loaded successfully!")
    
    def generate(self, character, location, prompt_word, 
                 max_length=100, temperature=0.8, top_k=50, top_p=0.9,
                 num_return_sequences=1, repetition_penalty=1.2):
        """Generate dialogue based on character, location, and prompt word."""
        # Format input with special tokens
        input_text = f"[CHAR:{character}][LOC:{location}][PROMPT:{prompt_word}]"
        
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
                no_repeat_ngram_size=3
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=False)
            
            # Extract just the dialogue part
            dialogue = full_text.split(f"[PROMPT:{prompt_word}]")[-1].strip()
            dialogue = dialogue.replace('<|endoftext|>', '').strip()
            
            generated_texts.append(dialogue)
        
        return generated_texts

# Initialize the generator
MODEL_PATH = './models/simpsons_final_model_from_GPT/'
generator = SimpsonsGenerator(MODEL_PATH)

def generate_dialogue(character, location, prompt_word, temperature, top_k, top_p, max_length, num_outputs, repetition_penalty):
    """Wrapper function for Gradio interface."""
    try:
        results = generator.generate(
            character=character,
            location=location,
            prompt_word=prompt_word,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            max_length=int(max_length),
            num_return_sequences=int(num_outputs),
            repetition_penalty=repetition_penalty
        )
        
        # Format output
        output = ""
        for i, text in enumerate(results, 1):
            output += f"**Output {i}:**\n\n{character}: {text}\n\n---\n\n"
        
        return output.strip()
    
    except Exception as e:
        return f"Error generating dialogue: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Simpsons Dialogue Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🍩 Simpsons Dialogue Generator
        Generate custom Simpsons dialogue using a fine-tuned GPT-2 model!
        
        Enter a character, location, and prompt word to generate dialogue in the style of The Simpsons.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Parameters")
            
            character = gr.Textbox(
                label="Character Name",
                value="Homer Simpson",
                placeholder="e.g., Homer Simpson, Bart Simpson, Marge Simpson",
                info="The character who will speak the dialogue"
            )
            
            location = gr.Textbox(
                label="Location",
                value="Simpson Home",
                placeholder="e.g., Power Plant, Moe's Tavern, Springfield Elementary",
                info="Where the scene takes place"
            )
            
            prompt_word = gr.Textbox(
                label="Prompt Word",
                value="hey",
                placeholder="e.g., nuclear, doh, beer",
                info="Starting word/topic for the dialogue"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random, Lower = more focused"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k",
                    info="Limits vocabulary to top k tokens"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p (Nucleus Sampling)",
                    info="Cumulative probability threshold"
                )
                
                max_length = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max Length",
                    info="Maximum length of generated text"
                )
                
                num_outputs = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Outputs",
                    info="How many different dialogues to generate"
                )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Repetition Penalty",
                    info="Penalize repeated phrases"
                )
            
            generate_btn = gr.Button("Generate Dialogue 🎬", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Generated Dialogue")
            output = gr.Markdown(label="Output")
    
    # Example inputs
    gr.Markdown("### 💡 Try these examples:")
    gr.Examples(
        examples=[
            ["Homer Simpson", "Power Plant", "nuclear", 0.8, 50, 0.9, 100, 1, 1.2],
            ["Bart Simpson", "Springfield Elementary", "school", 0.9, 50, 0.9, 80, 1, 1.2],
            ["Marge Simpson", "Simpson Home", "family", 0.7, 50, 0.9, 100, 1, 1.2],
            ["Mr. Burns", "Power Plant", "excellent", 0.8, 50, 0.9, 100, 1, 1.2],
            ["Lisa Simpson", "School", "saxophone", 0.8, 50, 0.9, 100, 1, 1.2],
        ],
        inputs=[character, location, prompt_word, temperature, top_k, top_p, max_length, num_outputs, repetition_penalty],
        outputs=output,
        fn=generate_dialogue,
        cache_examples=False,
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_dialogue,
        inputs=[character, location, prompt_word, temperature, top_k, top_p, max_length, num_outputs, repetition_penalty],
        outputs=output
    )
    
    gr.Markdown(
        """
        ---
        ### About
        This app uses a GPT-2 model fine-tuned on Simpsons dialogue to generate new conversations.
        Adjust the parameters to control the creativity and style of the generated text.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Makes it accessible on your network
        server_port=7860  # Default Gradio port
    )