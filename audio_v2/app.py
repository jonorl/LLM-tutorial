import os
import torch
import scipy.io.wavfile
import gradio as gr
import logging
from huggingface_hub import snapshot_download

# Suppress transformers warnings
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

# Fix for transformers generation config issue
import transformers.generation.configuration_utils as utils
def dummy_to_diff_dict(self): return {}
utils.GenerationConfig.to_diff_dict = dummy_to_diff_dict

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model files...")

# Download base XTTS v2 model
base_model_path = snapshot_download(
    repo_id="coqui/XTTS-v2",
    allow_patterns=["*.json", "vocab.json", "*.pth"],
)

checkpoint_path = snapshot_download(
    repo_id="jonorl/pepe", 
    allow_patterns=["*.pth"],
)

speaker_repo = snapshot_download(
    repo_id="jonorl/pepe", 
    allow_patterns=["*.wav"],
)

# Find the speaker wav file
speaker_wav = None
for root, dirs, files in os.walk(speaker_repo):
    for file in files:
        if file.endswith('.wav'):
            speaker_wav = os.path.join(root, file)
            break
    if speaker_wav:
        break

print(f"Base model path: {base_model_path}")
print(f"Checkpoint path: {checkpoint_path}")
print(f"Speaker wav: {speaker_wav}")

# Load config
config = XttsConfig()
config.load_json(os.path.join(base_model_path, "config.json"))
config.model_args.tokenizer_file = os.path.join(base_model_path, "vocab.json")
config.model_args.dvae_checkpoint = os.path.join(base_model_path, "dvae.pth")
config.model_args.mel_norm_file = os.path.join(base_model_path, "mel_stats.pth")

# Initialize model
print("Initializing model...")
model = Xtts.init_from_config(config)

# Find the checkpoint file
checkpoint_file = None
for root, dirs, files in os.walk(checkpoint_path):
    for file in files:
        if file.endswith('.pth'):
            checkpoint_file = os.path.join(root, file)
            break
    if checkpoint_file:
        break

print(f"Loading checkpoint from: {checkpoint_file}")

# Load fine-tuned weights
model.load_checkpoint(
    config, 
    checkpoint_path=checkpoint_file,
    vocab_path=os.path.join(base_model_path, "vocab.json"),
    use_deepspeed=False
)

# Move to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model.cuda()
else:
    print("GPU not available, using CPU (inference will be slower)")

print("Model loaded successfully!")

def generate_speech(
    text,
    temperature=0.7,
    top_p=0.85,
    top_k=50,
    seed=67
):
    """
    Generate speech from text using the fine-tuned voice clone model.
    
    Args:
        text: The text to synthesize
        temperature: Controls randomness (0.1-1.0, higher = more varied)
        top_p: Nucleus sampling threshold (0.1-1.0)
        top_k: Top-k sampling value (1-100)
        seed: Random seed for reproducibility
    """
    if not text.strip():
        return None, "Please enter some text to synthesize."
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        print(f"Generating audio for: {text[:50]}...")
        
        # Generate audio
        outputs = model.synthesize(
            text,
            config,
            speaker_wav=speaker_wav,
            gpt_cond_len=3,
            language="es",  # Change if needed
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
        )
        
        # Save to temporary file
        output_path = "output.wav"
        scipy.io.wavfile.write(output_path, 24000, outputs["wav"])
        
        return output_path, "✅ Audio generated successfully!"
        
    except Exception as e:
        error_msg = f"❌ Error generating audio: {str(e)}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
with gr.Blocks(title="Voice Clone - Pedro") as demo:
    gr.Markdown(
        """
        # 🎙️ Voice Cloning Demo
        ### AI-powered voice synthesis using XTTS v2
        
        Enter your text in Spanish and adjust the parameters to generate speech 
        in the cloned voice. Play with the settings to find your preferred output!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Hola, soy Pedro con la voz clonada hecha por inteligencia artificial...",
                lines=3,
                value="Hola, ¿cómo estás? Esta es una demostración de clonación de voz."
            )
            
            with gr.Accordion("Advanced Parameters", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                    info="Controls randomness - higher = more varied prosody"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling - controls diversity"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top K",
                    info="Limits vocabulary selection"
                )
                
                seed = gr.Number(
                    value=67,
                    label="Seed",
                    info="Random seed for reproducibility",
                    precision=0
                )
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath"
            )
            status_output = gr.Textbox(
                label="Status",
                interactive=False
            )
    
    gr.Markdown(
        """
        ### Tips for best results:
        - **Temperature**: Lower (0.3-0.5) for consistent, clear speech. Higher (0.7-0.9) for more natural variation.
        - **Top P & Top K**: Default values work well for most cases. Adjust if output sounds too repetitive or too random.
        - **Seed**: Use the same seed to reproduce identical audio from the same text.
        
        ---
        *Powered by [Coqui XTTS v2](https://github.com/coqui-ai/TTS) | Fine-tuned for Spanish*
        """
    )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, temperature, top_p, top_k, seed],
        outputs=[audio_output, status_output]
    )
    
    # Example button
    gr.Examples(
        examples=[
            ["Hola, soy Pedro. Bienvenidos a mi demostración de clonación de voz.", 0.7, 0.85, 50, 67],
            ["La inteligencia artificial ha avanzado mucho en los últimos años.", 0.5, 0.9, 40, 42],
            ["¿Cómo va todo? Espero que estés teniendo un excelente día.", 0.8, 0.8, 60, 123],
        ],
        inputs=[text_input, temperature, top_p, top_k, seed],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)