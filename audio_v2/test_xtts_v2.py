import os
import torch
import scipy.io.wavfile
import logging

SEED = 67

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. THE NUCLEAR FIX: Completely bypass the broken 'to_diff_dict' in transformers
import transformers.generation.configuration_utils as utils
def dummy_to_diff_dict(self): return {}
utils.GenerationConfig.to_diff_dict = dummy_to_diff_dict

# Also suppress the logger just in case
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- PATHS ---
checkpoint_dir = "/home/jonorl/Git/LLM-tutorial/audio_v2/checkpoints/xtts_arg_es-January-07-2026_04+25PM-3139000"
base_model_path = "/home/jonorl/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
out_path = "/home/jonorl/Git/LLM-tutorial/audio_v2/output.wav"
speaker_wav = "/home/jonorl/Git/LLM-tutorial/audio_v2/samples/sample5.wav" 

# 2. LOAD CONFIG
config = XttsConfig()
config.load_json(os.path.join(base_model_path, "config.json"))
config.model_args.tokenizer_file = os.path.join(base_model_path, "vocab.json")
config.model_args.dvae_checkpoint = os.path.join(base_model_path, "dvae.pth")
config.model_args.mel_norm_file = os.path.join(base_model_path, "mel_stats.pth")

# 3. LOAD MODEL
print("Initializing model...")
model = Xtts.init_from_config(config)

print("Loading fine-tuned weights...")
model.load_checkpoint(
    config, 
    checkpoint_path=os.path.join(checkpoint_dir, "best_model_822.pth"), 
    vocab_path=os.path.join(base_model_path, "vocab.json"),
    use_deepspeed=False
)
model.cuda() 

# 4. INFERENCE
print("Generating audio...")
outputs = model.synthesize(
    "Hola, soy Pedro con la voz clonada hecha por inteligencia artificial, como va todo?",
    config,
    speaker_wav=speaker_wav,
    gpt_cond_len=3,
    language="es",
    temperature=0.7,
    top_p=0.85,
    top_k=50,
)

# 5. SAVE
scipy.io.wavfile.write(out_path, 24000, outputs["wav"])
print(f"Success! Saved to: {out_path}")