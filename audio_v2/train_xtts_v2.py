import os
import json
from pathlib import Path
from TTS.utils.manage import ModelManager
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from trainer import Trainer, TrainerArgs

# 1. FIX FOR RECURSION ERROR (Python 3.12)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT_PATH = Path("/home/jonorl/Git/LLM-tutorial/audio_v2")

# 2. SETUP PATHS
# The standard XTTS v2 download doesn't include separate training files
# We need to either use the XTTS checkpoint directly or skip preloading
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
manager = ModelManager()
download_result = manager.download_model(model_name)

if isinstance(download_result, list):
    model_path = download_result[0]
else:
    base_path = os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")), "tts")
    model_path = os.path.join(base_path, model_name.replace("/", "--"))

print(f"Confirmed model path: {model_path}")
print(f"Available files: {os.listdir(model_path)}")

# Verify required files
tokenizer_file = os.path.join(model_path, "vocab.json")
model_checkpoint = os.path.join(model_path, "model.pth")

if not os.path.exists(tokenizer_file):
    raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
if not os.path.exists(model_checkpoint):
    raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

print(f"Using tokenizer: {tokenizer_file}")
print(f"Using checkpoint: {model_checkpoint}")

# 3. LOAD CONFIG
with open(os.path.join(model_path, "config.json"), "r") as f:
    base_config = json.load(f)

# 4. INITIALIZE MODEL ARGS
# Check for required training files
dvae_file = os.path.join(model_path, "dvae.pth")
mel_stats_file = os.path.join(model_path, "mel_stats.pth")

missing_files = []
if not os.path.exists(dvae_file):
    missing_files.append("dvae.pth")
if not os.path.exists(mel_stats_file):
    missing_files.append("mel_stats.pth")

if missing_files:
    print(f"\n⚠ Missing required files: {', '.join(missing_files)}")
    print("Please run: python download_xtts_training_files.py")
    raise FileNotFoundError(f"Missing files: {missing_files}")

model_args = GPTArgs(
    tokenizer_file=tokenizer_file,
    xtts_checkpoint=model_checkpoint,
    dvae_checkpoint=dvae_file,
    mel_norm_file=mel_stats_file,
)

# Sync architecture parameters from base config
# But preserve our local file paths
local_files = {
    'tokenizer_file': tokenizer_file,
    'xtts_checkpoint': model_checkpoint,
    'dvae_checkpoint': dvae_file,
    'mel_norm_file': mel_stats_file,
}

for key, value in base_config["model_args"].items():
    # Skip None values and file paths we've set locally
    if value is not None and key not in local_files:
        setattr(model_args, key, value)

# Ensure all our file paths are set correctly
for key, path in local_files.items():
    setattr(model_args, key, path)

print(f"\nModel args configured:")
print(f"  - tokenizer_file: {model_args.tokenizer_file}")
print(f"  - xtts_checkpoint: {model_args.xtts_checkpoint}")
print(f"  - dvae_checkpoint: {model_args.dvae_checkpoint}")
print(f"  - mel_norm_file: {model_args.mel_norm_file}")
print(f"  - gpt_n_model_channels: {model_args.gpt_n_model_channels}")
print(f"  - gpt_layers: {model_args.gpt_layers}")

# 5. DATASET CONFIG
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="xtts_dataset",
    path=str(ROOT_PATH),
    meta_file_train="metadata.csv",
    language="es",
)

# 6. TRAINING CONFIG
config = GPTTrainerConfig(
    output_path=str(ROOT_PATH / "checkpoints"),
    model_args=model_args,
    batch_size=2,
    eval_batch_size=2,
    eval_split_size=5,
    num_loader_workers=0,  # Fix for Python 3.12 recursion
    num_eval_loader_workers=0,
    project_name="xtts_fine_tuning",
    run_name="xtts_arg_es",
    epochs=10,
    optimizer="AdamW",  # Specify optimizer
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-6,  # Learning rate for fine-tuning
)

# 7. INITIALIZE MODEL
print("\nInitializing model from config...")
try:
    model = GPTTrainer.init_from_config(config)
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"✗ Error initializing model: {e}")
    print("\nIf the error is about missing DVAE or mel_stats files,")
    print("you may need to train from scratch or use a different model source.")
    raise

# 8. LOAD SAMPLES
print("\nLoading training samples...")
train_samples, eval_samples = load_tts_samples(
    [config_dataset],
    eval_split=True,
    eval_split_size=config.eval_split_size,
)
print(f"✓ Loaded {len(train_samples)} training samples, {len(eval_samples)} eval samples")

# 9. CREATE TRAINER
args = TrainerArgs()
trainer = Trainer(
    args,
    config,
    output_path=str(ROOT_PATH / "checkpoints"),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Worker fix for Python 3.12
trainer.num_workers = 0 
trainer.eval_num_workers = 0

# 10. START TRAINING
print("\nStarting training...")
trainer.fit()