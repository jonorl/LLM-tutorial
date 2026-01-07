# XTTS_v2 Argentine Spanish Voice Cloning

This project documents the process of fine-tuning the **XTTS_v2** model to clone a specific Argentine Spanish voice using WhatsApp voice notes as the primary data source.

## Project Overview
The goal was to move beyond generic Spanish TTS by capturing the specific "Argentine tone" (Rioplatense) through fine-tuning on localized audio data.

### 1. Data Pipeline
* **Source:** Audio messages extracted from WhatsApp in `.ogg` format.
* **Preprocessing:**
    * Converted audio from `.ogg` to `.wav`.
    * Segmented audio into ~9-second batches for optimal training.
    * Transcribed audio into a `metadata.csv` using the **ljspeech** formatter.
* **Structure:**
    ```text
    /audio_v2/
    ├── metadata.csv
    ├── *.wav (segmented files)
    └── checkpoints/
    ```

### Model & Training
* **Base Model:** `tts_models/multilingual/multi-dataset/xtts_v2`.
* **Architecture:** Fine-tuning via `GPTTrainer` with custom `GPTArgs`.
* **Hardware used:** AMD Radeon **RX 6950 XT** (Running via ROCm/PyTorch).
* **Key Fixes:** * Disabled `TOKENIZERS_PARALLELISM` to prevent recursion errors on Python 3.12.
    * Set `num_workers=0` and `num_loader_workers=0` to ensure stability on the local environment.
* **Hyperparameters:** * **Optimizer:** AdamW with weight decay $1e-2$.
    * **Learning Rate:** $5e-6$.
    * **Batch Size:** 2.
    * **Epochs:** 10.

### 2. Inference (`test_xtts_v2.py`)
To generate audio from the fine-tuned weights:
* **The Nuclear Fix:** Implemented a bypass for the `transformers` library `to_diff_dict` error by monkey-patching `GenerationConfig` to allow compatibility with newer Python versions.
* **Generation:** Uses a reference `speaker_wav` to maintain vocal characteristics while applying the fine-tuned linguistic nuances.
* **Settings:** Temperature: 0.7, Top_p: 0.85, Top_k: 50.
* **Output:** Audio is saved at 24kHz using `scipy.io.wavfile`.

## Challenges & Solutions
* **Local AMD GPU:** Configured the environment to handle training and inference on a 6950 XT, overcoming typical CUDA-centric library limitations.
* **Regional Nuance:** Successfully captured Argentine-specific phrasing and cadence (e.g., *"Bueno muchachos, gracias por la bancada"*) through targeted fine-tuning.

## How to Run
1.  **Train:** Ensure `dvae.pth` and `mel_stats.pth` are present in the model folder, then run:
    ```bash
    python train_xtts_v2.py
    ```
2.  **Test:** Update the `checkpoint_dir` in `test_xtts_v2.py` to your latest run and execute:
    ```bash
    python test_xtts_v2.py
    ```