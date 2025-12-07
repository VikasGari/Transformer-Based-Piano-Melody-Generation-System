# 🎹 Transformer-Based Piano Melody Generation System

A deep learning system that generates original piano MIDI compositions using transformer-based neural networks. Generate classical, jazz, ambient, and pop piano melodies conditioned on composer, genre, and period metadata.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [System Architecture](#system-architecture)
- [Training](#training)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Conditional Generation**: Generate MIDI based on composer, genre, and music period
- **Transformer Architecture**: State-of-the-art GPT-style decoder for sequence generation
- **Web Interface**: User-friendly Gradio interface for easy interaction
- **Audio Synthesis**: Convert generated MIDI to WAV using FluidSynth
- **Metadata Tokenization**: Advanced tokenization of composer/genre/period metadata
- **Pre-trained Checkpoints**: Use pre-trained models for immediate inference
- **Jupyter Notebooks**: Complete pipeline from preprocessing to inference

## 📁 Project Structure

```
.
├── 01_preprocessing.ipynb           # Data preprocessing and tokenization
├── 02_model_architecture.ipynb      # Model architecture exploration
├── 03_training.ipynb                # Training loop and checkpointing
├── 04_inference.ipynb               # Inference and MIDI generation
├── app.py                           # Gradio web interface
├── model.py                         # Transformer model implementation
├── requirements.txt                 # Python dependencies
├── checkpoints/                     # Pre-trained model checkpoints
│   ├── checkpoint_best.pt           # Best model weights
│   └── checkpoint_latest.pt         # Latest checkpoint
├── processed_data/                  # Tokenized training data
│   ├── id_to_token.json            # Token vocabulary mapping
│   └── train/test splits           # Training/validation data
├── generated_midi/                  # Generated MIDI output files
├── sound_fonts/                     # SF2 soundfont files for audio synthesis
└── aria-midi-v1-deduped-ext/       # ARIA MIDI dataset (optional)
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- Git

### Step 1: Clone Repository
```bash
cd d:\VS Code
git clone https://github.com/VikasGari/Transformer-Based-Piano-Melody-Generation-System.git
cd "Transformer Based Piano Melody Generation System"
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch with CUDA (Optional but Recommended)

For GPU acceleration on Windows:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

See `INSTALL_PYTORCH_CUDA.md` for detailed GPU setup instructions.

### Step 5: Download Soundfonts (Optional, for WAV generation)

Add SF2 soundfont files to the `sound_fonts/` folder:
- General MIDI soundfonts work best
- Example: FluidR3_GM.sf2, SGM-v2.sf2

**Note**: WAV generation requires FluidSynth. On Windows, download from:
- [FluidSynth Windows Binary](https://github.com/FluidSynth/fluidsynth/releases)

## 🎯 Quick Start

### Run the Web Interface
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

**Features:**
- Select genre (classical, jazz, pop, ambient)
- Choose composer (chopin, beethoven, mozart, yiruma, debussy)
- Select music period (baroque, classical, romantic, modern)
- Adjust temperature (0.5-1.5) for creativity
- Set generation length (1000-2500 tokens)
- Download generated MIDI and optional WAV audio

### Run Inference in Python
```python
import torch
from model import PianoMIDIGenerator
import json

# Load model
model = PianoMIDIGenerator(
    vocab_size=2560,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=8192
)

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load vocabulary
with open('processed_data/id_to_token.json') as f:
    id_to_token = json.load(f)

# Generate MIDI
generated_tokens = model.generate(
    metadata_tokens=[...],  # Your metadata token IDs
    max_length=2000,
    temperature=0.8,
    device='cuda'
)
```

## 📚 Detailed Usage

### 1. Preprocessing Data
Open and run `01_preprocessing.ipynb` to:
- Load raw MIDI files from `aria-midi-v1-deduped-ext/`
- Extract metadata (composer, genre, period)
- Tokenize MIDI events
- Create vocabulary mappings

### 2. Explore Model Architecture
Run `02_model_architecture.ipynb` to:
- Understand transformer blocks and attention mechanisms
- Visualize positional encoding
- Inspect token embeddings

### 3. Train Model
Run `03_training.ipynb` to:
- Configure training hyperparameters
- Set up data loaders
- Train transformer model with checkpointing
- Monitor training metrics

### 4. Generate Melodies
Run `04_inference.ipynb` to:
- Load pre-trained weights
- Generate MIDI from metadata conditions
- Convert to WAV audio
- Listen to results

## 🏗️ System Architecture

### Model Components

**PositionalEncoding**: Sinusoidal positional embeddings
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**TransformerBlock**: Multi-head self-attention + Feed-forward
```
Output = LayerNorm(x + MultiHeadAttention(x))
Output = LayerNorm(Output + FeedForward(Output))
```

**Token Vocabulary**:
- Metadata tokens: `[COMPOSER:*]`, `[GENRE:*]`, `[PERIOD:*]`
- MIDI tokens: `[NOTE_ON:pitch]`, `[NOTE_OFF:pitch]`, `[TIME_SHIFT:duration]`, etc.
- Total: ~2,560 tokens

### Hyperparameters
- **d_model**: 512 (embedding dimension)
- **n_heads**: 8 (attention heads)
- **n_layers**: 6 (transformer blocks)
- **d_ff**: 2,048 (feedforward hidden dimension)
- **max_seq_len**: 8,192 tokens
- **dropout**: 0.1

## 🔧 Troubleshooting

### Issue: Module not found errors
**Solution**: Ensure you're in the project directory and virtual environment is activated
```bash
cd "Transformer Based Piano Melody Generation System"
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate    # Linux/macOS
```

### Issue: CUDA out of memory during training
**Solution**: Reduce batch size or max_seq_len in notebooks

### Issue: WAV generation fails
**Solution**: Install FluidSynth and add to PATH, or update CUSTOM_FLUIDSYNTH_PATH in app.py

### Issue: No soundfonts found
**Solution**: Add SF2 files to `sound_fonts/` folder and restart app.py

### Issue: Model loads but generates gibberish
**Solution**: Ensure checkpoint is from a fully-trained model and vocabulary matches

## 📊 Results

The system generates:
- **Classical**: Chopin-inspired etudes, Mozart concertos
- **Jazz**: Improvisation-style piano with swing rhythms
- **Ambient**: Peaceful, atmospheric compositions
- **Pop**: Catchy melodies and chord progressions

Examples in `generated_midi/` folder with naming convention:
```
generated_[genre]_[composer]_[timestamp].mid
```

## 🤝 Contributing

To add new features:
1. Create a feature branch
2. Update relevant notebooks/scripts
3. Test with sample data
4. Submit pull request

## 📝 License

See LICENSE file in repository

## 👨‍💻 Author

Vikas Gari - [GitHub](https://github.com/VikasGari)

## 📞 Support

For issues, questions, or suggestions:
- Open a GitHub issue
- Check existing documentation
- Review notebook comments for detailed explanations

---

**Happy composing! 🎵**
