# checkpoint_best.pt - Download Instructions

## ⚠️ File Not Included in Repository

The `checkpoint_best.pt` file is **not included** in this repository because it exceeds GitHub's file size limits.

## 📥 How to Download

### Option 1: Download from Releases (Recommended)
1. Go to the **Releases** section of this repository
2. Find the latest release
3. Download `checkpoint_best.pt` from the release assets
4. Place it in the `checkpoints/` folder:
   ```
   checkpoints/checkpoint_best.pt
   ```

### Option 2: Build from Scratch
If you prefer to train the model yourself:
1. Open `03_training.ipynb`
2. Configure training parameters
3. Run the notebook to train the model
4. The best checkpoint will be saved automatically to `checkpoints/checkpoint_best.pt`

## 📊 File Details

- **File Name**: `checkpoint_best.pt`
- **Size**: ~300MB
- **Format**: PyTorch checkpoint dictionary
- **Contents**:
  - Model state dict (weights)
  - Optimizer state (if saved)
  - Training epoch/step information
  - Hyperparameters

## ✅ Verification

After downloading, verify the file:

```python
import torch

checkpoint = torch.load('checkpoints/checkpoint_best.pt')
print("Keys in checkpoint:", checkpoint.keys())
print("Model state dict keys:", len(checkpoint['model_state_dict']))
```

Should output model parameters and architecture information.

## 🚀 Usage

Load the checkpoint for inference:

```python
import torch
from model import PianoMIDIGenerator

# Create model
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

print("✅ Model loaded successfully!")
```

## ❓ Troubleshooting

**Error: "No such file or directory"**
- Ensure you're in the correct directory
- Run: `cd "Transformer Based Piano Melody Generation System"`

**Error: "Invalid checkpoint"**
- Download file may be corrupted
- Try downloading again from Releases

**Error: "Model state dict size mismatch"**
- Model hyperparameters don't match checkpoint
- Check that model is initialized with correct parameters:
  - vocab_size=2560, d_model=512, n_heads=8, n_layers=6

## 📝 Notes

- This is the **best performing model** from training (lowest validation loss)
- Trained on the ARIA MIDI dataset
- Conditioned on composer, genre, and music period
- Use `checkpoint_latest.pt` for continued training

---

**Need help?** Check the main `README.md` for complete documentation.
