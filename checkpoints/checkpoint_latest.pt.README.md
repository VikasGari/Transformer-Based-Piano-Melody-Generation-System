# checkpoint_latest.pt - Download Instructions

## ⚠️ File Not Included in Repository

The `checkpoint_latest.pt` file is **not included** in this repository because it exceeds GitHub's file size limits.

## 📥 How to Download

### Option 1: Download from Releases (Recommended)
1. Go to the **Releases** section of this repository
2. Find the latest release
3. Download `checkpoint_latest.pt` from the release assets
4. Place it in the `checkpoints/` folder:
   ```
   checkpoints/checkpoint_latest.pt
   ```

### Option 2: Build from Scratch
If you prefer to train the model yourself:
1. Open `03_training.ipynb`
2. Configure training parameters
3. Run the notebook to train the model
4. The latest checkpoint will be saved automatically to `checkpoints/checkpoint_latest.pt` at each epoch

## 📊 File Details

- **File Name**: `checkpoint_latest.pt`
- **Size**: ~300MB
- **Format**: PyTorch checkpoint dictionary
- **Contents**:
  - Model state dict (weights)
  - Optimizer state dict
  - Learning rate scheduler state
  - Current epoch/step information
  - Training loss history
  - Full training metadata

## ✅ Verification

After downloading, verify the file:

```python
import torch

checkpoint = torch.load('checkpoints/checkpoint_latest.pt')
print("Keys in checkpoint:", checkpoint.keys())
print("Model state dict keys:", len(checkpoint['model_state_dict']))
if 'epoch' in checkpoint:
    print(f"Checkpoint from epoch: {checkpoint['epoch']}")
```

Should output model parameters and training metadata.

## 🚀 Usage

### For Inference (Recommended: Use checkpoint_best.pt instead)

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
checkpoint = torch.load('checkpoints/checkpoint_latest.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Model loaded successfully!")
```

### For Continued Training

```python
import torch
from model import PianoMIDIGenerator
import torch.optim as optim

# Create model
model = PianoMIDIGenerator(...)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_latest.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint.get('epoch', 0)

print(f"✅ Resuming training from epoch {start_epoch}")

# Continue training...
```

## 📝 Key Differences

| Aspect | checkpoint_best.pt | checkpoint_latest.pt |
|--------|-------------------|----------------------|
| **When to use** | Inference/Evaluation | Continued training |
| **Loss** | Lowest validation loss | Most recent state |
| **Optimizer state** | May not be saved | Always saved |
| **Use case** | Production inference | Resume training |

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

**Error: "CUDA out of memory when loading"**
- Load on CPU first: `checkpoint = torch.load(..., map_location='cpu')`
- Then move to GPU if needed

## 💡 Recommendation

- **For inference**: Download and use `checkpoint_best.pt` (better quality)
- **For training continuation**: Use `checkpoint_latest.pt` (has optimizer state)
- **For production**: Always use `checkpoint_best.pt`

---

**Need help?** Check the main `README.md` for complete documentation.
