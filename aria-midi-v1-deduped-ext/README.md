# ARIA MIDI Dataset

## 📥 Dataset Download Instructions

This folder should contain the **ARIA MIDI Dataset (deduplicated extended version)** used for training the piano melody generation model.

The dataset is **not included** in this repository due to its large size (~2GB+).

## 🔗 Download the Dataset

### Option 1: Direct Download (Recommended - Same version used)
Download the exact dataset version used for training:

```
https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-deduped-ext.tar.gz?download=true
```

**Steps:**
1. Click the link above or copy it to your browser
2. Wait for download to complete (~2GB)
3. Extract the `.tar.gz` file:
   ```bash
   tar -xzf aria-midi-v1-deduped-ext.tar.gz
   ```
4. Move contents to this folder (`aria-midi-v1-deduped-ext/`)

### Option 2: Browse Full Dataset on Hugging Face
Visit the complete ARIA MIDI dataset page:

```
https://huggingface.co/datasets/loubb/aria-midi
```

Here you can:
- Browse different dataset versions
- View dataset statistics
- Download alternative versions if needed
- Read the dataset paper and documentation

## 📝 Dataset Information

- **Name**: ARIA MIDI v1 (Deduplicated Extended)
- **Size**: ~2GB (compressed), ~4GB+ (extracted)
- **Format**: MIDI files (.mid)
- **Total Tracks**: ~10,000+ unique compositions
- **Genres**: Classical, Jazz, Pop, Ambient, and more
- **Composers**: Various classical and contemporary composers

## 🎵 Usage

Once extracted, the preprocessing notebook uses this data:

```bash
# Run preprocessing
jupyter notebook 01_preprocessing.ipynb
```

The notebook will:
1. Load MIDI files from `data/` subfolder
2. Extract metadata (composer, genre, period)
3. Tokenize MIDI events
4. Create training/validation datasets

## ⚠️ Important Notes

- **License**: Check `LICENSE.txt` in the dataset folder for usage rights
- **Processing Time**: Initial preprocessing may take 1-2 hours
- **GPU Recommended**: For faster preprocessing

## ❓ Troubleshooting

**Download is slow:**
- Try downloading at a different time
- Use a download manager for resume capability

**File extraction fails:**
- Ensure you have enough disk space
- Install 7-Zip or WinRAR for Windows
- For Linux/macOS: `tar` should be pre-installed

**Preprocessing script can't find files:**
- Check file paths in `01_preprocessing.ipynb`

**Out of memory during preprocessing:**
- Reduce batch size in the notebook
- Process data in chunks

## 📚 References

- **Dataset Paper**: Check Hugging Face page for citations
- **MIDI Format**: https://en.wikipedia.org/wiki/MIDI

## 🚀 Next Steps

Once you have the dataset:

1. Run `01_preprocessing.ipynb` to tokenize MIDI files
2. Run `03_training.ipynb` to train the model
3. Use `04_inference.ipynb` to generate new compositions

---

**Need help?** See the main `README.md` for complete documentation.
