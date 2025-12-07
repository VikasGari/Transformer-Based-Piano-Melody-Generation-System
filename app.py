"""
Gradio Web App for Piano MIDI Generation
"""
import torch
import torch.nn as nn
import json
from pathlib import Path
import re
from typing import List, Optional, Dict
import mido
from datetime import datetime
import subprocess
import os
import random
import gradio as gr

# Import model architecture
from model import PianoMIDIGenerator, PositionalEncoding, TransformerBlock


# Configuration

DATA_DIR = Path("processed_data")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("generated_midi")
OUTPUT_DIR.mkdir(exist_ok=True)
SOUNDFONT_DIR = Path("sound_fonts")

# Custom fluidsynth path (Windows)
CUSTOM_FLUIDSYNTH_PATH = r"C:\Users\Vikas Gari\Downloads\fluidsynth-v2.5.1-win10-x64-cpp11\fluidsynth-v2.5.1-win10-x64-cpp11\bin"
FLUIDSYNTH_EXE = os.path.join(CUSTOM_FLUIDSYNTH_PATH, "fluidsynth.exe")


# Load Model and Vocabulary (Global)

print("Loading model and vocabulary...")

# Load vocabulary
with open(DATA_DIR / "vocab.json", 'r') as f:
    vocab = json.load(f)

with open(DATA_DIR / "id_to_token.json", 'r') as f:
    id_to_token = json.load(f)
    id_to_token = {int(k): v for k, v in id_to_token.items()}

vocab_size = len(vocab)
pad_token_id = vocab.get('<PAD>', 0)

# Load model configuration
with open(DATA_DIR / "preprocessing_config.json", 'r') as f:
    preprocess_config = json.load(f)

MODEL_CONFIG = {
    'vocab_size': vocab_size,
    'max_seq_length': preprocess_config['max_sequence_length'],
    'd_model': 512,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 2048,
    'dropout': 0.0,
    'pad_token_id': pad_token_id,
}

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f"Using device: {device}")

# Load model checkpoint
checkpoint_path = CHECKPOINT_DIR / 'checkpoint_best.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model = PianoMIDIGenerator(MODEL_CONFIG)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("✅ Model loaded")


# Tokenizers

class MetadataTokenizer:
    def __init__(self, include_composer=True, top_n_composers=100):
        self.include_composer = include_composer
        self.valid_genres = {'classical', 'pop', 'soundtrack', 'jazz', 'rock', 'folk', 'ambient', 'ragtime', 'blues', 'atonal'}
        self.valid_periods = {'contemporary', 'modern', 'romantic', 'classical', 'baroque', 'impressionist'}
        self.top_composers = self._load_top_composers(top_n_composers)
    
    def _load_top_composers(self, n):
        top = {'hisaishi', 'satie', 'yiruma', 'einaudi', 'joplin', 'chopin', 'beethoven', 'bach', 'mozart', 'debussy',
               'schubert', 'schumann', 'liszt', 'rachmaninoff', 'tchaikovsky', 'ravel', 'poulenc', 'faure', 'bartok'}
        return {self._normalize_composer(c) for c in top}
    
    def _normalize_composer(self, composer):
        if not composer:
            return ""
        normalized = composer.lower().strip()
        normalized = normalized.replace('é', 'e').replace('è', 'e').replace('á', 'a').replace('à', 'a')
        normalized = normalized.replace('í', 'i').replace('ì', 'i').replace('ó', 'o').replace('ò', 'o')
        normalized = normalized.replace('ú', 'u').replace('ù', 'u').replace('ñ', 'n')
        normalized = re.sub(r'[^a-z0-9\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def metadata_to_tokens(self, metadata, include_start=True):
        tokens = []
        if include_start:
            tokens.append("START")
        
        if metadata.get('genre'):
            genre = metadata['genre'].lower().strip()
            if genre in self.valid_genres:
                tokens.append(f"GENRE:{genre}")
        
        if metadata.get('music_period'):
            period = metadata['music_period'].lower().strip()
            if period in self.valid_periods:
                tokens.append(f"PERIOD:{period}")
        
        if self.include_composer and metadata.get('composer'):
            composer = self._normalize_composer(metadata['composer'])
            if composer in self.top_composers:
                tokens.append(f"COMPOSER:{composer}")
        
        return tokens

meta_tokenizer = MetadataTokenizer(include_composer=True)

class MIDITokenizer:
    def __init__(self, time_quantization=10):
        self.time_quantization = time_quantization
    
    def tokens_to_midi(self, tokens: List[str], output_path: Path, tempo=120, ticks_per_beat=500):
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        tempo_us = mido.bpm2tempo(tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_us))
        
        current_time_ticks = 0
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token.startswith('START') or token.startswith('GENRE:') or token.startswith('PERIOD:') or token.startswith('COMPOSER:') or token in ['<END>', '<PAD>', '<UNK>']:
                i += 1
                continue
            if token.startswith('TIME_SHIFT:'):
                # Accumulate all consecutive TIME_SHIFT tokens
                accumulated_ticks = 0
                while i < len(tokens) and tokens[i].startswith('TIME_SHIFT:'):
                    try:
                        time_ticks = int(tokens[i].split(':')[1])
                        accumulated_ticks += time_ticks
                    except (ValueError, IndexError):
                        pass
                    i += 1
                current_time_ticks = accumulated_ticks
                continue  # Continue to process the next token (which should be a note event)
            
            elif token.startswith('NOTE_ON:'):
                try:
                    note = int(token.split(':')[1])
                    velocity = 64  # Default velocity
                    
                    if i + 1 < len(tokens) and tokens[i + 1].startswith('VELOCITY:'):
                        velocity = int(tokens[i + 1].split(':')[1])
                        i += 2
                    else:
                        i += 1
                    
                    track.append(mido.Message('note_on', channel=0, note=note, velocity=velocity, time=current_time_ticks))
                    current_time_ticks = 0
                except (ValueError, IndexError):
                    i+=1 # Skip malformed token
                    continue

            elif token.startswith('NOTE_OFF:'):
                try:
                    note = int(token.split(':')[1])
                    track.append(mido.Message('note_off', channel=0, note=note, velocity=0, time=current_time_ticks))
                    current_time_ticks = 0
                    i += 1
                except (ValueError, IndexError):
                    i+=1 
                    continue
            
            elif token.startswith('VELOCITY:'):
                i += 1
                continue
            else:
                i += 1
        
        mid.save(output_path)
        return mid

midi_tokenizer = MIDITokenizer(time_quantization=10)


# Generation Function

def generate_midi(
    model,
    vocab,
    id_to_token,
    meta_tokenizer,
    metadata: Dict[str, str],
    max_length: int = 2000,
    min_length: int = 1500,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device='cpu',
    use_cache: bool = True  # Enable KV cache for faster generation
):
    model.eval()
    
    metadata_tokens = meta_tokenizer.metadata_to_tokens(metadata, include_start=True)
    input_ids = [vocab.get(token, vocab.get('<UNK>', 1)) for token in metadata_tokens]
    generated_tokens = metadata_tokens.copy()
    
    # Get metadata token IDs to mask out after initial metadata
    metadata_token_ids = set()
    for token in ['START', 'GENRE:', 'PERIOD:', 'COMPOSER:']:
        if token in vocab:
            metadata_token_ids.add(vocab[token])
        # Also add all variants (e.g., GENRE:classical, GENRE:pop, etc.)
        for key in vocab.keys():
            if key.startswith(token):
                metadata_token_ids.add(vocab[key])
    
    # Track if we've already generated initial metadata
    initial_metadata_generated = True
    
    # KV cache for fast incremental generation
    past_key_values = None
    
    with torch.no_grad():
        # First pass: process initial sequence and cache KV
        if use_cache and len(input_ids) > 0:
            initial_input = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(initial_input, dtype=torch.long)
            
            # Forward pass with cache enabled
            result = model(
                initial_input, 
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True
            )
            
            # Handle return value (could be tuple or single value)
            if isinstance(result, tuple):
                logits, past_key_values = result
            else:
                logits = result
                past_key_values = None
            
            # Get logits for last position
            next_token_logits = logits[0, -1, :] / temperature
            
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                next_token_logits = filtered_logits
            
            # Mask out metadata tokens after initial metadata has been generated
            if initial_metadata_generated:
                for metadata_id in metadata_token_ids:
                    if metadata_id < len(next_token_logits):
                        next_token_logits[metadata_id] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Check for END token before appending
            current_length = len(generated_tokens)
            end_token_id = vocab.get('<END>', 3)
            
            if (next_token_id == end_token_id):
                should_stop = (current_length >= min_length) and (current_length >= max_length * 0.8)
                
                if should_stop:
                    next_token = id_to_token.get(next_token_id, '<UNK>')
                    generated_tokens.append(next_token)
                    input_ids.append(next_token_id)
                    print(f"   Generated {len(generated_tokens)} tokens (min: {min_length}, max: {max_length}) - stopping at <END> token")
                    return generated_tokens
                else:
                    # Ignore early END token - resample to get a different token
                    if current_length % 100 == 0:
                        print(f"   Ignoring early <END> token at {current_length} tokens (target: {max_length})")
                    # Resample excluding END token
                    filtered_logits = next_token_logits.clone()
                    filtered_logits[end_token_id] = float('-inf')
                    probs = torch.softmax(filtered_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            next_token = id_to_token.get(next_token_id, '<UNK>')
            generated_tokens.append(next_token)
            input_ids.append(next_token_id)
            
            # Continue to subsequent passes for more generation
        
        # Subsequent passes: incremental generation with KV cache
        while len(generated_tokens) < max_length:
            if use_cache and past_key_values is not None:
                try:
                    # Incremental decoding: only process the new token
                    # The token we want to process is the last one in input_ids
                    new_token_id = input_ids[-1]
                    new_input = torch.tensor([[new_token_id]], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(new_input, dtype=torch.long)
                    
                    # Forward pass with cached KV
                    result = model(
                        new_input,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # Handle return value
                    if isinstance(result, tuple):
                        logits, past_key_values = result
                    else:
                        logits = result
                        past_key_values = None
                    
                    # Get logits for the new token
                    next_token_logits = logits[0, -1, :] / temperature
                except Exception as e:
                    # Fallback to non-cached generation if cache fails
                    print(f"⚠️  KV cache failed during generation, falling back: {e}")
                    use_cache = False
                    past_key_values = None
                    # Continue to fallback branch below
                    continue
            else:
                # Fallback: process full sequence
                # Truncate if too long to avoid memory issues
                current_input = input_ids[-MODEL_CONFIG['max_seq_length']:] if len(input_ids) > MODEL_CONFIG['max_seq_length'] else input_ids
                input_tensor = torch.tensor([current_input], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
                
                output = model(input_tensor, attention_mask=attention_mask, use_cache=False)
                # Handle both return types: single logits or tuple (logits, cache)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                next_token_logits = logits[0, -1, :] / temperature
                # Reset cache if we truncated
                if len(input_ids) > MODEL_CONFIG['max_seq_length']:
                    past_key_values = None
            
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                next_token_logits = filtered_logits
            
            # Mask out metadata tokens after initial metadata has been generated
            if initial_metadata_generated:
                for metadata_id in metadata_token_ids:
                    if metadata_id < len(next_token_logits):
                        next_token_logits[metadata_id] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Check for END token before appending
            current_length = len(generated_tokens)
            end_token_id = vocab.get('<END>', 3)
            
            if (next_token_id == end_token_id):
                # Only stop if we've reached minimum length AND we're at least 80% of max_length
                should_stop = (current_length >= min_length) and (current_length >= max_length * 0.8)
                
                if should_stop:
                    next_token = id_to_token.get(next_token_id, '<UNK>')
                    generated_tokens.append(next_token)
                    input_ids.append(next_token_id)
                    print(f"   Generated {len(generated_tokens)} tokens (min: {min_length}, max: {max_length}) - stopping at <END> token")
                    return generated_tokens
                else:
                    # Ignore early END token - resample to get a different token
                    if current_length % 100 == 0:
                        print(f"   Ignoring early <END> token at {current_length} tokens (target: {max_length})")
                    # Resample excluding END token
                    filtered_logits = next_token_logits.clone()
                    filtered_logits[end_token_id] = float('-inf')
                    probs = torch.softmax(filtered_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            next_token = id_to_token.get(next_token_id, '<UNK>')
            generated_tokens.append(next_token)
            input_ids.append(next_token_id)
    
    print(f"   Finished generation: {len(generated_tokens)} tokens (max: {max_length})")
    return generated_tokens


# MIDI to WAV Conversion

USE_FLUIDSYNTH_CMD = False
FLUIDSYNTH_CMD = None

# Check custom path first, then system PATH
if os.path.exists(FLUIDSYNTH_EXE):
    FLUIDSYNTH_CMD = FLUIDSYNTH_EXE
    try:
        result = subprocess.run([FLUIDSYNTH_CMD, '--version'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            USE_FLUIDSYNTH_CMD = True
            print("✅ Found fluidsynth at custom path")
    except:
        pass

if not USE_FLUIDSYNTH_CMD:
    try:
        result = subprocess.run(['fluidsynth', '--version'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            USE_FLUIDSYNTH_CMD = True
            FLUIDSYNTH_CMD = 'fluidsynth'
            print("✅ Found fluidsynth in system PATH")
    except:
        pass

def midi_to_wav(midi_path: Path, soundfont_path: Path, wav_output_path: Path, sample_rate=44100):
    if USE_FLUIDSYNTH_CMD and FLUIDSYNTH_CMD:
        try:
            cmd = [
                FLUIDSYNTH_CMD,
                '-a', 'file',
                '-F', str(wav_output_path),
                '-r', str(sample_rate),
                str(soundfont_path),
                str(midi_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return True
            else:
                return False
        except:
            return False
    return False


# Main Generation Function for Gradio

def generate_audio(
    genre: str,
    composer: Optional[str],
    period: Optional[str],
    soundfont: str,
    temperature: float = 0.8,
    max_length: int = 2000,
    tempo: int = 120
):
    """
    Main function called by Gradio interface
    
    Args:
        genre: Music genre
        composer: Composer name (can be None)
        period: Music period (can be None)
        soundfont: Soundfont filename
        temperature: Generation temperature
        max_length: Maximum tokens to generate
        tempo: MIDI tempo in BPM
    
    Returns:
        Tuple of (MIDI file path, WAV file path, info message)
    """
    try:
        # Prepare metadata
        metadata = {'genre': genre}
        if composer and composer != "None":
            metadata['composer'] = composer
        if period and period != "None":
            metadata['music_period'] = period
        
        # Generate tokens
        print(f"🎹 Generating up to {max_length} tokens (min: {max(100, max_length // 10)})...")
        generated_tokens = generate_midi(
            model=model,
            vocab=vocab,
            id_to_token=id_to_token,
            meta_tokenizer=meta_tokenizer,
            metadata=metadata,
            max_length=max_length,
            min_length=max(100, max_length // 10),  # Minimum tokens: 10% of max_length or 100, whichever is larger
            temperature=temperature,
            top_k=50,
            device=device
        )
        
        print(f"✅ Generated {len(generated_tokens)} total tokens")
        
        # Debug: Count token types
        metadata_count = sum(1 for t in generated_tokens if t.startswith('START') or t.startswith('GENRE:') or t.startswith('PERIOD:') or t.startswith('COMPOSER:'))
        control_count = sum(1 for t in generated_tokens if t in ['<END>', '<PAD>', '<UNK>'])
        midi_count = len(generated_tokens) - metadata_count - control_count
        print(f"   Token breakdown: {metadata_count} metadata, {control_count} control, {midi_count} MIDI tokens")
        
        # Filter MIDI tokens
        midi_tokens = [
            token for token in generated_tokens 
            if not (token.startswith('START') or token.startswith('GENRE:') or 
                    token.startswith('PERIOD:') or token.startswith('COMPOSER:') or 
                    token in ['<END>', '<PAD>', '<UNK>'])
        ]
        
        print(f"   Filtered MIDI tokens: {len(midi_tokens)}")
        
        # Show first 20 tokens to debug
        if len(midi_tokens) < 100:
            print(f"   First 20 MIDI tokens: {midi_tokens[:20]}")
            print(f"   Last 20 MIDI tokens: {midi_tokens[-20:]}")
            print(f"   All generated tokens (first 50): {generated_tokens[:50]}")
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        composer_str = composer if (composer and composer != "None") else 'none'
        filename = f"generated_{genre}_{composer_str}_{timestamp}"
        midi_path = OUTPUT_DIR / f"{filename}.mid"
        wav_path = OUTPUT_DIR / f"{filename}.wav"
        
        # Convert to MIDI
        midi_tokenizer.tokens_to_midi(
            tokens=midi_tokens,
            output_path=midi_path,
            tempo=tempo,
            ticks_per_beat=500
        )
        
        # Convert to WAV if soundfont selected
        wav_path_str = None
        if soundfont and soundfont != "None":
            soundfont_path = SOUNDFONT_DIR / soundfont
            if soundfont_path.exists():
                success = midi_to_wav(
                    midi_path=midi_path,
                    soundfont_path=soundfont_path,
                    wav_output_path=wav_path,
                    sample_rate=44100
                )
                if success and wav_path.exists():
                    wav_path_str = str(wav_path)
        
        info_msg = f"✅ Generated {len(midi_tokens)} MIDI tokens"
        if wav_path_str:
            info_msg += f"\n✅ WAV file created using {soundfont}"
        
        return str(midi_path), wav_path_str, info_msg
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return None, None, error_msg


# Get Available Options

genres = ['classical', 'pop', 'jazz', 'soundtrack', 'rock', 'folk', 'ambient', 'ragtime', 'blues']
composers = [None, 'chopin', 'beethoven', 'bach', 'mozart', 'debussy', 'yiruma', 'einaudi', 'joplin', 'hisaishi', 'satie', 'schubert', 'schumann', 'liszt', 'rachmaninoff', 'tchaikovsky']
periods = [None, 'romantic', 'classical', 'baroque', 'contemporary', 'modern', 'impressionist']

# Get soundfont list
soundfont_files = sorted(list(SOUNDFONT_DIR.glob("*.sf2")) + list(SOUNDFONT_DIR.glob("*.SF2")))
soundfont_names = ["None"] + [sf.name for sf in soundfont_files]


# Gradio Interface

with gr.Blocks(title="Piano MIDI Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎹 Piano MIDI Generator
        
        Generate piano music using a Transformer model trained on the ARIA MIDI dataset.
        
        Select your preferences below and click **Generate** to create a MIDI file and WAV audio.
        """
    )
    
    with gr.Row():
        with gr.Column():
            genre_dropdown = gr.Dropdown(
                choices=genres,
                value='classical',
                label="Genre",
                info="Select the music genre"
            )
            
            composer_dropdown = gr.Dropdown(
                choices=["None"] + [c for c in composers if c is not None],
                value="None",
                label="Composer (Optional)",
                info="Select a composer or 'None' to skip",
                allow_custom_value=False
            )
            
            period_dropdown = gr.Dropdown(
                choices=["None"] + [p for p in periods if p is not None],
                value="None",
                label="Music Period (Optional)",
                info="Select a historical music period or 'None' to skip",
                allow_custom_value=False
            )
            
            soundfont_dropdown = gr.Dropdown(
                choices=soundfont_names,
                value="None" if len(soundfont_names) > 1 else soundfont_names[0] if soundfont_names else "None",
                label="Soundfont",
                info=f"Choose a soundfont for WAV generation ({len(soundfont_names)-1} available)"
            )
        
        with gr.Column():
            temperature_slider = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=0.8,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more deterministic"
            )
            
            max_length_slider = gr.Slider(
                minimum=100,
                maximum=2000,
                value=2000,
                step=100,
                label="Max Tokens",
                info="Maximum number of tokens to generate (100-2000)"
            )
            
            tempo_slider = gr.Slider(
                minimum=60,
                maximum=180,
                value=120,
                step=10,
                label="Tempo (BPM)",
                info="MIDI tempo in beats per minute"
            )
    
    generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            midi_output = gr.File(
                label="Generated MIDI File",
                type="filepath"
            )
            
            wav_output = gr.Audio(
                label="Generated WAV Audio",
                type="filepath"
            )
        
        with gr.Column():
            info_output = gr.Textbox(
                label="Generation Info",
                lines=5,
                interactive=False
            )
    
    # Connect interface
    generate_btn.click(
        fn=generate_audio,
        inputs=[
            genre_dropdown,
            composer_dropdown,
            period_dropdown,
            soundfont_dropdown,
            temperature_slider,
            max_length_slider,
            tempo_slider
        ],
        outputs=[midi_output, wav_output, info_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📝 Notes:
        - **MIDI files** are always generated and available for download
        - **WAV files** are only generated if a soundfont is selected
        - Generation may take 30-60 seconds depending on max tokens
        - Model checkpoint: Best validation loss model from training
        """
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎹 Starting Gradio Web App...")
    print("="*60)
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"✅ Checkpoint: Epoch {checkpoint['epoch']+1}, Step {checkpoint['step']:,}")
    print(f"✅ Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"✅ Available soundfonts: {len(soundfont_names)-1}")
    print(f"✅ Fluidsynth: {'Available' if USE_FLUIDSYNTH_CMD else 'Not available (WAV generation disabled)'}")
    print("="*60)
    print("\n🚀 Opening web interface...")
    print("   Access it at the URL shown below\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,      
        share=False             
    )
