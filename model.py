"""
Piano MIDI Generator - Transformer Model Architecture

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None, past_key_value=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: [seq_len, seq_len] - causal mask where True = masked (don't attend)
            key_padding_mask: [batch, seq_len] - True = pad (mask), False = valid
            past_key_value: Tuple of (past_key, past_value) for KV cache
                          Each is [batch, num_heads, past_len, head_dim] or None
        
        Returns:
            output: [batch, seq_len, d_model]
            present_key_value: Tuple of (key, value) for next step KV cache
        """
        # Check for NaN/Inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️  WARNING: NaN/Inf in TransformerBlock input!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # KV Cache support for inference
        # PyTorch's MultiheadAttention doesn't directly support KV cache,
        # so we need to manually handle it
        if past_key_value is not None:
            # Incremental decoding: use cached K, V
            past_key, past_value = past_key_value
            batch_size, seq_len, d_model = x.shape
            
            # Get Q, K, V projections from attention module
            # MultiheadAttention uses a combined projection, so we need to split it
            # For incremental decoding, x is [batch, 1, d_model]
            batch_size, seq_len, d_model = x.shape
            num_heads = self.attention.num_heads
            head_dim = d_model // num_heads
            
            # Use the combined in_proj_weight and split Q, K, V
            # MultiheadAttention stores: [in_proj_q | in_proj_k | in_proj_v]
            in_proj_weight = self.attention.in_proj_weight  # [3*d_model, d_model]
            in_proj_bias = self.attention.in_proj_bias  # [3*d_model] or None
            
            # Split weights
            q_weight = in_proj_weight[:d_model, :]  # [d_model, d_model]
            k_weight = in_proj_weight[d_model:2*d_model, :]  # [d_model, d_model]
            v_weight = in_proj_weight[2*d_model:, :]  # [d_model, d_model]
            
            # Split bias if exists
            if in_proj_bias is not None:
                q_bias = in_proj_bias[:d_model]
                k_bias = in_proj_bias[d_model:2*d_model]
                v_bias = in_proj_bias[2*d_model:]
            else:
                q_bias = k_bias = v_bias = None
            
            # Compute Q, K, V
            q = F.linear(x, q_weight, q_bias)  # [batch, seq_len, d_model]
            k = F.linear(x, k_weight, k_bias)  # [batch, seq_len, d_model]
            v = F.linear(x, v_weight, v_bias)  # [batch, seq_len, d_model]
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            
            # Concatenate with cached K, V
            k = torch.cat([past_key, k], dim=2)  # [batch, num_heads, past_len + seq_len, head_dim]
            v = torch.cat([past_value, v], dim=2)  # [batch, num_heads, past_len + seq_len, head_dim]
            
            # Store present K, V for next step
            present_key_value = (k, v)
            
            # Compute attention manually
            # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
            scale = (head_dim) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, past_len + seq_len]
            
            # Apply causal mask (only attend to previous tokens + current)
            # For incremental decoding, we only need to mask future tokens (none in this case)
            # But we need to ensure we don't attend beyond the sequence
            total_len = k.size(2)
            if attn_mask is not None:
                # For incremental decoding, attn_mask should be None or handle differently
                pass
            
            # Apply softmax
            attn = F.softmax(attn, dim=-1)
            # Use F.dropout instead of self.attention.dropout (which is a float, not a module)
            attn = F.dropout(attn, p=self.attention.dropout, training=self.training)
            
            # Apply attention to values
            attn_out = torch.matmul(attn, v)  # [batch, num_heads, seq_len, head_dim]
            
            # Reshape back
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # [batch, seq_len, d_model]
            
            # Apply output projection
            attn_out = self.attention.out_proj(attn_out)
        else:
            # Full sequence: no cache - use standard attention
            attn_out, _ = self.attention(
                x, x, x, 
                attn_mask=attn_mask,  # Causal mask: [seq_len, seq_len], True = masked
                key_padding_mask=key_padding_mask  # Padding mask: [batch, seq_len] or None
            )
            
            # For caching, we need to extract K, V from the attention computation
            # We'll compute them manually for the full sequence
            batch_size, seq_len, d_model = x.shape
            num_heads = self.attention.num_heads
            head_dim = d_model // num_heads
            
            # Use the combined in_proj_weight and split K, V
            in_proj_weight = self.attention.in_proj_weight  # [3*d_model, d_model]
            in_proj_bias = self.attention.in_proj_bias  # [3*d_model] or None
            
            # Split weights for K, V
            k_weight = in_proj_weight[d_model:2*d_model, :]  # [d_model, d_model]
            v_weight = in_proj_weight[2*d_model:, :]  # [d_model, d_model]
            
            # Split bias if exists
            if in_proj_bias is not None:
                k_bias = in_proj_bias[d_model:2*d_model]
                v_bias = in_proj_bias[2*d_model:]
            else:
                k_bias = v_bias = None
            
            # Compute K, V
            k = F.linear(x, k_weight, k_bias)  # [batch, seq_len, d_model]
            v = F.linear(x, v_weight, v_bias)  # [batch, seq_len, d_model]
            
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
            
            present_key_value = (k, v)
        
        # Check for NaN/Inf after attention
        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            print("⚠️  WARNING: NaN/Inf in attention output!")
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Residual + normalization with numerical stability
        x = self.norm1(x + self.dropout(attn_out))
        
        # Check for NaN/Inf after norm1
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️  WARNING: NaN/Inf after norm1!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        
        # Check for NaN/Inf after FF
        if torch.isnan(ff_out).any() or torch.isinf(ff_out).any():
            print("⚠️  WARNING: NaN/Inf in feed-forward output!")
            ff_out = torch.nan_to_num(ff_out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.norm2(x + ff_out)
        
        # Final check
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️  WARNING: NaN/Inf after norm2!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x, present_key_value


class PianoMIDIGenerator(nn.Module):
    """
    GPT-style decoder-only transformer for conditional MIDI generation
    
    Input: [metadata_tokens] + [midi_tokens]
    Output: Next token predictions for all positions
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        
        # Token embeddings
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config['d_model'], config['max_seq_length'])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config['d_model'],
                config['n_heads'],
                config['d_ff'],
                config['dropout']
            )
            for _ in range(config['n_layers'])
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with better numerical stability"""
        if isinstance(module, nn.Linear):
            # Use Xavier uniform for better stability
            torch.nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal initialization with smaller std for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm already has proper initialization, but ensure eps is set
            if hasattr(module, 'eps'):
                module.eps = 1e-6  # Ensure numerical stability
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        """
        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] 1 = attend, 0 = mask
            past_key_values: Tuple of tuples, one per layer, each containing (past_key, past_value)
            use_cache: Whether to return past_key_values for next step
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            past_key_values: Tuple of (key, value) tuples for each layer (if use_cache=True)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # Check for NaN/Inf in embeddings
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️  WARNING: NaN/Inf in embeddings!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        x = self.pos_encoding(x)
        
        # Check for NaN/Inf after positional encoding
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️  WARNING: NaN/Inf after positional encoding!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Create causal mask (only needed for full sequence, not incremental decoding)
        # For incremental decoding with past_key_values, we don't need a causal mask
        attn_mask = None
        key_padding_mask = None
        
        if past_key_values is None:
            # Full sequence: create causal mask
            causal_mask_base = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
            
            # If attention_mask provided, handle padding
            if attention_mask is not None:
                # attention_mask: [batch, seq_len] where 1 = valid, 0 = pad
                # Use key_padding_mask for padding (better for MultiheadAttention)
                key_padding_mask = (attention_mask == 0)  # True = pad (mask), False = valid
                
                # Causal mask: [seq_len, seq_len] - will be broadcasted
                attn_mask = causal_mask_base
            else:
                # Just causal mask, no padding
                attn_mask = causal_mask_base  # [seq_len, seq_len]
        else:
            # Incremental decoding: no causal mask needed, but handle padding if provided
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)  # True = pad (mask), False = valid
        
        # Transformer blocks with KV cache support
        present_key_values = () if use_cache else None
        if past_key_values is not None:
            past_key_values = list(past_key_values)
        else:
            past_key_values = [None] * len(self.blocks)
        
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if i < len(past_key_values) else None
            x, present_key_value = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, past_key_value=past_key_value)
            
            if use_cache:
                present_key_values = present_key_values + (present_key_value,)
            
            # Check for NaN/Inf after each block
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️  WARNING: NaN/Inf after block {i}!")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Output projection
        x = self.ln_f(x)
        
        # Check for NaN/Inf before head
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️  WARNING: NaN/Inf before output head!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        logits = self.head(x)  # [batch, seq_len, vocab_size]
        
        # Final check on logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("⚠️  WARNING: NaN/Inf in final logits!")
            # Clamp logits to reasonable range before returning
            logits = torch.clamp(logits, min=-50.0, max=50.0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        
        if use_cache:
            return logits, present_key_values
        return logits
    
    def generate(self, input_ids, vocab, max_length=2048, temperature=1.0, top_k=50, top_p=0.9):
        """
        Autoregressive generation

        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Forward pass
                logits = self.forward(generated)[:, -1, :]  # [batch, vocab_size]
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if END token
                if next_token.item() == vocab.get('<END>', 3):
                    break
        
        return generated

