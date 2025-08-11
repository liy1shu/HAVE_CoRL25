import torch
import math
import torch.nn as nn
from flash_attn.modules.mha import FlashCrossAttention
from flash_attn.bert_padding import unpad_input, pad_input

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', torch.zeros(max_len, d_model))
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]


class FlashCrossMHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.head_dim = d_model // nhead
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.flash_cross_attn = FlashCrossAttention(softmax_scale=self.scaling, attention_dropout=dropout)
        
    def forward(self, q, kv, key_padding_mask=None):
        batch_size, tgt_len, _ = q.shape
        src_len = kv.shape[1]
        
        # Project queries, keys, and values
        q = self.q_proj(q).view(batch_size, tgt_len, self.nhead, self.head_dim)
        k = self.k_proj(kv).view(batch_size, src_len, self.nhead, self.head_dim)
        v = self.v_proj(kv).view(batch_size, src_len, self.nhead, self.head_dim)
        
        # Handle padding if present
        if key_padding_mask is not None:
            # Unpad the inputs
            q_unpad, q_indices, q_cu_seqlens, q_max_seqlen, q_used_seqlens = unpad_input(q, ~key_padding_mask[:, :tgt_len])
            kv_unpad, kv_indices, kv_cu_seqlens, kv_max_seqlen, kv_used_seqlens = unpad_input(kv, ~key_padding_mask)
            
            # Reshape k and v and combine them
            k_unpad = self.k_proj(kv_unpad)
            v_unpad = self.v_proj(kv_unpad)
            kv_unpad = torch.stack([k_unpad, v_unpad], dim=2)  # Shape: [num_tokens, 2, nhead, head_dim]
            
            # Convert to half precision as required by flash attention
            q_unpad = q_unpad.half()
            kv_unpad = kv_unpad.half()
            
            # Call flash attention with unpadded inputs
            context = self.flash_cross_attn(
                q_unpad.view(-1, self.nhead, self.head_dim),  # [num_tokens, nhead, head_dim]
                kv_unpad.view(-1, 2, self.nhead, self.head_dim),  # [num_tokens, 2, nhead, head_dim]
                cu_seqlens=q_cu_seqlens,
                max_seqlen=q_max_seqlen,
                cu_seqlens_k=kv_cu_seqlens,
                max_seqlen_k=kv_max_seqlen
            )
            
            # Pad the output back
            context = pad_input(context, q_indices, batch_size, tgt_len)
        else:
            # If no padding, reshape q and combine k,v
            q = q.view(batch_size, tgt_len, self.nhead, self.head_dim).half()
            kv = torch.stack([k, v], dim=2).half()  # [batch_size, src_len, 2, nhead, head_dim]
            
            context = self.flash_cross_attn(
                q,  # [batch_size, tgt_len, nhead, head_dim]
                kv  # [batch_size, src_len, 2, nhead, head_dim]
            )
        
        # Project output
        output = self.out_proj(context.float().view(batch_size, tgt_len, self.d_model))
        return output


class SequentialEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        # pnp.PN2Encoder(3, d_model)
        # self.motion_attention_
        # self.motion_decoder = nn.Linear(d_model, 1)
        self.d_model = d_model
        
        # Self-attention layers for history processing
        self.history_layers = nn.ModuleList([
            FlashCrossMHA(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.history_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.history_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, action_results, padding_mask=None):
        """
        # query: [batch_size, 1, d_model]  # Your query embedding
        history: [batch_size, seq_len, d_model]  # Your history embeddings
        padding_mask: [batch_size, seq_len]  # Mask for padding in history
        """
        # Process history with self-attention
        x_history = action_results
        for layer, norm, ffn in zip(self.history_layers, self.history_norms, self.history_ffns):
            # Self attention on history
            attended = layer(x_history, x_history, padding_mask)
            x_history = x_history + attended
            x_history = norm(x_history)
            
            # FFN
            x_history = x_history + ffn(x_history)
            x_history = norm(x_history)
        
        # Process query with cross-attention to processed history
        # history_scores = self.history_scorer(x_history)
        # return history_scores
            
        return x_history