import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

def precompute_freqs(dim: int, max_seq_len: int, base: int = 10000):
    """
    Precompute RoPE cos/sin tables.

    Standard convention (Su et al. 2023): pair dimension i with dimension
    i + dim//2, i.e. the first half and second half of the head dimension
    are rotated together — NOT adjacent pairs (d0,d1), (d2,d3).

    Returns cos, sin of shape (max_seq_len, dim // 2).
    """
    half_dim = dim // 2
    freqs    = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
    t        = torch.arange(max_seq_len).float()
    freqs    = torch.outer(t, freqs)   # (T, dim//2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x of shape (B, T, H, D).

    Fix: split along the last dimension into two halves (x1, x2) of size D//2,
    rather than interleaving adjacent pairs.  This matches the standard
    formulation where dimension i is paired with dimension i + D//2.
    """
    B, T, H, D = x.shape
    x1 = x[..., : D // 2]                          # (B, T, H, D//2)
    x2 = x[..., D // 2 :]                          # (B, T, H, D//2)

    cos = cos[:T].unsqueeze(0).unsqueeze(2)         # (1, T, 1, D//2)
    sin = sin[:T].unsqueeze(0).unsqueeze(2)         # (1, T, 1, D//2)

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)          # (B, T, H, D)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head attention with RoPE and causal mask
# ─────────────────────────────────────────────────────────────────────────────

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, max_seq_len: int = 512):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.nhead    = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_freqs(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        # Causal mask: upper-triangular portion is -inf, lower (including
        # diagonal) is 0.  Pre-allocated to max_seq_len; sliced at runtime.
        # FIX: this was missing entirely in the original implementation.
        causal_mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x:             torch.Tensor,
        ablate_heads:  list[int] = [],
        return_attn:   bool = False,
    ):
        """
        Parameters
        ----------
        x            : (B, T, D)
        ablate_heads : list of head indices whose value outputs are zeroed
                       before o_proj.  All other heads are unaffected.
                       This implements a clean additive ablation: the head's
                       contribution to the residual stream is removed while
                       every other head's contribution is preserved exactly.
        return_attn  : if True, return (output, attn_probs) where
                       attn_probs has shape (B, H, T, T).  Used by
                       measure_induction_score in metrics.py.

        Returns
        -------
        output : (B, T, D)
        attn_probs : (B, H, T, T)  only if return_attn=True
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim)

        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)

        # Attention scores: (B, H, T, T)
        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)

        # Apply causal mask: token t cannot attend to positions s > t
        attn_scores = attn_scores + self.causal_mask[:T, :T]

        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.einsum("bhts,bshd->bthd", attn_probs, v)  # (B, T, H, D_head)

        # Head ablation: zero the value output of each ablated head before
        # o_proj.  Because o_proj is linear, zeroing head h's slot is
        # equivalent to removing its additive contribution to the residual
        # stream entirely, matching the paper's definition of Δ^(l)_{l'}.
        if ablate_heads:
            out = out.clone()
            for h in ablate_heads:
                out[:, :, h, :] = 0.0

        out = out.reshape(B, T, D)
        result = self.o_proj(out)
        if return_attn:
            return result, attn_probs          # (B, T, D), (B, H, T, T)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformer layer
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_ratio: int = 4):
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, nhead)

        # FIX: use ffn_ratio * d_model as the hidden size rather than a
        # hardcoded 256, which was equivalent to a 1× ratio for d_model=256.
        ffn_hidden = ffn_ratio * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),                   # GELU is standard for modern Transformers
            nn.Linear(ffn_hidden, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x:            torch.Tensor,
        ablate_heads: list[int] = [],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, D)
        ablate_heads : head indices to zero inside this layer's attention.
                       Passed straight through to RoPEMultiheadAttention.
        """
        x = x + self.attn(self.ln1(x), ablate_heads=ablate_heads)
        x = x + self.ff(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full Transformer
# ─────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 128,
        n_layers:   int = 4,
        nhead:      int = 4,
        ffn_ratio:  int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, ffn_ratio)
            for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x:             torch.Tensor,
        ablate_layers: list[int]        = [],
        ablate_heads:  dict[int, list[int]] = {},
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x             : (B, T)  long tensor of token ids
        ablate_layers : list of layer indices to skip entirely (layer-level ablation).
        ablate_heads  : dict mapping layer_index → list of head indices to zero
                        within that layer's attention module (head-level ablation).
                        Example: {2: [0, 3], 5: [1]} zeros heads 0 and 3 in
                        layer 2, and head 1 in layer 5.
                        Layer-level and head-level ablation can be combined,
                        though typically only one is used at a time.

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        x = self.embed(x)
        for i, layer in enumerate(self.layers):
            if i in ablate_layers:
                continue
            heads_to_ablate = ablate_heads.get(i, [])
            x = layer(x, ablate_heads=heads_to_ablate)
        x = self.final_ln(x)
        return self.unembed(x)
