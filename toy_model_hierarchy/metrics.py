"""
metrics.py
──────────
All analysis / measurement functions, decoupled from the training loop.

Public API
──────────
evaluate(model, test_set, device)
    → float  (masked cross-entropy loss)

calculate_layerwise_hydra(model, test_set, device, rejection_rank=None)
    → float  (scalar: max entry of the layerwise Hydra matrix)

calculate_headwise_hydra(model, test_set, device)
    → HydraResult  (namedtuple: .matrix (L,H), .full_tensor (L,H,L))

measure_induction_score(model, test_set, device)
    → float  (max per-head induction score, Olsson et al. 2022)

function_vector_analysis(model, test_set, seg_len, device)
    → float  (max mean layer patching effect)

Each function is stateless w.r.t. the model (eval mode in / train mode out,
no gradient side-effects).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Named return type for headwise Hydra
# ─────────────────────────────────────────────────────────────────────────────

HydraResult = namedtuple("HydraResult", ["matrix", "full_tensor"])
"""
matrix      : np.ndarray (n_layers, n_heads)
              matrix[i, h] = max_{j > i} hydra[i, h, j]
              Peak downstream compensation when head (i, h) is ablated.
full_tensor : np.ndarray (n_layers, n_heads, n_layers)
              full_tensor[i, h, j] = hydra[i, h, j]
              Full 3-D Hydra tensor for detailed analysis.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Loss evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:    nn.Module,
    test_set: dict,
    device:   torch.device,
) -> float:
    """
    Masked cross-entropy loss on a test set, excluding [QUERY] cue positions.

    Parameters
    ----------
    model    : Transformer (called in eval mode)
    test_set : dict with 'tokens' (N, T) and 'loss_mask' (N, T-1)
    device   : torch device

    Returns
    -------
    Mean masked loss as a float.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    tokens    = test_set["tokens"]
    loss_mask = test_set["loss_mask"]

    x    = torch.tensor(tokens[:, :-1], dtype=torch.long, device=device)
    y    = torch.tensor(tokens[:, 1:],  dtype=torch.long, device=device)
    mask = torch.tensor(loss_mask,      dtype=torch.bool,  device=device)

    logits = model(x)                                   # (N, T-1, V)
    N, T, V = logits.shape

    per_token = criterion(
        logits.reshape(N * T, V), y.reshape(N * T)
    ).reshape(N, T)

    loss = (per_token * mask).sum() / mask.sum()

    model.train()
    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Layerwise Hydra
# ─────────────────────────────────────────────────────────────────────────────

def calculate_layerwise_hydra(
    model:          nn.Module,
    test_set:       dict,
    device:         torch.device,
    rejection_rank: Optional[int] = None,
) -> float:
    """
    Layer-level Hydra effect measurement.

    For each upstream layer i, ablates the entire layer, then measures the
    change in predictive influence (logit on correct token) at every
    downstream layer j > i.

    hydra_matrix[i, j] = E[ Δ^(j)_ablated(i) − Δ^(j)_full ]

    Returns the scalar max over all (i, j) pairs — the peak compensation
    signal in the network.

    Parameters
    ----------
    rejection_rank : if set, only averages over examples where the correct
                     token is within the top-k predictions (filters clean
                     predictions for a cleaner signal).
    """
    model.eval()
    x = torch.tensor(test_set["tokens"][:, :-1], device=device)
    y = torch.tensor(test_set["tokens"][:, -1],  device=device)

    n_layers = len(model.layers)
    W_U      = model.unembed.weight                     # (V, D)

    # ── Hooks to capture layer outputs ───────────────────────────────────────
    activations: dict[int, torch.Tensor] = {}

    def _save_hook(idx: int):
        def hook(module, inp, out):
            activations[idx] = out
        return hook

    handles = [
        model.layers[i].register_forward_hook(_save_hook(i))
        for i in range(n_layers)
    ]

    def _project_to_logit(h: torch.Tensor) -> torch.Tensor:
        """h : (N, T, D) → (N,) logit on correct token at last position."""
        return (h[:, -1, :] * W_U[y]).sum(dim=-1)

    # ── Full forward pass ─────────────────────────────────────────────────────
    pred_logits = model(x, ablate_layers=[])[:, -1, :]

    # Optional: restrict to examples where prediction is within top-k
    mask = torch.ones(y.shape[0], dtype=torch.bool, device=device)
    if rejection_rank is not None:
        sorted_idx = torch.argsort(pred_logits, descending=True, dim=-1)
        ranks      = (sorted_idx == y.unsqueeze(1)).nonzero()[:, 1]
        mask       = ranks < rejection_rank

    full_layer_logits = {j: _project_to_logit(activations[j])
                         for j in range(n_layers)}

    # ── Hydra matrix ──────────────────────────────────────────────────────────
    hydra_matrix = torch.zeros(n_layers, n_layers, device=device)

    for i in range(n_layers - 1):
        activations.clear()
        model(x, ablate_layers=[i])

        for j in range(i + 1, n_layers):
            delta = _project_to_logit(activations[j]) - full_layer_logits[j]
            hydra_matrix[i, j] = (delta * mask.float()).mean()

    for h in handles:
        h.remove()

    model.train()
    return torch.max(hydra_matrix).item()


# ─────────────────────────────────────────────────────────────────────────────
# Headwise Hydra
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calculate_headwise_hydra(
    model:    nn.Module,
    test_set: dict,
    device:   torch.device,
) -> HydraResult:
    """
    Head-level Hydra effect measurement.

    For every attention head (layer_i, head_h), ablates that head by zeroing
    its value output before o_proj, then measures the change in predictive
    influence at each downstream layer j > i.

    hydra[i, h, j] = E[ Δ^(j)_ablated(i,h) − Δ^(j)_full ]

    Positive = layer j's predictive influence increased after ablating head h
    in layer i — the definition of the Hydra effect (McGrath et al. 2023).

    Returns
    -------
    HydraResult with:
        matrix      : (n_layers, n_heads) — peak downstream compensation
        full_tensor : (n_layers, n_heads, n_layers) — full 3-D Hydra tensor
    """
    model.eval()
    tokens = test_set["tokens"]
    x = torch.tensor(tokens[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(tokens[:, -1],  dtype=torch.long, device=device)

    n_layers = len(model.layers)
    n_heads  = model.layers[0].attn.nhead
    W_U      = model.unembed.weight                     # (V, D)

    def _project_last(h: torch.Tensor) -> torch.Tensor:
        """h : (N, T, D) → (N,) predictive influence at last position."""
        return (h[:, -1, :] * W_U[y]).sum(dim=-1)

    # ── Hooks ─────────────────────────────────────────────────────────────────
    activations: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(module, inp, out):
            activations[idx] = out.detach()
        return hook

    handles = [
        model.layers[j].register_forward_hook(_make_hook(j))
        for j in range(n_layers)
    ]

    # ── Full forward pass ─────────────────────────────────────────────────────
    model(x)
    full_influence = {j: _project_last(activations[j]) for j in range(n_layers)}

    # ── Head ablation loop ────────────────────────────────────────────────────
    full_tensor = np.zeros((n_layers, n_heads, n_layers), dtype=np.float32)

    for i in range(n_layers):
        for h in range(n_heads):
            activations.clear()
            model(x, ablate_heads={i: [h]})
            for j in range(i + 1, n_layers):
                delta = _project_last(activations[j]) - full_influence[j]
                full_tensor[i, h, j] = delta.mean().item()

    for handle in handles:
        handle.remove()

    # ── Summary matrix: peak downstream compensation per (layer, head) ────────
    hydra_matrix = np.zeros((n_layers, n_heads), dtype=np.float32)
    for i in range(n_layers):
        for h in range(n_heads):
            downstream = full_tensor[i, h, i + 1:]
            if downstream.size > 0:
                hydra_matrix[i, h] = downstream.max()

    model.train()
    return HydraResult(matrix=hydra_matrix, full_tensor=full_tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Induction head score
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_induction_score(
    model:       nn.Module,
    test_set:    dict,         # kept for API compatibility; not used
    device:      torch.device,
    n_samples:   int = 64,
    seq_len:     int = 64,
) -> float:
    """
    Standard attention-based induction head score (Olsson et al. 2022).

    Uses internally generated *repeated random sequences* — independent of the
    DGP test set — so the score measures a pure circuit property of the model
    rather than a statistical artifact of the training distribution.

    Stimulus construction
    ─────────────────────
    Each sequence has the form:

        [prefix ...][A][B][middle ...][A][?]

    where A and B are single random tokens and prefix / middle are random
    filler.  The induction target for the final query (A) is position
    prev(A) + 1, which holds token B.  A perfect induction head places all
    its attention weight at that position.

    Concretely, we build sequences of the form:

        x = [r_0, r_1, ..., r_{L-1}, A, r_L, ..., r_{2L-2}, A]

    where L = seq_len // 2 and each r_i is a random token.  The repeated A
    appears at positions L-1 and 2L-2 (0-indexed), so the induction target
    for the final A is position L (the token immediately following the first A).

    Score
    ─────
    For head (l, h) and query position t = 2L-2 (the second A):

        score(l, h) = A^(l,h)[t, L]

    i.e. the attention weight the head places on the induction target.
    We average over all n_samples sequences and return the max over heads.

    Returns
    -------
    float : max over all (layer, head) pairs of their per-head induction score.
            Ranges in [0, 1].  A random head scores ≈ 1/T; a perfect induction
            head scores 1.
    """
    model.eval()

    vocab_size = model.embed.num_embeddings
    L          = seq_len // 2       # half-length; A appears at L-1 and 2L-2
    T          = 2 * L - 1          # total input length (last A is the query)

    n_layers = len(model.layers)
    n_heads  = model.layers[0].attn.nhead

    # ── Build stimulus batch ───────────────────────────────────────────────────
    # Shape: (n_samples, T)
    tokens = torch.randint(0, vocab_size, (n_samples, T), device=device)

    # Choose A for each sample (avoid special tokens if vocab_size > 2)
    A = torch.randint(0, vocab_size, (n_samples,), device=device)

    # Place A at positions L-1 (first occurrence) and 2L-2 = T-1 (second, query)
    tokens[:, L - 1] = A
    tokens[:, T - 1] = A

    # The induction target: position L (token immediately after first A)
    induction_target_pos = L   # the position we expect each induction head to attend to

    # ── Single forward pass collecting per-layer attn weights ─────────────────
    all_attn: list[torch.Tensor] = [None] * n_layers

    h = model.embed(tokens)
    for l_idx, layer in enumerate(model.layers):
        h_ln = layer.ln1(h)
        attn_out, attn_probs = layer.attn(h_ln, return_attn=True)  # (B,H,T,T)
        all_attn[l_idx] = attn_probs.detach()
        h = h + attn_out
        h = h + layer.ff(layer.ln2(h))

    # ── Score: attention weight at (query=T-1, key=induction_target_pos) ──────
    # Shape of each all_attn[l]: (n_samples, n_heads, T, T)
    head_scores = torch.stack([
        all_attn[l][:, :, T - 1, induction_target_pos]   # (n_samples, n_heads)
        for l in range(n_layers)
    ])                                                      # (n_layers, n_samples, n_heads)

    # Mean over samples → (n_layers, n_heads); take max over all (l, h)
    mean_scores = head_scores.mean(dim=1)   # (n_layers, n_heads)

    model.train()
    return mean_scores.max().item()

# ─────────────────────────────────────────────────────────────────────────────
# Function vector analysis
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def function_vector_analysis(
    model:    nn.Module,
    test_set: dict,
    seg_len:  int,
    device:   torch.device,
) -> float:
    """
    Function vector analysis via activation patching (Todd et al. 2023 style).

    For each layer, computes the mean logit boost on the correct query token
    when the attention output cached on the *source* pass (all segments except
    the last) is added into the *target* pass (last segment only) at that layer.

    This identifies which layer's attention outputs act as "function vectors" —
    compressed representations that, when injected into a minimal context,
    steer the model towards the correct answer.

    Protocol
    ────────
    Source context : tokens[:, :-seg_len-1]  (all evidence segments)
    Target context : tokens[:, -seg_len:-1]  (final segment only, no context)
    Gold token     : tokens[:, -1]

    For each layer, we:
      1. Cache attn_out at the final source position.
      2. Run the target context cleanly (no patch) → baseline logit.
      3. Add the cached attn_out to that layer's target pass → patched logit.
      4. Record delta = patched_logit − clean_logit.

    Returns
    -------
    Max mean layer patching effect across layers (scalar).
    """
    model.eval()

    source = test_set["tokens"][:, :-seg_len - 1]
    target = test_set["tokens"][:, -seg_len:-1]
    gold   = test_set["tokens"][:, -1]

    n_layers = len(model.layers)
    all_effects: list[list[float]] = []

    for s, t, g in zip(source, target, gold):
        src = torch.tensor(s, dtype=torch.long, device=device).unsqueeze(0)
        tgt = torch.tensor(t, dtype=torch.long, device=device).unsqueeze(0)
        g_t = torch.tensor(g, dtype=torch.long, device=device)

        # ── Source pass: cache attention outputs at final position ────────────
        cached_attn: dict[int, torch.Tensor] = {}
        h = model.embed(src)
        for idx, layer in enumerate(model.layers):
            attn_out           = layer.attn(layer.ln1(h))
            cached_attn[idx]   = attn_out[0, -1].clone()   # (D,)
            h = h + attn_out
            h = h + layer.ff(layer.ln2(h))

        # ── Clean target pass ─────────────────────────────────────────────────
        h_clean = model.embed(tgt)
        for layer in model.layers:
            h_clean = layer(h_clean)
        clean_logit = model.unembed(model.final_ln(h_clean))[0, -1, g_t].item()

        # ── Patched passes: one per layer ─────────────────────────────────────
        layer_effects: list[float] = []
        for patch_layer in range(n_layers):
            h = model.embed(tgt)
            for idx, layer in enumerate(model.layers):
                attn_out = layer.attn(layer.ln1(h))
                if idx == patch_layer:
                    attn_out = attn_out.clone()
                    attn_out[0, -1] = attn_out[0, -1] + cached_attn[idx]
                h = h + attn_out
                h = h + layer.ff(layer.ln2(h))
            patched_logit = model.unembed(model.final_ln(h))[0, -1, g_t].item()
            layer_effects.append(patched_logit - clean_logit)

        all_effects.append(layer_effects)

    effects_tensor = torch.tensor(all_effects, dtype=torch.float32)  # (N, L)
    model.train()
    return effects_tensor.mean(dim=0).max().item()

def measure_directional_concavity(
    model:      nn.Module,
    test_set:   dict,
    device:     torch.device,
    max_samples: int = 32,
) -> dict:
    """
    Empirical test of Assumption 6 (Directional Concavity of Readout).

    Assumption 6 states that the scalar readout
        ϕ(u) = ⟨g(u), w_t⟩
    is locally concave along directions induced by *redundant latent estimators*
    — i.e., for any unit contribution u₁ that is an independent estimator of
    Z⁽⁰⁾ already partially captured by v:
        E[ u₁ᵀ H_ϕ(v) u₁ ] ≤ 0

    This is a strictly *directional* claim, not a claim that H_ϕ is globally
    negative semi-definite (NSD).  The correct operationalisation is therefore:

        1. For each sample, run the full forward pass and collect:
             v  = final residual stream at the query position (the "current" rep)
             u₁ = attention output of an early layer at that position
                  (a partial posterior estimator, the "redundant" direction)
        2. Compute H_ϕ(v) via exact second-order autodiff.
        3. Evaluate the directional curvature  c = u₁ᵀ H_ϕ(v) u₁.
        4. Report the mean directional curvature and the proportion of
           samples where c < 0 (supporting the assumption).

    Parameters
    ----------
    model       : trained Transformer (called in eval mode)
    test_set    : dict with 'tokens' (N, T)
    device      : torch device
    max_samples : cap on number of sequences (Hessian is O(D²) memory)

    Returns
    -------
    dict with:
        directional_curvatures  : np.ndarray (n_samples,)
                                  c_i = u₁ᵀ H_ϕ(v_i) u₁ per sample
        mean_directional_curv   : float  — primary test statistic (≤ 0 supports A6)
        prop_negative_direction : float  — fraction of samples with c < 0
    """
    model.eval()

    n = min(max_samples, len(test_set["tokens"]))
    # Use tokens[:, :-1] as input (standard autoregressive setup);
    # the query position is the final position of this input.
    tokens_in  = torch.tensor(
        test_set["tokens"][:n, :-1], dtype=torch.long, device=device
    )                                                        # (n, T-1)
    target_ids = torch.tensor(
        test_set["tokens"][:n, -1],  dtype=torch.long, device=device
    )                                                        # (n,)

    # ── Collect v (final residual stream) and u₁ (early-layer attn output) ───
    # We pick layer 0's attention output as the "early redundant estimator" u₁.
    # It is the first partial posterior update in the residual stream — exactly
    # the kind of independent estimator Theorem 3 / Assumption 6 refers to.
    early_layer_idx = 0

    v_list:  list[torch.Tensor] = []   # (D,) final residual at query pos
    u1_list: list[torch.Tensor] = []   # (D,) early attn output at query pos

    with torch.no_grad():
        for i in range(n):
            x = tokens_in[i].unsqueeze(0)         # (1, T-1)
            h = model.embed(x)

            for l_idx, layer in enumerate(model.layers):
                h_ln    = layer.ln1(h)
                attn_out = layer.attn(h_ln)
                if l_idx == early_layer_idx:
                    u1_list.append(attn_out[0, -1].detach().clone())   # (D,)
                h = h + attn_out
                h = h + layer.ff(layer.ln2(h))

            # v = final residual stream at query position, after final LN
            # v_list.append(model.final_ln(h)[0, -1].detach().clone())   # (D,)
            v_list.append(h[0, -1].detach().clone())   # (D,)

    # ── Hessian computation and directional curvature ─────────────────────────
    directional_curvatures: list[float] = []

    for i in range(n):
        v_i  = v_list[i].requires_grad_(True)    # (D,) — leaf for Hessian
        u1_i = u1_list[i]                         # (D,) — direction vector

        target = target_ids[i]

        def readout(v_in: torch.Tensor) -> torch.Tensor:
            """ϕ(v) = ⟨g(v), w_t⟩  using the model's own unembedding."""
            # g is implicitly the identity here — the final LN has already been
            # applied when building v_i, so we go straight to unembed.
            return model.unembed(model.final_ln(v_in.unsqueeze(0)))[0, target]
            #return model.unembed.weight[target] @ v_in   # scalar

        # Exact Hessian via double autodiff: (D, D)
        H = torch.autograd.functional.hessian(readout, v_i)   # (D, D)

        # Directional curvature: u₁ᵀ H u₁  (the quantity A6 bounds)
        direc_curv = (u1_i @ H @ u1_i).item()
        directional_curvatures.append(direc_curv)

    dc_arr    = np.array(directional_curvatures)

    model.train()
    return {
        "directional_curvatures":   dc_arr,
        "mean_directional_curv":    float(dc_arr.mean()),
        "prop_negative_direction":  float((dc_arr < 0).mean())
    }
