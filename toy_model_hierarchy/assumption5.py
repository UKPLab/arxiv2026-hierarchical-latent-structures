"""
Empirical verification of Assumption 5 (Representational Transparency).

The theory's functional units f_k are the *additive* components of h^(L).
Attention heads are the natural operationalisation: each head contributes
an independent additive term to the residual stream via

    f_{l,h}(X) = out_{l,h} @ W_O_h^T      shape (B, T, D)

where out_{l,h} = attn_probs_{l,h} @ V_{l,h} is the per-head value-weighted
sum and W_O_h is the corresponding column-slice of layer l's o_proj weight.

Both sub-claims are measured at head granularity (L*H functional units total):

  Sub-claim A  —  Jacobian rank-preservation
  ───────────────────────────────────────────
  For each head (l, h), ||J_{l,h}||_F estimated via randomised
  finite-difference probing of the per-head residual contribution.
  rho_{l,h} = ||J_{l,h}||_F / mean_{l,h}(||J_{l,h}||_F) ≈ 1 ⟺ A5A holds.

  Sub-claim B  —  Cross-unit gradient decorrelation
  ──────────────────────────────────────────────────
  cosine similarity between ∂L/∂f_{l,h} and ∂L/∂f_{l',h'} for all pairs.
  mean_off_diag = mean_{(l,h)≠(l',h')} |cos(g_{l,h}, g_{l',h'})| — primary scalar.


"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
from Transformer import apply_rope


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_head_dims(model: nn.Module) -> tuple[int, int, int]:
    """Return (n_layers, n_heads, head_dim)."""
    n_layers = len(model.layers)
    n_heads  = model.layers[0].attn.nhead
    head_dim = model.layers[0].attn.head_dim
    return n_layers, n_heads, head_dim


def _forward_collecting_head_contribs(
    model: nn.Module,
    x:     torch.Tensor,    # (B, T) long
) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
    """
    Manual autoregressive forward pass that builds per-head residual
    contribution tensors as explicit intermediate nodes in the autograd graph.

    For each head (l, h) we build:

        f_{l,h} = out_{l,h} @ W_O_h^T    (B, T, D)

    as a separate tensor, attach requires_grad=True to it as a leaf so its
    gradient can be read after backward, then sum all head contributions
    together with the FFN contribution to form the residual.

    Returns
    -------
    logits      : (B, T, V)  final logits
    head_leaves : list[list[Tensor(B,T,D)]]  head_leaves[l][h] — leaf with grad
    """
    n_layers, n_heads, head_dim = _get_head_dims(model)
    B, T = x.shape

    h_res = model.embed(x).float()
    # Anchor so upstream embeddings are in the graph
    anchor = torch.zeros_like(h_res, requires_grad=True)
    h_res  = h_res + anchor

    head_leaves: list[list[torch.Tensor]] = []

    for l_idx, layer in enumerate(model.layers):
        attn_mod = layer.attn

        # Pre-LN
        h_ln = layer.ln1(h_res)

        # QKV projections
        q = attn_mod.q_proj(h_ln).view(B, T, n_heads, head_dim)
        k = attn_mod.k_proj(h_ln).view(B, T, n_heads, head_dim)
        v = attn_mod.v_proj(h_ln).view(B, T, n_heads, head_dim)

        q = apply_rope(q, attn_mod.cos, attn_mod.sin)
        k = apply_rope(k, attn_mod.cos, attn_mod.sin)

        scores = (
            torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
            + attn_mod.causal_mask[:T, :T]
        )
        probs     = torch.softmax(scores, dim=-1)            # (B, H, T, T)
        out_heads = torch.einsum("bhts,bshd->bthd", probs, v)  # (B, T, H, head_dim)

        W_O = attn_mod.o_proj.weight   # (D, D)

        layer_leaves: list[torch.Tensor] = []
        attn_total = torch.zeros(B, T, W_O.shape[0],
                                 device=x.device, dtype=h_res.dtype)

        for h_idx in range(n_heads):
            W_O_h   = W_O[:, h_idx * head_dim : (h_idx + 1) * head_dim]
            contrib = out_heads[:, :, h_idx, :] @ W_O_h.T   # (B, T, D)

            # Detach and re-attach as a leaf so we can read its gradient.
            leaf = contrib.detach().requires_grad_(True)
            layer_leaves.append(leaf)

            # Add the leaf back into the residual via a +0*contrib path so
            # the graph is connected for backprop.
            attn_total = attn_total + leaf + 0.0 * contrib

        head_leaves.append(layer_leaves)

        # Residual update: attention total + FFN
        h_res = h_res + attn_total
        h_res = h_res + layer.ff(layer.ln2(h_res))

    logits = model.unembed(model.final_ln(h_res))
    return logits, head_leaves


# ─────────────────────────────────────────────────────────────────────────────
# Sub-claim B: head-level gradient decorrelation
# ─────────────────────────────────────────────────────────────────────────────

def measure_gradient_decorrelation(
    model:       nn.Module,
    test_set:    dict,
    device:      torch.device,
    max_samples: int = 256,
) -> dict:
    """
    Measure cross-head gradient cosine similarities (Sub-claim B).

    Returns
    -------
    dict with:
        cosine_matrix  : (L*H, L*H) ndarray, row/col ordered layer-major
        mean_off_diag  : float  primary scalar (lower = more decorrelated)
        grad_norms     : (L*H,) ndarray, ||g_{l,h}||_F per head
    """
    n = min(max_samples, len(test_set["tokens"]))
    tokens    = test_set["tokens"][:n]
    loss_mask = test_set["loss_mask"][:n]

    x = torch.tensor(tokens[:, :-1], dtype=torch.long, device=device)
    y = torch.tensor(tokens[:, 1:],  dtype=torch.long, device=device)
    m = torch.tensor(loss_mask,      dtype=torch.float, device=device)

    model.eval()

    with torch.enable_grad():
        logits, head_leaves = _forward_collecting_head_contribs(model, x)

        B, T, V = logits.shape
        criterion = nn.CrossEntropyLoss(reduction="none")
        per_token = criterion(
            logits.reshape(B * T, V), y.reshape(B * T)
        ).reshape(B, T)
        loss = (per_token * m).sum() / m.sum()
        loss.backward()

    # Collect gradients
    n_layers, n_heads, _ = _get_head_dims(model)
    grads = []
    for l in range(n_layers):
        for h in range(n_heads):
            leaf = head_leaves[l][h]
            g = (leaf.grad.detach() if leaf.grad is not None
                 else torch.zeros_like(leaf))
            grads.append(g.mean(dim=0).reshape(-1))   # (T*D,)

    # Zero param grads — no side effects on optimiser state
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    model.train()

    flat_grads = torch.stack(grads)                     # (L*H, T*D)
    norms  = flat_grads.norm(dim=1, keepdim=True).clamp(min=1e-10)
    normed = flat_grads / norms
    N      = n_layers * n_heads

    cosine_matrix = (normed @ normed.T).cpu().numpy()   # (N, N)
    grad_norms    = norms.squeeze(1).cpu().numpy()       # (N,)

    mask_off      = ~np.eye(N, dtype=bool)
    mean_off_diag = float(np.abs(cosine_matrix[mask_off]).mean())

    return {
        "min_cosine": cosine_matrix.min(),
        "mean_off_diag": mean_off_diag,
        "grad_norms":    grad_norms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sub-claim A: head-level Jacobian rank-preservation
# ─────────────────────────────────────────────────────────────────────────────

def measure_jacobian_rank_preservation(
    model:           nn.Module,
    test_set:        dict,
    device:          torch.device,
    n_probe_vectors: int = 16,
    max_samples:     int = 64,
    eps:             float = 1e-3,
) -> dict:
    """
    Estimate ||J_{l,h}||_F per attention head via randomised finite-difference
    probing on the per-head residual contribution f_{l,h}.

    Returns
    -------
    dict with:
        jacobian_norms  : (L, H) ndarray
        relative_ratios : (L, H) ndarray, rho_{l,h} = norm / mean_norm
    """
    n = min(max_samples, len(test_set["tokens"]))
    x = torch.tensor(
        test_set["tokens"][:n, :-1], dtype=torch.long, device=device
    )

    model.eval()
    n_layers, n_heads, head_dim = _get_head_dims(model)
    D = model.layers[0].ln1.normalized_shape[0]

    jacobian_norms = np.zeros((n_layers, n_heads), dtype=np.float64)

    with torch.no_grad():
        for l_tgt in range(n_layers):
            for h_tgt in range(n_heads):
                W_O   = model.layers[l_tgt].attn.o_proj.weight
                W_O_h = W_O[:, h_tgt * head_dim : (h_tgt + 1) * head_dim]

                probe_sq = []
                for _ in range(n_probe_vectors):
                    # ── Forward up to and including head h_tgt's contribution ──
                    h_res = model.embed(x).float()

                    for l_idx in range(l_tgt):
                        h_res = model.layers[l_idx](h_res)

                    # At the target layer: compute head h_tgt's contribution
                    layer    = model.layers[l_tgt]
                    attn_mod = layer.attn
                    B, T, _  = h_res.shape
                    h_ln     = layer.ln1(h_res)

                    q = attn_mod.q_proj(h_ln).view(B, T, n_heads, head_dim)
                    k = attn_mod.k_proj(h_ln).view(B, T, n_heads, head_dim)
                    v = attn_mod.v_proj(h_ln).view(B, T, n_heads, head_dim)

                    q = apply_rope(q, attn_mod.cos, attn_mod.sin)
                    k = apply_rope(k, attn_mod.cos, attn_mod.sin)

                    scores    = (
                        torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
                        + attn_mod.causal_mask[:T, :T]
                    )
                    probs     = torch.softmax(scores, dim=-1)
                    out_heads = torch.einsum("bhts,bshd->bthd", probs, v)

                    # Head h_tgt's residual contribution: (B, T, D)
                    h_contrib = out_heads[:, :, h_tgt, :] @ W_O_h.T

                    # Probe vector in residual space
                    v_probe = torch.randn(1, 1, D, device=device, dtype=h_res.dtype)
                    v_probe = v_probe / v_probe.norm()

                    # Sum all head contributions for target layer
                    # (other heads unperturbed)
                    attn_full  = layer.attn(h_ln)             # (B, T, D)
                    # Build base and perturbed residuals after full target layer
                    h_base = h_res + attn_full
                    h_base = h_base + layer.ff(layer.ln2(h_base))

                    h_pert_contrib = h_res + attn_full + eps * v_probe
                    h_pert_contrib = h_pert_contrib + layer.ff(
                        layer.ln2(h_pert_contrib)
                    )

                    # Propagate both through remaining layers
                    for l_idx in range(l_tgt + 1, n_layers):
                        h_base         = model.layers[l_idx](h_base)
                        h_pert_contrib = model.layers[l_idx](h_pert_contrib)

                    jv = (h_pert_contrib - h_base) / eps     # (B, T, D)
                    probe_sq.append(jv.pow(2).mean(dim=0).sum().item())

                jacobian_norms[l_tgt, h_tgt] = float(np.mean(probe_sq)) ** 0.5

    model.train()

    mean_norm       = jacobian_norms.mean()
    relative_ratios = jacobian_norms / (mean_norm + 1e-10)

    return {
        "jacobian_norms":   jacobian_norms,
        "relative_ratios":  relative_ratios,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined entry point
# ─────────────────────────────────────────────────────────────────────────────

def measure_assumption5(
    model:           nn.Module,
    test_set:        dict,
    device:          torch.device,
    max_samples:     int = 256,
    n_probe_vectors: int = 16,
) -> dict:
    """
    Compute both sub-claims at head granularity and return a flat dict.

    Keys
    ----
    mean_off_diag_cosine : float      Sub-claim B scalar (lower = more decorrelated)
    min_cosine           : float      Minimum value of cosine matrix
    grad_norms           : (L*H,)     Sub-claim B per-head upstream gradient norms
    jacobian_norms       : (L, H)     Sub-claim A per-head ||J_{l,h}||_F
    relative_ratios      : (L, H)     Sub-claim A rho_{l,h} (1.0 = uniform)
    """
    b = measure_gradient_decorrelation(
        model, test_set, device, max_samples=max_samples
    )
    a = measure_jacobian_rank_preservation(
        model, test_set, device,
        n_probe_vectors=n_probe_vectors,
        max_samples=min(64, max_samples),
    )
    return {
        "mean_off_diag_cosine": b["mean_off_diag"],
        "min_cosine":           b["min_cosine"],
        "grad_norms":           b["grad_norms"],
        "jacobian_norms":       a["jacobian_norms"],
        "relative_ratios":      a["relative_ratios"],
    }
