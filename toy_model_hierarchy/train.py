"""
train.py
────────
Training loop for autoregressive Transformer models on HierarchicalDGP /
FlatDGP data.  All analysis / measurement logic lives in metrics.py.

Returns a history dict with the following keys:

    trained_model          : nn.Module  — the trained model on its device
    loss_history           : list[float]          — per-step train CE loss
    ppl_history            : list[float]          — per-step train perplexity
    lr_history             : list[float]          — LR at each step
    eval_steps             : list[int]            — steps where eval ran
    test_loss_history      : list[float]          — masked CE on test set
    layerwise_hydra_history: list[float]          — scalar max layerwise Hydra
    headwise_hydra_history : list[np.ndarray]     — (n_layers, n_heads) matrices
    full_hydra_history     : list[np.ndarray]     — (n_layers, n_heads, n_layers)
    induction_history      : list[float]          — induction head score (max over heads)
    function_vector_history: list[float]          — function vector patching score
    a6_mean_curv_history   : list[float]          — mean directional curvature (A6)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dgp import _BaseDGP
from metrics import (
    evaluate,
    calculate_layerwise_hydra,
    calculate_headwise_hydra,
    measure_induction_score,
    function_vector_analysis,
    measure_directional_concavity,
)
from assumption5 import measure_assumption5


def train(
    model:            nn.Module,
    dgp:              _BaseDGP,
    # ── data ──────────────────────────────────────────────────────────────────
    num_batches:      int,
    batch_size:       int,
    replacement:      bool                      = True,
    dataset_size:     Optional[int]             = None,
    # ── test sets ─────────────────────────────────────────────────────────────
    test_sets:        Optional[dict]            = None,
    eval_every:       int                       = 50,
    # ── optimiser ─────────────────────────────────────────────────────────────
    lr:               float                     = 3e-4,
    weight_decay:     float                     = 0.1,
    betas:            tuple[float, float]       = (0.9, 0.95),
    grad_clip:        Optional[float]           = 1.0,
    # ── LR schedule ───────────────────────────────────────────────────────────
    warmup_batches:   int                       = 0,
    use_cosine_decay: bool                      = False,
    # ── logging ───────────────────────────────────────────────────────────────
    log_every:        int                       = 10,
    # ── misc ──────────────────────────────────────────────────────────────────
    device:           Optional[str]             = None,
) -> dict:
    """
    Train an autoregressive Transformer on sequences from a DGP instance.

    Parameters
    ----------
    model            : nn.Module whose forward(x) returns (B, T, V) logits
    dgp              : HierarchicalDGP or FlatDGP instance
    num_batches      : total optimisation steps
    batch_size       : sequences per batch
    replacement      : whether to sample with replacement from the DGP
    dataset_size     : pre-generated dataset size (replacement=False only)
    test_sets        : dict with 'tokens' and 'loss_mask' arrays.
                       If None, no evaluation is performed.
    eval_every       : evaluate every this many steps (also at step 0 and last)
    lr               : AdamW peak learning rate
    weight_decay     : AdamW weight decay
    betas            : AdamW beta coefficients
    grad_clip        : max gradient norm (None to disable)
    warmup_batches   : linear LR warmup steps (0 to disable)
    use_cosine_decay : cosine-anneal from peak LR → 0 after warmup
    log_every        : train-loss print interval (0 for silent)
    device           : 'cpu' | 'cuda' | 'mps' | None (auto-detect)
    """

    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    model = model.to(dev)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr, betas=betas,
    )

    # ── LR schedule ───────────────────────────────────────────────────────────
    scheduler = None
    if use_cosine_decay:
        t_max     = max(num_batches - warmup_batches, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0.0)

    def _current_lr() -> float:
        return optimizer.param_groups[0]["lr"]

    def _apply_warmup(step: int) -> None:
        if warmup_batches > 0 and step < warmup_batches:
            scale = (step + 1) / warmup_batches
            for g in optimizer.param_groups:
                g["lr"] = lr * scale

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(reduction="none")

    # ── History ───────────────────────────────────────────────────────────────
    seg_len = dgp.seq_len // dgp.num_segments

    loss_history:               list[float]       = []
    ppl_history:                list[float]       = []
    lr_history:                 list[float]       = []
    eval_steps:                 list[int]         = []
    test_loss_history:          list[float]       = []
    layerwise_hydra_history:    list[float]       = []
    headwise_hydra_history:     list[np.ndarray]  = []
    full_hydra_history:         list[np.ndarray]  = []
    induction_history:          list[float]       = []
    function_vector_history:    list[float]       = []
    # Assumption 5 — collected at each eval step so max/AUC aggregations are valid
    a5B_min_cosine_history:     list[np.ndarray]  = []   # Sub-claim B: min[cosine]
    a5B_cosine_history:         list[float]       = []   # Sub-claim B: mean |cosine|
    a5A_rho_std_history:        list[float]       = []   # Sub-claim A: std of ρ_k
    # Assumption 6 — directional concavity (≤0 supports A6; primary test statistic)
    a6_directional_curvature_history: list[np.ndarray] = []
    a6_mean_curv_history:       list[float]       = []   # mean u₁ᵀ H_ϕ u₁
    a6_prop_neg_history:        list[float]       = []   # proportion of samples with c < 0

    # ── Eval closure ──────────────────────────────────────────────────────────
    def _run_eval(step: int) -> None:
        if test_sets is None:
            return

        test_loss_history.append(evaluate(model, test_sets, dev))
        
        layerwise_hydra_history.append(
            calculate_layerwise_hydra(model, test_sets, dev)
        )

        hw = calculate_headwise_hydra(model, test_sets, dev)
        headwise_hydra_history.append(hw.matrix)
        full_hydra_history.append(hw.full_tensor)
        
        induction_history.append(
            measure_induction_score(model, test_sets, dev)
        )
        function_vector_history.append(
            function_vector_analysis(model, test_sets, seg_len, dev)
        )

        # Assumption 5: measure at every eval step so time-series aggregations
        # (max, AUC) are meaningful for significance testing.
        a5 = measure_assumption5(model, test_sets, dev)
        a5B_min_cosine_history.append(a5["min_cosine"])
        a5B_cosine_history.append(float(a5["mean_off_diag_cosine"]))
        a5A_rho_std_history.append(float(a5["relative_ratios"].std()))

        # Assumption 6: directional concavity of the readout ϕ along u₁.
        # max_samples=32 keeps Hessian cost manageable at each eval step.
        a6 = measure_directional_concavity(model, test_sets, dev, max_samples=32)
        a6_directional_curvature_history.append(a6["directional_curvatures"])
        a6_mean_curv_history.append(a6["mean_directional_curv"])
        a6_prop_neg_history.append(a6["prop_negative_direction"])

        eval_steps.append(step)

        if log_every > 0:
            lw   = layerwise_hydra_history[-1]
            pos  = hw.matrix[hw.matrix > 0]
            hw_s = pos.mean() if pos.size > 0 else 0.0
            print(
                f"  {'':>14s}  eval  "
                f"loss={test_loss_history[-1]:.4f}  "
                f"lw_hydra={lw:.4f}  "
                f"hw_hydra={hw_s:.4f}  "
                f"induction={induction_history[-1]:.4f}  "
                f"fv={function_vector_history[-1]:.4f}  "
                f"a5B={a5B_cosine_history[-1]:.4f}  "
                f"a5A_rho_std={a5A_rho_std_history[-1]:.4f}  "
                f"a6_curv={a6_mean_curv_history[-1]:.4f}  "
                f"a6_neg={a6_prop_neg_history[-1]:.2%}"
            )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.perf_counter()

    loader = dgp.dataloader(
        batch_size   = batch_size,
        num_batches  = num_batches,
        replacement  = replacement,
        dataset_size = dataset_size,
    )

    for batch in loader:
        step    = batch["batch_idx"]
        is_last = step == num_batches - 1

        _apply_warmup(step)
        lr_history.append(_current_lr())

        if step == 0 or step % eval_every == 0 or is_last:
            _run_eval(step)

        # ── Forward / backward ────────────────────────────────────────────────
        x    = torch.tensor(batch["input_tokens"],  dtype=torch.long, device=dev)
        y    = torch.tensor(batch["target_tokens"], dtype=torch.long, device=dev)
        mask = torch.tensor(batch["loss_mask"],     dtype=torch.bool,  device=dev)

        logits = model(x)
        B, T, V = logits.shape

        per_token_loss = criterion(
            logits.reshape(B * T, V), y.reshape(B * T)
        ).reshape(B, T)
        loss = (per_token_loss * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if scheduler is not None and step >= warmup_batches:
            scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        ppl_history.append(float(torch.exp(loss).item()))

        if log_every > 0 and (step % log_every == 0 or is_last):
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:>6d}/{num_batches}  "
                f"loss={loss_val:.4f}  "
                f"ppl={ppl_history[-1]:.2f}  "
                f"lr={_current_lr():.2e}  "
                f"elapsed={elapsed:.1f}s"
            )
    del model
    torch.cuda.empty_cache()
    elapsed = time.perf_counter() - t0
    return {
        #"trained_model":           model,
        "loss_history":            loss_history,
        "ppl_history":             ppl_history,
        "lr_history":              lr_history,
        "eval_steps":              eval_steps,
        "test_loss_history":       test_loss_history,
        "layerwise_hydra_history": layerwise_hydra_history,
        "headwise_hydra_history":  headwise_hydra_history,
        "full_hydra_history":      full_hydra_history,
        "induction_history":       induction_history,
        "function_vector_history": function_vector_history,
        # Assumption 5 time-series (one value per eval step)
        "a5B_min_cosine_history": a5B_min_cosine_history,
        "a5B_cosine_history":      a5B_cosine_history,
        "a5A_rho_std_history":     a5A_rho_std_history,
        # Assumption 6 time-series (one value per eval step)
        "a6_directional_curvature_history": a6_directional_curvature_history,
        "a6_mean_curv_history":    a6_mean_curv_history,
        "a6_prop_neg_history":     a6_prop_neg_history,
        "elapsed_sec":             elapsed,
    }