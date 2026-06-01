"""
experiment.py
─────────────
Main experiment script.  Trains one Transformer on HierarchicalDGP and one on
FlatDGP, then plots test loss and all mechanistic metrics side-by-side.

Usage
-----
    python experiment.py               # uses defaults from config.py
    python experiment.py --steps 2000  # quick override via argparse
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import cfg
from dgp import HierarchicalDGP, FlatDGP, build_hierarchical_test_set, build_flat_test_set, _bayes_ce_hierarchical
from train import train
from Transformer import Transformer


# ─────────────────────────────────────────────────────────────────────────────
# CLI overrides (lightweight — no need for a full config file)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int,   default=cfg.experiment.steps)
    p.add_argument("--seed",       type=int,   default=cfg.experiment.seed)
    p.add_argument("--batch_size", type=int,   default=cfg.experiment.batch_size)
    p.add_argument("--eval_every", type=int,   default=cfg.experiment.eval_every)
    p.add_argument("--log_every",  type=int,   default=cfg.experiment.log_every)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DGP / model factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_hierarchical_dgp(seed: int) -> HierarchicalDGP:
    dc = cfg.data
    return HierarchicalDGP(
        vocab_size        = dc.vocab_size,
        seq_len           = dc.seq_len,
        num_segments      = dc.num_segments,
        alpha_query       = dc.alpha_query,
        k0                = dc.K0,
        k1                = dc.K1,
        seed              = seed,
        pi1_concentration = dc.pi1_concentration,
        embedding_noise   = dc.embedding_noise,
    )


def make_flat_dgp(seed: int) -> FlatDGP:
    dc = cfg.data
    return FlatDGP(
        vocab_size   = dc.vocab_size,
        seq_len      = dc.seq_len,
        num_segments = dc.num_segments,
        k1           = dc.K1,
        seed         = seed,
    )


def make_model() -> Transformer:
    mc = cfg.model
    return Transformer(
        vocab_size = cfg.data.vocab_size + 1,
        n_layers   = mc.num_layers,
        d_model    = mc.d_model,
        nhead      = mc.num_heads,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-run training helper
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(args: argparse.Namespace) -> tuple[dict, dict]:
    """Train one hierarchical and one flat model; return their history dicts."""
    torch.manual_seed(args.seed)
    seed = args.seed
    while True:
        h_dgp  = make_hierarchical_dgp(seed)
        f_dgp  = make_flat_dgp(seed)
        delta = _bayes_ce_hierarchical(h_dgp, 0)["H_query"] - _bayes_ce_hierarchical(h_dgp, h_dgp.num_segments-1)["H_query"]
        if delta < 0.08:
            print('Insufficeint improvment from context, trying new seed...')
            seed += 100
        else:
            break
    
    h_test = build_hierarchical_test_set(h_dgp, cfg.experiment.test_size)
    f_test = build_flat_test_set(f_dgp,  cfg.experiment.test_size)

    train_kwargs = dict(
        num_batches      = args.steps,
        batch_size       = args.batch_size,
        lr               = cfg.experiment.lr,
        warmup_batches   = cfg.experiment.warmup_batches,
        use_cosine_decay = cfg.experiment.use_cosine_decay,
        eval_every       = args.eval_every,
        log_every        = args.log_every,
    )

    print("=== Training on HierarchicalDGP ===")
    h_history = train(make_model(), h_dgp, test_sets=h_test, **train_kwargs)

    print("\n=== Training on FlatDGP ===")
    f_history = train(make_model(), f_dgp, test_sets=f_test, **train_kwargs)

    return h_history, f_history


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(h_history: dict, f_history: dict) -> None:
    """
    Four-panel plot:
      1. Test loss
      2. Layerwise Hydra (scalar max)
      3. Headwise Hydra (mean of positive entries per eval step)
      4. Generalised Induction + Function Vector score
    """
    steps = h_history["eval_steps"]

    def _hw_scalar(hist: dict) -> list[float]:
        """Mean of positive entries in the headwise Hydra matrix per step."""
        out = []
        for mat in hist["headwise_hydra_history"]:
            pos = mat[mat > 0]
            out.append(float(pos.mean()) if pos.size > 0 else 0.0)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Hierarchical vs Flat DGP — mechanistic metrics", fontsize=13)

    panel_data = [
        (axes[0, 0], "Test loss (CE)",
         h_history["test_loss_history"],
         f_history["test_loss_history"]),
        (axes[0, 1], "Layerwise Hydra (max)",
         h_history["layerwise_hydra_history"],
         f_history["layerwise_hydra_history"]),
        (axes[1, 0], "Headwise Hydra (mean positive)",
         _hw_scalar(h_history),
         _hw_scalar(f_history)),
        (axes[1, 1], "Generalised Induction score",
         h_history["induction_history"],
         f_history["induction_history"]),
    ]

    for ax, title, h_vals, f_vals in panel_data:
        ax.plot(steps, h_vals, label="hierarchical", color="steelblue")
        ax.plot(steps, f_vals, label="flat",         color="tomato", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Overlay function vector on the induction panel (right y-axis)
    ax2 = axes[1, 1].twinx()
    ax2.plot(steps, h_history["function_vector_history"],
             color="steelblue", alpha=0.5, linestyle=":", label="FV hier.")
    ax2.plot(steps, f_history["function_vector_history"],
             color="tomato",    alpha=0.5, linestyle=":", label="FV flat")
    ax2.set_ylabel("Function vector score", color="grey", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="grey")

    fig.tight_layout()
    plt.show()

    # ── Head-wise Hydra heatmap at final eval step ─────────────────────────────
    _plot_headwise_heatmap(h_history, f_history)


def _plot_headwise_heatmap(h_history: dict, f_history: dict) -> None:
    """Plot end-of-training headwise Hydra matrix as a heatmap for both DGPs."""
    import matplotlib.colors as mcolors

    h_mat = h_history["headwise_hydra_history"][-1]   # (L, H)
    f_mat = f_history["headwise_hydra_history"][-1]
    vmax  = max(np.abs(h_mat).max(), np.abs(f_mat).max(), 1e-6)
    norm  = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    n_layers, n_heads = h_mat.shape
    fig, axes = plt.subplots(1, 2, figsize=(max(6, n_heads * 1.5), max(4, n_layers)))
    fig.suptitle("End-of-training Headwise Hydra matrix", fontsize=12)

    for ax, mat, title in zip(axes, [h_mat, f_mat], ["Hierarchical", "Flat"]):
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, label="Hydra score")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
        for l in range(n_layers):
            for h in range(n_heads):
                ax.text(h, l, f"{mat[l, h]:.2f}", ha="center", va="center",
                        fontsize=7)
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    h_history, f_history = run_single_experiment(args)
    plot_results(h_history, f_history)
