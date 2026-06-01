"""
significance_test.py
────────────────────
Statistical significance testing: Hierarchical DGP vs Flat DGP.

Scientific question
───────────────────
Are the following mechanistic effects selectively and significantly stronger
when training on HierarchicalDGP compared to FlatDGP?

Mechanistic effects tested (H₁: hierarchical > flat):
    layerwise_hydra      — max entry of the layerwise Hydra matrix
    headwise_hydra       — mean of positive entries in the (L, H) Hydra matrix
    induction            — generalised induction probability boost
    function_vector      — max mean layer patching effect

Assumption 5 sub-claims tested at end of training (single measurement per seed):
    a5B_cosine           — mean off-diagonal gradient cosine similarity
                           H₁: hierarchical < flat  (more decorrelated)
    a5A_rho_uniformity   — std of per-layer Jacobian norm ratios ρ_k
                           H₁: hierarchical < flat  (more uniform / rank-preserving)

Metric types
────────────
Time-series metrics (Hydra, induction, function vector):
    Tested at three aggregation windows: final / max / auc
    (final = last eval step, max = peak over training, auc = normalised area)

Point-in-time metrics (A5B, A5A):
    Measured once on the final trained model. Tested at the "final" window only.
    Not replicated across windows — that would inflate Bonferroni count spuriously.

Statistical tests (one-sided, paired across seeds):
    paired t-test          — parametric; H₁: hierarchical {>/<} flat
    Wilcoxon signed-rank   — non-parametric; same directional hypothesis

Effect size: Cohen's d (paired formulation: mean_diff / std_diff)
Multiple comparisons: Bonferroni correction over actual test count
    = (4 time-series metrics × 3 windows) + (2 A5 metrics × 1 window) = 14

Outputs (written to --out_dir, default: results/)
───────────────────────────────────────────────────
significance_summary.csv     — full table: means, diffs, effect sizes, p-values
metric_distributions.svg     — box plots per metric × window × DGP type
training_curves_mean.svg     — mean ± 1 std curves over seeds (time-series metrics)
pvalue_heatmap.svg           — heatmap of one-sided p-values (metric × window)
effect_sizes.svg             — horizontal bar chart of Cohen's d (final window)
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from assumption5 import measure_assumption5
from config import cfg
from dgp import build_hierarchical_test_set, build_flat_test_set, _bayes_ce_hierarchical
from experiment import make_hierarchical_dgp, make_flat_dgp, make_model
from train import train


# ─────────────────────────────────────────────────────────────────────────────
# Metric registry
#
# Each entry:
#   key        : column name used throughout
#   direction  : "greater" → H₁: hier > flat;  "less" → H₁: hier < flat
#   label      : human-readable axis label
#   windows    : which aggregation windows apply
#                "all"   → final / max / auc  (time-series metrics logged in train())
#                "final" → final only         (point-in-time, measured post-hoc)
# ─────────────────────────────────────────────────────────────────────────────

MECHANISTIC_METRICS: list[dict] = [
    # ── Mechanistic effects: H₁ = hierarchical > flat ─────────────────────────
    {"key": "layerwise_hydra",    "direction": "greater", "windows": "all",
     "label": "Layerwise Hydra (max)"},
    {"key": "headwise_hydra",     "direction": "greater", "windows": "all",
     "label": "Headwise Hydra (mean pos.)"},
    {"key": "induction",          "direction": "greater", "windows": "all",
     "label": "Induction head score (max)"},
    {"key": "function_vector",    "direction": "greater", "windows": "all",
     "label": "Function Vector score"},
    # ── Assumption 5: H₁ = hierarchical < flat ────────────────────────────────
    # Both are measured at each eval step inside train(), so all three
    # aggregation windows (final / max / auc) are valid and tested.
    # {"key": "a5B_cosine",         "direction": "less",    "windows": "all",
    #  "label": "A5B Mean |cosine| (↓ = more decorr.)"},
    # {"key": "a5A_rho_uniformity", "direction": "less",    "windows": "all",
    #  "label": "A5A ρ std (↓ = more uniform)"},
    # ── Assumption 6: directional concavity ───────────────────────────────────
    # H₁: hierarchical produces more negative directional curvature (u₁ᵀ H_ϕ u₁ < 0)
    # mean_curv: lower in hierarchical (more negative = more concave)
    # prop_neg:  higher in hierarchical (more samples with c < 0)
    # {"key": "a6_mean_curv",  "direction": "less",    "windows": "all",
    #  "label": "A6 Mean directional curvature (↓ = more concave)"},
    # {"key": "a6_prop_neg",   "direction": "greater", "windows": "all",
    #  "label": "A6 Prop. negative curvature (↑ = supports A6)"},
]

ALL_WINDOWS     = ["final", "max", "auc"]
TIMESERIES_KEYS = [
    "layerwise_hydra",
    "headwise_hydra",
    "induction",
    "function_vector",
    "a5B_min_cosine",
    "a6_mean_curv",
    ]   

METRIC_DIRECTION: dict[str, str] = {m["key"]: m["direction"] for m in MECHANISTIC_METRICS}
METRIC_LABEL:     dict[str, str] = {m["key"]: m["label"]     for m in MECHANISTIC_METRICS}

# 8 metrics × 3 windows = 24 tests total (Bonferroni denominator)
N_TESTS_TOTAL = len(MECHANISTIC_METRICS) * len(ALL_WINDOWS)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Significance test: Hierarchical vs Flat DGP"
    )
    p.add_argument("--n_seeds",   type=int,   default=cfg.significance.n_seeds,
                   help="Independent seeds per DGP (default: %(default)s)")
    p.add_argument("--fast",      action="store_true",
                   help="Reduce steps/test_size for a quick smoke test")
    p.add_argument("--out_dir",   type=str,   default="results",
                   help="Output directory for CSV and figures")
    p.add_argument("--base_seed", type=int,   default=115,
                   help="Seeds are base_seed, base_seed+1, …")
    p.add_argument("--alpha",     type=float, default=0.05,
                   help="Significance threshold (default: 0.05)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Scalar extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _headwise_scalar(history: dict) -> list[float]:
    """Mean of positive entries in the headwise Hydra matrix per eval step."""
    out = []
    for mat in history["headwise_hydra_history"]:
        pos = mat[mat > 0]
        out.append(float(pos.mean()) if pos.size > 0 else 0.0)
    return out


def _auc(steps: list[int], values: list[float]) -> float:
    """Trapezoidal AUC normalised by the training span."""
    if len(steps) < 2:
        return float(values[0]) if values else 0.0
    span  = steps[-1] - steps[0]
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(values, steps) / max(span, 1))


def extract_timeseries(history: dict) -> dict[str, list[float]]:
    """
    Return {metric_key: time_series} for all significance-tested metrics.

    All metrics are now proper time-series (one value per eval step), enabling
    final / max / auc aggregations for Hydra, induction, function vector, AND
    Assumption 5 alike.  A5 is measured inside train() at every eval step.
    """
    return {
        "layerwise_hydra":    history["layerwise_hydra_history"],
        "headwise_hydra":     _headwise_scalar(history),
        "induction":          history["induction_history"],
        "function_vector":    history["function_vector_history"],
        "a5B_min_cosine":    history["a5B_min_cosine_history"],
        # "a5B_cosine":         history["a5B_cosine_history"],
        # "a5A_rho_uniformity": history["a5A_rho_std_history"],
        "a6_mean_curv":       history["a6_mean_curv_history"],
        # "a6_prop_neg":        history["a6_prop_neg_history"],
    }


def summarise_run(history: dict) -> dict[str, float]:
    """
    Collapse each metric time-series into three scalars (final / max / auc).
    Returns a flat dict: {metric__window: value}.

    All six metrics — including Assumption 5 — are proper time-series, so
    all three aggregation windows are meaningful for each.
    """
    steps  = history["eval_steps"]
    series = extract_timeseries(history)
    out: dict[str, float] = {}
    for name, values in series.items():
        out[f"{name}__final"] = float(values[-1])
        out[f"{name}__max"]   = float(max(values))
        out[f"{name}__auc"]   = _auc(steps, values)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Paired Cohen's d = mean(a − b) / std(a − b, ddof=1).
    Positive → a > b (hierarchical > flat).
    """
    diff = a - b
    sd   = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else float("nan")


def run_tests(
    h_values:   np.ndarray,
    f_values:   np.ndarray,
    metric_key: str,
    window:     str,
    alpha:      float = 0.05,
) -> dict:
    """
    One-sided paired t-test and Wilcoxon signed-rank test.

    The alternative hypothesis direction is looked up from METRIC_DIRECTION:
        "greater" → H₁: hierarchical > flat   (Hydra, induction, FV)
        "less"    → H₁: hierarchical < flat   (A5B cosine, A5A rho std)

    Bonferroni threshold is computed from N_TESTS_TOTAL (14 actual tests).

    Returns a dict with all statistics and significance flags.
    """
    direction    = METRIC_DIRECTION[metric_key]
    diff         = h_values - f_values     # positive → hier > flat
    alpha_bonf   = alpha / N_TESTS_TOTAL

    # ── Paired t-test ─────────────────────────────────────────────────────────
    t_stat, t_pval_two = stats.ttest_rel(h_values, f_values)
    if direction == "greater":
        t_pval_one = t_pval_two / 2 if t_stat > 0 else 1 - t_pval_two / 2
    else:
        t_pval_one = t_pval_two / 2 if t_stat < 0 else 1 - t_pval_two / 2

    # ── Wilcoxon signed-rank ──────────────────────────────────────────────────
    try:
        w_stat, w_pval_one = stats.wilcoxon(
            diff, alternative=direction, zero_method="wilcox"
        )
        _, w_pval_two = stats.wilcoxon(diff, alternative="two-sided",
                                        zero_method="wilcox")
    except ValueError:
        w_stat = w_pval_one = w_pval_two = float("nan")

    return {
        "metric_key":   metric_key,
        "window":       window,
        "metric_label": METRIC_LABEL.get(metric_key, metric_key),
        "direction":    direction,
        "h_mean":       float(h_values.mean()),
        "h_std":        float(h_values.std(ddof=1)),
        "f_mean":       float(f_values.mean()),
        "f_std":        float(f_values.std(ddof=1)),
        "mean_diff":    float(diff.mean()),
        "cohens_d":     cohens_d_paired(h_values, f_values),
        # t-test
        "t_stat":       float(t_stat),
        "t_pval_one":   float(t_pval_one),
        "t_pval_two":   float(t_pval_two),
        # Wilcoxon
        "w_stat":       float(w_stat),
        "w_pval_one":   float(w_pval_one),
        "w_pval_two":   float(w_pval_two),
        # Significance flags (one-sided)
        "sig_t":        bool(t_pval_one < alpha),
        "sig_w":        bool(w_pval_one < alpha),
        "sig_t_bonf":   bool(t_pval_one < alpha_bonf),
        "sig_w_bonf":   bool(w_pval_one < alpha_bonf),
    }


def build_results_df(
    h_summaries: list[dict[str, float]],
    f_summaries: list[dict[str, float]],
    alpha: float,
) -> pd.DataFrame:
    """
    Run all tests and return a single results DataFrame.

    All six metrics (Hydra ×2, induction, function vector, A5B, A5A) are
    tested across all three aggregation windows (final / max / auc), giving
    18 tests total.  Bonferroni correction is applied to that denominator.
    """
    rows: list[dict] = []
    for meta in MECHANISTIC_METRICS:
        key = meta["key"]
        for window in ALL_WINDOWS:
            col_key = f"{key}__{window}"
            h_vals  = np.array([s[col_key] for s in h_summaries])
            f_vals  = np.array([s[col_key] for s in f_summaries])
            rows.append(run_tests(h_vals, f_vals, key, window, alpha=alpha))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(
    h_summaries: list[dict],
    f_summaries: list[dict],
    results_df:  pd.DataFrame,
    out_dir:     str,
    alpha:       float,
) -> None:
    """
    Box plots for all 6 metrics × 3 aggregation windows in a single figure.
    A horizontal separator divides mechanistic metrics (top) from A5 (bottom).
    Significant panels are highlighted with a gold border.
    """
    keys    = [m["key"] for m in MECHANISTIC_METRICS]
    n_rows  = len(keys)
    n_cols  = len(ALL_WINDOWS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Metric distributions: Hierarchical (blue) vs Flat (orange)\n"
        f"n={len(h_summaries)} seeds   Gold border = one-sided t-test p < {alpha}",
        fontsize=11,
    )

    for row, meta in enumerate(MECHANISTIC_METRICS):
        key = meta["key"]
        for col, window in enumerate(ALL_WINDOWS):
            col_key = f"{key}__{window}"
            h_vals  = np.array([s[col_key] for s in h_summaries])
            f_vals  = np.array([s[col_key] for s in f_summaries])

            ax = axes[row, col]
            bp = ax.boxplot(
                [h_vals, f_vals],
                labels=["Hier.", "Flat"],
                patch_artist=True, widths=0.5,
                medianprops={"color": "black", "linewidth": 2},
            )
            bp["boxes"][0].set_facecolor("steelblue")
            bp["boxes"][1].set_facecolor("tomato")
            ax.set_title(f"{meta['label']}\n({window})", fontsize=8)
            ax.grid(True, alpha=0.3)

            row_data = results_df[
                (results_df["metric_key"] == key) &
                (results_df["window"] == window)
            ]
            if not row_data.empty:
                p   = float(row_data["t_pval_one"].iloc[0])
                sig = bool(row_data["sig_t"].iloc[0])
                d   = float(row_data["cohens_d"].iloc[0])
                stars = ("***" if p < 0.001 else "**" if p < 0.01
                         else "*" if p < 0.05 else "ns")
                ax.set_xlabel(f"p={p:.3f} {stars}  d={d:.2f}", fontsize=7)
                if sig:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("goldenrod")
                        spine.set_linewidth(2.5)

    fig.savefig(os.path.join(out_dir, "metric_distributions.svg"))
    plt.close(fig)
    print("Saved metric_distributions.svg")


def plot_training_curves(
    all_h_histories: list[dict],
    all_f_histories: list[dict],
    out_dir:         str,
) -> None:
    """
    Mean ± 1 std training curves for all six significance-tested metrics
    (Hydra ×2, induction, function vector, A5B, A5A), arranged in a 2×3 grid.
    A5 curves are now proper time-series (measured at each eval step in train()).
    """
    min_evals = min(len(h["eval_steps"]) for h in all_h_histories)
    steps     = np.array(all_h_histories[0]["eval_steps"][:min_evals])

    n_cols = 3
    n_rows = 2  # 6 metrics, 3 per row
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True)
    fig.suptitle(
        f"Mean ± 1 std training curves  (n={len(all_h_histories)} seeds)",
        fontsize=12,
    )

    for ax, key in zip(axes.flat, TIMESERIES_KEYS):
        h_curves, f_curves = [], []
        for h_hist, f_hist in zip(all_h_histories, all_f_histories):
            
            h_curves.append(extract_timeseries(h_hist)[key][:min_evals])
            f_curves.append(extract_timeseries(f_hist)[key][:min_evals])

        h_arr = np.array(h_curves)
        f_arr = np.array(f_curves)

        for arr, label, color in [
            (h_arr, "Hierarchical", "steelblue"),
            (f_arr, "Flat",         "tomato"),
        ]:
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(steps, mean, label=label, color=color, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_title(key, fontsize=9)
        ax.set_xlabel("Training step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "training_curves_mean.svg"))
    plt.close(fig)
    print("Saved training_curves_mean.svg")


def plot_pvalue_heatmap(results_df: pd.DataFrame, out_dir: str, alpha: float) -> None:
    """
    One heatmap per test type (t-test / Wilcoxon) showing one-sided p-values
    for all 6 metrics × 3 windows.

    Gold border = p < alpha.  Navy dashed border = Bonferroni-corrected significance.
    """
    alpha_bonf = alpha / N_TESTS_TOTAL
    keys       = [m["key"]   for m in MECHANISTIC_METRICS]
    row_labels = [m["label"] for m in MECHANISTIC_METRICS]

    def _build_matrix(col_name: str) -> np.ndarray:
        mat = np.full((len(keys), len(ALL_WINDOWS)), np.nan)
        for r, k in enumerate(keys):
            for c, w in enumerate(ALL_WINDOWS):
                sub = results_df[
                    (results_df["metric_key"] == k) &
                    (results_df["window"] == w)
                ]
                if not sub.empty:
                    mat[r, c] = float(sub[col_name].iloc[0])
        return mat

    p_t = _build_matrix("t_pval_one")
    p_w = _build_matrix("w_pval_one")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, max(4, len(keys) * 1.1)),
        constrained_layout=True,
    )
    fig.suptitle(
        f"One-sided p-values: Hierarchical vs Flat DGP\n"
        f"Gold = p < {alpha}   Navy dashed = Bonferroni (p < {alpha_bonf:.4f})\n"
        f"Hydra/Induction/FV: H₁ hier > flat   A5B/A5A: H₁ hier < flat",
        fontsize=10,
    )

    for ax, pmat, title in zip(axes, [p_t, p_w],
                                ["Paired t-test (one-sided)",
                                 "Wilcoxon signed-rank (one-sided)"]):
        im = ax.imshow(pmat, vmin=0, vmax=0.2, cmap="RdYlGn_r", aspect="auto")

        ax.set_xticks(range(len(ALL_WINDOWS)))
        ax.set_xticklabels(ALL_WINDOWS)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(row_labels, fontsize=8)

        for r in range(pmat.shape[0]):
            for c in range(pmat.shape[1]):
                v = pmat[r, c]
                if np.isnan(v):
                    continue
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v < 0.08 else "black",
                        fontweight="bold")
                if v < alpha:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        fill=False, edgecolor="goldenrod", linewidth=2.5))
                if v < alpha_bonf:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        fill=False, edgecolor="navy",
                        linewidth=1.5, linestyle="--"))

        # Separator line between mechanistic metrics and A5
        n_mech = sum(1 for m in MECHANISTIC_METRICS if m["direction"] == "greater")
        ax.axhline(n_mech - 0.5, color="white", linewidth=2, linestyle="--")

        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="p-value")

    fig.savefig(os.path.join(out_dir, "pvalue_heatmap.svg"))
    plt.close(fig)
    print("Saved pvalue_heatmap.svg")


def plot_effect_sizes(results_df: pd.DataFrame, out_dir: str) -> None:
    """
    Horizontal bar chart of Cohen's d at the 'final' window for all metrics.
    Sign convention: positive always means 'effect in the predicted direction'.
    Significant bars (one-sided t-test) are coloured blue; others grey.
    """
    final_df = results_df[results_df["window"] == "final"].copy()

    # Flip sign for "less" metrics so positive = predicted direction
    final_df["d_directional"] = final_df.apply(
        lambda r: -r["cohens_d"] if r["direction"] == "less" else r["cohens_d"],
        axis=1,
    )
    final_df = final_df.sort_values("d_directional", ascending=True)

    colors = ["steelblue" if sig else "lightgrey" for sig in final_df["sig_t"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(final_df) * 0.65)))
    bars = ax.barh(
        final_df["metric_label"], final_df["d_directional"],
        color=colors, edgecolor="black", linewidth=0.7,
    )
    ax.axvline(0,    color="black", linewidth=1)
    ax.axvline( 0.5, color="grey",  linewidth=0.8, linestyle=":")
    ax.axvline(-0.5, color="grey",  linewidth=0.8, linestyle=":")
    ax.set_xlabel("Cohen's d  (positive = effect in predicted direction)")
    ax.set_title(
        "Effect sizes (Cohen's d, final window)\n"
        "Blue = significant (p < α, one-sided t-test)   Grey = not significant"
    )
    ax.grid(True, axis="x", alpha=0.3)

    for bar, (_, row) in zip(bars, final_df.iterrows()):
        p     = row["t_pval_one"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        x     = bar.get_width()
        ax.text(x + 0.02, bar.get_y() + bar.get_height() / 2,
                stars, va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "effect_sizes.svg"))
    plt.close(fig)
    print("Saved effect_sizes.svg")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    steps      = 500 if args.fast else cfg.experiment.steps
    test_size  = 64  if args.fast else cfg.experiment.test_size
    batch_size = cfg.experiment.batch_size
    eval_every = 50  if args.fast else cfg.experiment.eval_every

    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    h_summaries:     list[dict[str, float]] = []
    f_summaries:     list[dict[str, float]] = []
    all_h_histories: list[dict]             = []
    all_f_histories: list[dict]             = []
    h_losses: list[float] = []
    f_losses: list[float] = []

    # ── Multi-seed training loop ───────────────────────────────────────────────
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Run {run_idx + 1}/{args.n_seeds}  (seed={seed})")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        while True:
            h_dgp  = make_hierarchical_dgp(seed)
            f_dgp  = make_flat_dgp(seed)
            delta = _bayes_ce_hierarchical(h_dgp, 0)["H_query"] - _bayes_ce_hierarchical(h_dgp, h_dgp.num_segments-1)["H_query"]
            if delta < 0.08:
                print('Insufficeint improvment from context, trying new seed...')
                seed += 100
            else:
                break
        h_test = build_hierarchical_test_set(h_dgp, test_size)
        f_test = build_flat_test_set(f_dgp, test_size)

        train_kwargs = dict(
            num_batches      = steps,
            batch_size       = batch_size,
            lr               = cfg.experiment.lr,
            warmup_batches   = cfg.experiment.warmup_batches,
            use_cosine_decay = cfg.experiment.use_cosine_decay,
            eval_every       = eval_every,
            log_every        = 0,
        )

        print("  [hierarchical]")
        h_history = train(make_model(), h_dgp, test_sets=h_test, **train_kwargs)
        print("  [flat]")
        f_history = train(make_model(), f_dgp, test_sets=f_test, **train_kwargs)

        h_sum = summarise_run(h_history)
        f_sum = summarise_run(f_history)

        h_summaries.append(h_sum)
        f_summaries.append(f_sum)
        all_h_histories.append(h_history)
        all_f_histories.append(f_history)
        h_losses.append(h_history["test_loss_history"][-1])
        f_losses.append(f_history["test_loss_history"][-1])

        # print(
        #     f"  Hier: loss={h_losses[-1]:.4f}  "
        #     f"lw_hydra={h_sum['layerwise_hydra__final']:.4f}  "
        #     f"a5B={h_sum['a5B_cosine__final']:.4f}"
        # )
        # print(
        #     f"  Flat: loss={f_losses[-1]:.4f}  "
        #     f"lw_hydra={f_sum['layerwise_hydra__final']:.4f}  "
        #     f"a5B={f_sum['a5B_cosine__final']:.4f}"
        # )

    # ── Statistical tests ──────────────────────────────────────────────────────
    results_df = build_results_df(h_summaries, f_summaries, alpha=args.alpha)

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SIGNIFICANCE TEST RESULTS")
    print(f"  n_seeds={args.n_seeds}  steps={steps}  test_size={test_size}")
    print(f"  alpha={args.alpha}  "
          f"N_tests={N_TESTS_TOTAL}  "
          f"Bonferroni threshold={args.alpha / N_TESTS_TOTAL:.5f}")
    print(f"{'='*80}")

    display_cols = ["metric_label", "window", "direction",
                    "h_mean", "f_mean", "mean_diff", "cohens_d",
                    "t_pval_one", "w_pval_one", "sig_t", "sig_w"]
    print(results_df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    # Headline
    print(f"\n{'='*60}")
    print("HEADLINE: effects confirmed significant (one-sided t-test)")
    print(f"{'='*60}")
    sig_rows = results_df[results_df["sig_t"]]
    if sig_rows.empty:
        print("  None reached significance at this seed count / training budget.")
    else:
        for _, row in sig_rows.iterrows():
            direction_str = "hier > flat" if row["direction"] == "greater" else "hier < flat"
            bonf_tag      = "  [Bonf✓]" if row["sig_t_bonf"] else ""
            print(
                f"  {row['metric_label']:<44s} [{row['window']}]  "
                f"{direction_str}  "
                f"Δ={row['mean_diff']:+.4f}  d={row['cohens_d']:.2f}  "
                f"p_t={row['t_pval_one']:.4f}  p_w={row['w_pval_one']:.4f}"
                + bonf_tag
            )

    # Test loss context (not significance-tested: it reflects DGP complexity)
    print(f"\n  Test loss (context only, not significance-tested):")
    print(f"    Hier: {np.mean(h_losses):.4f} ± {np.std(h_losses, ddof=1):.4f}")
    print(f"    Flat: {np.mean(f_losses):.4f} ± {np.std(f_losses, ddof=1):.4f}")

    # Save CSV
    csv_path = os.path.join(args.out_dir, "significance_summary.csv")
    results_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nFull results saved to {csv_path}")

    # ── Figures ────────────────────────────────────────────────────────────────
    plot_distributions(h_summaries, f_summaries, results_df, args.out_dir, args.alpha)
    plot_training_curves(all_h_histories, all_f_histories, args.out_dir)
    plot_pvalue_heatmap(results_df, args.out_dir, args.alpha)
    plot_effect_sizes(results_df, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()