import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

font_path = 'times new roman.ttf'
try:
    fm.fontManager.addfont(font_path)

    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()

    plt.rcParams['font.family'] = font_name
except FileNotFoundError:
    plt.rcParams['font.family'] = 'serif'


def get_available_font(preferred_font="Times New Roman"):
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    if preferred_font in available_fonts:
        return preferred_font

    fallbacks = ["DejaVu Serif", "XCharter", "serif"]
    for fallback in fallbacks:
        if fallback in available_fonts:
            return fallback

    return "serif"


PAPER_WIDTH_CM = 18.03
PAPER_WIDTH_IN = PAPER_WIDTH_CM / 2.54
GOLDEN_RATIO = 1.618
PAPER_HEIGHT_IN = PAPER_WIDTH_IN / GOLDEN_RATIO

SLIDE_WIDTH_IN = 11.5
SLIDE_HEIGHT_IN = 6.46875

PALETTE = sns.color_palette("Paired", n_colors=20)
COLOR_NGRAM = PALETTE[0]
COLOR_PCFG = PALETTE[2]
COLOR_PCFG_REC = PALETTE[4]


def setup_plotting_style(
    style="presentation", font_scale=1.35, font_family="Times New Roman"
):
    font_family = get_available_font(font_family)

    sns.set_theme(style="whitegrid", font_scale=font_scale, palette="tab10")

    if style == "presentation":
        plt.rcParams.update(
            {
                "lines.solid_capstyle": "projecting",
                "axes.unicode_minus": False,
                "font.family": font_family,
                "font.size": 14,
                "font.weight": "semibold",
                "axes.titleweight": "semibold",
                "axes.labelweight": "semibold",
                "axes.linewidth": 2.2,
                "xtick.major.width": 2.0,
                "ytick.major.width": 2.0,
                "xtick.minor.width": 1.6,
                "ytick.minor.width": 1.6,
            }
        )
    elif style == "paper":
        plt.rcParams.update(
            {
                "font.family": font_family,
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 12,
                "lines.linewidth": 2.0,
                "lines.solid_capstyle": "projecting",
                "axes.unicode_minus": False,
                "axes.linewidth": 1.0,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.minor.width": 0.6,
                "ytick.minor.width": 0.6,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linewidth": 0.5,
            }
        )
    else:
        raise ValueError(f"Unknown style: {style}. Choose 'presentation' or 'paper'.")


def style_ax(ax, xlabel="Step", ylabel="Loss", style="presentation"):
    if style == "presentation":
        ax.grid(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        for s in ax.spines.values():
            s.set_color("black")
            s.set_linewidth(2.2)
        ax.tick_params(
            axis="both",
            which="major",
            pad=0,
            colors="0.4",
            labelsize=14,
            length=8,
            width=2.0,
        )
        ax.tick_params(axis="both", which="minor", length=5, width=1.6)
        ax.set_xlabel(xlabel, fontsize=16, labelpad=8, weight="semibold")
        ax.set_ylabel(ylabel, fontsize=16, labelpad=8, weight="semibold")
    elif style == "paper":
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", which="major", length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", length=2, width=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        raise ValueError(f"Unknown style: {style}. Choose 'presentation' or 'paper'.")


def format_steps_k(value, _):
    return f"{int(value/1000)}k"


def get_step_formatter():
    return FuncFormatter(format_steps_k)


def save_figure(fig, filename, dpi=1200, bbox_inches="tight"):
    if filename.endswith(".svg"):
        fmt = "svg"
    elif filename.endswith(".pdf"):
        fmt = "pdf"
    elif filename.endswith(".png"):
        fmt = "png"
    else:
        fmt = "svg"

    figures_dir = Path("data/results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / filename
    fig.savefig(output_path, format=fmt, bbox_inches=bbox_inches, dpi=dpi)


def decaying_noise(
    n, base=0.25, final=0.02, spikes_rate=0.02, spike_scale=0.35, rng=None
):
    if rng is None:
        rng = np.random.default_rng(42)

    std = np.linspace(base, final, n)
    eps = rng.normal(0.0, std)
    spikes = rng.random(n) < spikes_rate
    eps[spikes] += rng.exponential(spike_scale, spikes.sum())
    return eps


def add_autocorr(x, strength=0.65):
    y = np.copy(x)
    for i in range(1, len(y)):
        y[i] = strength * y[i - 1] + (1 - strength) * y[i]
    return y


def median_filter_1d(x, k=3):
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i : i + k])
    return out
