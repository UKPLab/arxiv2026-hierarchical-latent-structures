from __future__ import annotations
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    vocab_size:   int   = 64
    seq_len:      int   = 128
    num_segments: int   = 16
    alpha_query:  float = 0.8    # mixing weight for Z^(0) in query emission
    K0:           int   = 4      # number of global latent states
    K1:           int   = 16     # number of local latent states

    # HierarchicalDGP-specific
    pi1_concentration: float = 30.0
    embedding_noise:   float = 0.1


@dataclass
class ModelConfig:
    d_model:    int = 64
    num_layers: int = 6
    num_heads:  int = 8


@dataclass
class ExperimentConfig:
    test_size:  int = 512
    steps:      int = 5000
    batch_size: int = 64
    seed:       int = 115

    # Training schedule
    lr:               float = 3e-4
    warmup_batches:   int   = 50
    use_cosine_decay: bool  = False

    eval_every: int = 100
    log_every:  int = 0


@dataclass
class SignificanceConfig:
    """Settings for significance_test.py."""
    n_seeds:    int  = 10       # independent runs per DGP type
    fast_mode:  bool = False    # if True, shrink steps/test_size for smoke tests


# ─────────────────────────────────────────────────────────────────────────────
# Top-level config (composed)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    data:         DataConfig        = field(default_factory=DataConfig)
    model:        ModelConfig       = field(default_factory=ModelConfig)
    experiment:   ExperimentConfig  = field(default_factory=ExperimentConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)


# Default global instance — scripts import and read from this directly,
# or replace fields before use.
cfg = Config()
