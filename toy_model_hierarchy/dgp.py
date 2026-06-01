"""
Synthetic Data Generation Processes for Latent Hierarchical Sequence Modelling.

Two classes are provided:
  - HierarchicalDGP  :  Z^(0) -> Z^(1) -> X   (full two-level hierarchy)
  - FlatDGP          :  Z^(1) -> X             (no global latent; fair ablation)

Design principles (rewrite)
---------------------------
Emissions are PARTIALLY disentangled:

    Evidence tokens : x_ev ~ Categorical(emit_ev[z1])              — Z^(1) only
    Query token     : x_q  ~ Categorical(emit_q[z0, z1])           — JOINT (Z^(0), Z^(1))

Evidence tokens carry clean Z^(1) signal only.  Query tokens carry a weighted
mixture of Z^(0) and Z^(1) signal, controlled by alpha_query:

    logit_q[z0, z1, x] = alpha_query * E0[z0, x] + (1 - alpha_query) * E1[z1, x]

This joint dependence at the query position is theoretically motivated:

  Assumption 2 (Hierarchical dependence) in the paper requires that the query
  token depend jointly on both latents.  This creates TWO disjoint evidence
  streams that the model must aggregate:

    Stream S1 (cross-segment): x_ev^(1..S) → Z^(1)^(1..S) → Z^(0) → x_q
    Stream S2 (within-segment): x_ev^(current) → Z^(1) → x_q

  Neither stream alone is sufficient to predict x_q optimally.  Gradient
  descent therefore specialises independent functional units to estimate each
  stream's partial posterior (Theorem 3), and ablating one unit causes the
  other to compensate — the Hydra effect (Theorem 4).

  With alpha_query = 1.0 (pure Z^(0)), only stream S1 matters; stream S2 is
  absent and no Hydra pressure develops.  With alpha_query ∈ (0.6, 0.8),
  both streams contribute meaningfully.

FlatDGP ablation:
  Query tokens drawn from the Z^(0)-marginalised table:
      emit_q_flat[z1, x] = Σ_{z0} pi0[z0] * emit_q[z0, z1, x]
  The flat model can still use within-segment Z^(1) signal to partially
  predict x_q, but loses the cross-segment Z^(0) information entirely.
  The Bayes gap = I(Z^(0); x_q | Z^(1), within-segment evidence) is
  guaranteed positive by the data-processing inequality.

The inference chain the model must learn:
    x_ev^(1..S) → Z^(1)^(1..S) → Z^(0) ─┐
                                           ├─→ x_q
    x_ev^(current) → Z^(1)_current  ──────┘

This is enabled by the structured pi1 transition: Z^(1)|Z^(0) is concentrated
on a subset of local states, so evidence tokens (which reveal Z^(1)) carry
indirect signal about Z^(0) through the transition table.

Segment structure
-----------------
Each segment of length L is split into:

    [ x_1  ...  x_{L-2} | [QUERY] | x_{L-1} ]
      ^-- evidence (L-2) -^    ^cue^  ^target^

  Evidence tokens  (positions 0 .. L-3):
      x_ev ~ Categorical(emit_ev[z1])
      Reveal Z^(1) but not Z^(0) directly.

  [QUERY] token  (position L-2):
      Deterministic special token (ID = vocab_size).
      Signals that the next token is a query.
      Masked out of the cross-entropy loss.

  Query token  (position L-1):
      x_q ~ Categorical(emit_q[z0, z1])
      Depends JOINTLY on Z^(0) and Z^(1), weighted by alpha_query.
      Predicting this well requires both cross-segment Z^(0) inference
      AND within-segment Z^(1) inference — two disjoint evidence streams.

Token-space convention
----------------------
  Content vocabulary   : token IDs  0 .. vocab_size-1
  [QUERY] special token: token ID   vocab_size
  Model embedding table must have vocab_size + 1 rows.

Loss mask
---------
  The dataloader returns loss_mask of shape (B, T-1).
  Positions where the TARGET is [QUERY] are False (masked); all others True.
  train.py uses this mask so [QUERY] never contributes to gradients.

"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along `axis`."""
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _safe_entropy(probs: np.ndarray) -> float:
    """
    H = -Σ p log p  (nats), computed safely (0 log 0 = 0).
    Operates on the last axis; returns a scalar when probs is 1-D.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(probs > 0, np.log(probs), 0.0)
    return float(-np.sum(probs * lp, axis=-1))


def _mi_discrete(joint: np.ndarray) -> float:
    """
    Mutual information I(A;B) from a normalised joint table P[a,b].
    Convention: 0 * log 0 = 0.
    """
    joint = joint / joint.sum()
    pa    = joint.sum(axis=1, keepdims=True)
    pb    = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / (pa * pb), 1.0)
        mi    = np.where(joint > 0, joint * np.log(ratio), 0.0)
    return float(mi.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class _BaseDGP:
    """
    Shared bookkeeping and dataloader logic.

    Subclasses must implement:
        _build_cpts()
        _sample_sequence(rng) -> (tokens, z0_or_None, z1_array)
    """

    def __init__(
        self,
        vocab_size:   int,
        seq_len:      int,
        num_segments: int,
        seed:         Optional[int] = None,
    ):
        if seq_len % num_segments != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by "
                f"num_segments ({num_segments})."
            )
        self.vocab_size       = vocab_size
        self.query_token_id   = vocab_size      # one past the content vocab
        self.model_vocab_size = vocab_size + 1  # embedding table size
        self.seq_len          = seq_len
        self.num_segments     = num_segments
        self.seg_len          = seq_len // num_segments
        self.seed             = seed

        if self.seg_len < 3:
            raise ValueError(
                f"seg_len must be >= 3 (need ≥1 evidence + [QUERY] + 1 query). "
                f"Got seg_len={self.seg_len}."
            )

        self._rng_cpt = np.random.default_rng(seed)
        self._build_cpts()
        self._rng_sample = np.random.default_rng(
            None if seed is None else seed + 1
        )

    def _build_cpts(self) -> None:
        raise NotImplementedError

    def _sample_sequence(self, rng: np.random.Generator) -> tuple:
        raise NotImplementedError

    # ── Loss mask ─────────────────────────────────────────────────────────────

    def _make_loss_mask(self, tokens: np.ndarray) -> np.ndarray:
        """
        Boolean mask of shape (N, seq_len-1) aligned with target_tokens.
        False where the TARGET is the [QUERY] cue token (never a loss target).
        """
        target_tokens = tokens[:, 1:]
        return target_tokens != self.query_token_id  # (N, T-1) bool

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, n: int = 1) -> dict:
        """
        Draw `n` independent sequences.

        Returns dict with keys:
            tokens     : (n, seq_len)   int32
            z0         : (n,)           int32  or None  (None for FlatDGP)
            z1         : (n, S)         int32
            loss_mask  : (n, seq_len-1) bool
        """
        all_tokens, all_z0, all_z1 = [], [], []
        for _ in range(n):
            tok, z0, z1 = self._sample_sequence(self._rng_sample)
            all_tokens.append(tok)
            all_z0.append(z0)
            all_z1.append(z1)

        tokens = np.stack(all_tokens).astype(np.int32)
        return {
            "tokens":    tokens,
            "z0":        None if all_z0[0] is None
                         else np.array(all_z0, dtype=np.int32),
            "z1":        np.stack(all_z1).astype(np.int32),
            "loss_mask": self._make_loss_mask(tokens),
        }

    # ── Dataloader ────────────────────────────────────────────────────────────

    def dataloader(
        self,
        batch_size:   int,
        num_batches:  Optional[int] = None,
        replacement:  bool          = True,
        dataset_size: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Yields batches for autoregressive training.  Each batch dict contains:

            input_tokens  : (B, T-1)   int32
            target_tokens : (B, T-1)   int32
            loss_mask     : (B, T-1)   bool
            z0            : (B,)       int32  or None
            z1            : (B, S)     int32
            batch_idx     : int
        """
        if replacement:
            yield from self._online(batch_size, num_batches)
        else:
            yield from self._offline(batch_size, num_batches, dataset_size)

    def _online(self, batch_size: int, num_batches: Optional[int]) -> Iterator[dict]:
        idx = 0
        while True:
            if num_batches is not None and idx >= num_batches:
                return
            yield self._make_batch(self.sample(batch_size), idx)
            idx += 1

    def _offline(
        self,
        batch_size:   int,
        num_batches:  Optional[int],
        dataset_size: Optional[int],
    ) -> Iterator[dict]:
        if dataset_size is None:
            dataset_size = 10 * batch_size

        dataset     = self.sample(dataset_size)
        idx         = 0
        batch_count = 0

        while True:
            if num_batches is not None and batch_count >= num_batches:
                return
            start, end = idx, idx + batch_size
            if end > dataset_size:
                perm    = self._rng_sample.permutation(dataset_size)
                dataset = {k: (v[perm] if v is not None else None)
                           for k, v in dataset.items()}
                idx, start, end = 0, 0, batch_size
            yield self._make_batch(
                {k: (v[start:end] if v is not None else None)
                 for k, v in dataset.items()},
                batch_count,
            )
            idx         += batch_size
            batch_count += 1

    @staticmethod
    def _make_batch(raw: dict, idx: int) -> dict:
        tokens = raw["tokens"]
        return {
            "input_tokens":  tokens[:, :-1],
            "target_tokens": tokens[:, 1:],
            "loss_mask":     raw["loss_mask"],
            "z0":            raw["z0"],
            "z1":            raw["z1"],
            "batch_idx":     idx,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HierarchicalDGP
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalDGP(_BaseDGP):
    """
    Two-level latent hierarchical DGP with partially disentangled emissions.

    Generative story
    ----------------
    1.  Z^(0)      ~ Categorical(pi0)                          global latent
    2.  Z^(1)_s    | Z^(0) ~ Categorical(pi1[Z^(0)])           local latent, segment s
    3a. x_t (ev)   | Z^(1)_s ~ Categorical(emit_ev[Z^(1)_s])  evidence: Z^(1) only
    3b. [QUERY]      : deterministic token ID = vocab_size
    3c. x_t (query) | Z^(0), Z^(1)_s
                     ~ Categorical(emit_q[Z^(0), Z^(1)_s])    query: JOINT (Z^(0), Z^(1))

    emit_q[z0, z1, x] = softmax(alpha_query * E0[z0] + (1 - alpha_query) * E1[z1])

    The joint query emission creates two disjoint evidence streams that the
    model must aggregate (satisfying Assumption 2/4 in the paper's Section 4),
    which drives specialisation of functional units (Theorem 3) and the
    Hydra effect (Theorem 4).

    The inference chain the model must learn:
        x_ev^(1..S) → Z^(1)^(1..S) → Z^(0) ─┐
                                               ├─→ x_q
        x_ev^(current) → Z^(1)_current  ──────┘

    Parameters
    ----------
    vocab_size         : V — content vocabulary size; [QUERY] gets ID V
    seq_len            : T — total tokens per sequence (must divide by num_segments)
    num_segments       : S — number of segments per sequence
    k0                 : K0 — number of global latent states
    k1                 : K1 — number of local latent states (must be >= k0)
    pi1_concentration  : Dirichlet concentration for pi1[z0]; higher = tighter
                         Z^(1)|Z^(0) clusters.  Recommend 10–50.
    embedding_scale    : controls separation between Z^(0) query clusters.
                         Larger = more separable = larger Bayes gap.
                         Recommend 3.0–6.0.
    embedding_noise    : std of noise added to build E1 from E0 parent.
                         Controls how much evidence tokens resemble their
                         parent query cluster.  Recommend 0.5–1.5.
    alpha_query        : mixing weight for Z^(0) in joint query emission.
                         alpha_query=1.0  → pure Z^(0), no Hydra pressure.
                         alpha_query=0.7  → strong Z^(0) with meaningful Z^(1)
                                            contribution; recommended for Hydra.
                         alpha_query=0.0  → pure Z^(1), no cross-segment signal.
                         Recommend 0.6–0.8.
    seed               : int or None
    z0_mask            : boolean array (K0,) restricting active Z^(0) states
    """

    def __init__(
        self,
        vocab_size:         int,
        seq_len:            int,
        num_segments:       int,
        k0:                 int                  = 4,
        k1:                 int                  = 16,
        pi1_concentration:  float                = 20.0,
        embedding_scale:    float                = 4.0,
        embedding_noise:    float                = 1.0,
        alpha_query:        float                = 0.7,
        seed:               Optional[int]        = None,
        z0_mask:            Optional[np.ndarray] = None,
    ):
        if k1 < k0:
            raise ValueError(f"k1 ({k1}) must be >= k0 ({k0}).")
        if pi1_concentration < 1.0:
            raise ValueError("pi1_concentration must be >= 1.0.")
        if embedding_scale <= 0:
            raise ValueError("embedding_scale must be > 0.")
        if embedding_noise < 0:
            raise ValueError("embedding_noise must be >= 0.")
        if not (0.0 < alpha_query <= 1.0):
            raise ValueError("alpha_query must be in (0, 1].")

        self.k0                = k0
        self.k1                = k1
        self.pi1_concentration = pi1_concentration
        self.embedding_scale   = embedding_scale
        self.embedding_noise   = embedding_noise
        self.alpha_query       = alpha_query
        self.z0_mask           = z0_mask

        super().__init__(vocab_size, seq_len, num_segments, seed)

    # ── CPT construction ─────────────────────────────────────────────────────

    def _build_cpts(self) -> None:
        rng = self._rng_cpt
        V   = self.vocab_size

        # ── p(Z^(0)) ─────────────────────────────────────────────────────────
        pi0_full      = rng.dirichlet(np.ones(self.k0))
        self.pi0_full = pi0_full

        if self.z0_mask is not None:
            mask     = np.asarray(self.z0_mask, dtype=bool)
            masked   = pi0_full * mask
            self.pi0 = masked / masked.sum()
        else:
            self.pi0 = pi0_full.copy()

        # ── p(Z^(1) | Z^(0)) ─────────────────────────────────────────────────
        # Partition k1 local states into k0 contiguous groups.
        # Each z0 gets pi1_concentration on its own group, 1.0 elsewhere.
        group_size  = self.k1 // self.k0
        conc_matrix = np.ones((self.k0, self.k1), dtype=float)
        for z0 in range(self.k0):
            lo = z0 * group_size
            hi = (z0 + 1) * group_size if z0 < self.k0 - 1 else self.k1
            conc_matrix[z0, lo:hi] = self.pi1_concentration
        self.pi1 = np.stack([
            rng.dirichlet(conc_matrix[z0]) for z0 in range(self.k0)
        ])  # (K0, K1)

        # ── Block-sparse embedding E0: (K0, V) ───────────────────────────────
        # Each Z^(0) state gets high logit on its own contiguous vocabulary block.
        # embedding_scale controls the logit gap → cluster sharpness.
        # This guarantees clearly distinct per-Z^(0) distributions regardless
        # of random seed, giving a reliable, large Bayes gap.
        block_v = V // self.k0
        E0      = np.full((self.k0, V), -self.embedding_scale)
        for z0 in range(self.k0):
            lo = z0 * block_v
            hi = (z0 + 1) * block_v if z0 < self.k0 - 1 else V
            E0[z0, lo:hi] = +self.embedding_scale
        self.E0 = E0                                           # (K0, V)

        # ── Evidence embedding E1: (K1, V) ───────────────────────────────────
        # E1[z1] = E0[parent(z1)] + embedding_noise * N(0, I)
        # parent(z1) = z1 // group_size
        # Evidence tokens cluster in the same vocabulary region as their parent
        # Z^(0) state's query tokens, giving the model a geometric inference path:
        #   "which vocab cluster do these evidence tokens come from?"
        #   → which Z^(1) group → which Z^(0) state → predict query token.
        self.z1_parents = np.array(
            [z1 // group_size for z1 in range(self.k1)], dtype=np.int32
        )
        noise    = rng.standard_normal((self.k1, V))
        E1       = self.E0[self.z1_parents] + self.embedding_noise * noise
        self.E1      = E1                                      # (K1, V)
        self.emit_ev = _softmax(E1, axis=-1)                  # (K1, V)

        # ── Query emission: emit_q[z0, z1, x] = p(x_query | Z^(0)=z0, Z^(1)=z1) ──
        #
        # JOINT dependence on both latents, weighted by alpha_query:
        #   logit[z0, z1, x] = alpha_query * E0[z0, x] + (1-alpha_query) * E1[z1, x]
        #
        # This satisfies Assumption 2 (hierarchical dependence) in the paper:
        # the query token jointly depends on both levels of the hierarchy.
        # It creates two disjoint evidence streams:
        #   S1 (cross-segment): accumulated evidence → Z^(0) → x_q contribution
        #   S2 (within-segment): current evidence → Z^(1) → x_q contribution
        # Both streams are needed for optimal prediction, driving specialisation
        # and ultimately the Hydra effect (Theorems 3 & 4).
        #
        # alpha_query=1.0 collapses to pure Z^(0) (no Hydra);
        # alpha_query∈(0.6,0.8) balances both streams (recommended).
        logits_q = (
              self.alpha_query       * E0[:, np.newaxis, :]   # (K0,  1, V)
            + (1 - self.alpha_query) * E1[np.newaxis, :, :]   # ( 1, K1, V)
        )
        self.emit_q = _softmax(logits_q, axis=-1)             # (K0, K1, V)

        # ── Vocabulary permutation (for OOD test sets) ────────────────────────
        self._vocab_perm: Optional[np.ndarray] = None

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _sample_sequence(self, rng: np.random.Generator) -> tuple:
        z0   = int(rng.choice(self.k0, p=self.pi0))
        n_ev = self.seg_len - 2
        tokens = np.empty(self.seg_len * self.num_segments, dtype=np.int32)
        z1s    = np.empty(self.num_segments, dtype=np.int32)

        for s in range(self.num_segments):
            z1     = int(rng.choice(self.k1, p=self.pi1[z0]))
            z1s[s] = z1
            start  = s * self.seg_len

            # Evidence: Z^(1) only
            ev_tokens = rng.choice(self.vocab_size, size=n_ev, p=self.emit_ev[z1])

            # [QUERY] cue: deterministic structural token
            query_cue = np.array([self.query_token_id], dtype=np.int32)

            # Query: JOINT Z^(0) and Z^(1)
            q_token = np.array(
                [rng.choice(self.vocab_size, p=self.emit_q[z0, z1])], dtype=np.int32
            )

            seg = np.concatenate([ev_tokens, query_cue, q_token])

            if self._vocab_perm is not None:
                seg = seg.copy()
                content_mask      = seg != self.query_token_id
                seg[content_mask] = self._vocab_perm[seg[content_mask]]

            tokens[start : start + self.seg_len] = seg

        return tokens, z0, z1s

    # ── Mutual information diagnostics ───────────────────────────────────────

    def mutual_information(self) -> dict[str, float]:
        """
        Analytical MI quantities for the DGP.

        I_Z0_Xq  : I(Z^(0); x_query) — marginalised over Z^(1).
                   Measures how much cross-segment inference is worth.
        I_Z1_Xev : I(Z^(1); x_ev)    per evidence token (evidence is pure Z^(1))
        I_Z0_Z1  : I(Z^(0); Z^(1))   — bottleneck for cross-segment inference.
                   If small, evidence tokens carry little Z^(0) signal.
        I_S1_S2  : I(x_query_S1; x_query_S2) — cross-segment query MI.
                   Zero for FlatDGP; positive here.
                   This is the observable signature of the hierarchy.
        """
        # Marginal Z^(1) distribution
        pi1_marginal = (self.pi0_full[:, None] * self.pi1).sum(0)  # (K1,)

        # p(x_q | z0) = Σ_{z1} pi1[z0,z1] * emit_q[z0,z1,x]       (K0, V)
        p_xq_given_z0 = np.einsum("zv,zvx->zx", self.pi1, self.emit_q)

        # I(Z^(0); x_query): joint p(z0, x_q) = pi0[z0] * p(xq|z0)
        joint_q = self.pi0[:, None] * p_xq_given_z0                # (K0, V)
        I_Z0_Xq = _mi_discrete(joint_q)

        # I(Z^(1); x_ev): joint p(z1, x_ev) = pi1_marginal[z1] * emit_ev[z1, x]
        joint_ev = pi1_marginal[:, None] * self.emit_ev             # (K1, V)
        I_Z1_Xev = _mi_discrete(joint_ev)

        # I(Z^(0); Z^(1)): joint p(z0, z1) = pi0[z0] * pi1[z0, z1]
        joint_z0z1 = self.pi0[:, None] * self.pi1                   # (K0, K1)
        I_Z0_Z1    = _mi_discrete(joint_z0z1)

        # I(x_q_S1; x_q_S2): cross-segment query MI
        # p(xq1, xq2) = Σ_{z0} pi0[z0] * p(xq1|z0) * p(xq2|z0)
        joint_xq12 = np.einsum(
            "z,zv,zu->vu", self.pi0, p_xq_given_z0, p_xq_given_z0
        )
        I_S1_S2 = _mi_discrete(joint_xq12)

        return {
            "I_Z0_Xq":   I_Z0_Xq,
            "I_Z1_Xev":  I_Z1_Xev,
            "I_Z0_Z1":   I_Z0_Z1,
            "I_S1_S2":   I_S1_S2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FlatDGP
# ─────────────────────────────────────────────────────────────────────────────

class FlatDGP(_BaseDGP):
    """
    Flat (no global latent, no block structure) DGP.

    This is a clean structural baseline — not a marginalisation of
    HierarchicalDGP, but an independently parametrised process with the
    same surface dimensions (vocab_size, seq_len, num_segments, k1).

    Generative story
    ----------------
    Per sequence:
        Sample S distinct Z^(1) states (without replacement from K1 states,
        using a uniform permutation).  This guarantees that within a sequence
        each segment uses a different local latent, but across sequences the
        latents are i.i.d. and carry zero cross-sequence correlation.

    Per segment s (with latent Z^(1)_s):
        x_ev   | Z^(1)_s ~ Categorical(emit_ev_flat[Z^(1)_s])
        [QUERY]            deterministic
        x_q    | Z^(1)_s ~ Categorical(emit_q_flat[Z^(1)_s])

    Key design decisions
    --------------------
    1.  No Z^(0) at all.  There is no global latent tying segments together.
        Cross-segment correlation is exactly zero — the uniqueness constraint
        is a within-sequence diversity device, not a source of cross-segment
        information about any shared latent.

    2.  Independently drawn embeddings.  E1_flat is drawn fresh from
        N(0, I), with NO block/parent structure inherited from HierarchicalDGP.
        This gives a clean separation: any difference in model behaviour
        between hierarchical and flat training is attributable to the presence
        or absence of the global latent Z^(0), not to differences in the
        embedding geometry.

    3.  Unique-Z^(1) sampling.  Within each sequence, the S local latents
        are a random subset of size S drawn without replacement from {0..K1-1}
        (equivalently: a random permutation of K1 states, first S taken).
        This preserves a uniform marginal over K1 per segment-position while
        ensuring no two segments in the same sequence share a latent.
        Across sequences the draws are independent.

    Parameters
    ----------
    vocab_size    : V  — must match HierarchicalDGP for fair comparison
    seq_len       : T
    num_segments  : S  — must satisfy S <= K1
    k1            : K1 — number of local latent states
    embedding_scale : logit scale for emit_ev_flat and emit_q_flat
    seed          : int or None
    """

    def __init__(
        self,
        vocab_size:      int,
        seq_len:         int,
        num_segments:    int,
        k1:              int   = 16,
        embedding_scale: float = 4.0,
        seed:            Optional[int] = None,
    ):
        if num_segments > k1:
            raise ValueError(
                f"num_segments ({num_segments}) must be <= k1 ({k1}) "
                f"to allow unique Z^(1) draws per sequence."
            )
        if embedding_scale <= 0:
            raise ValueError("embedding_scale must be > 0.")

        self.k1              = k1
        self.k0              = None   # no global latent
        self.embedding_scale = embedding_scale

        super().__init__(vocab_size, seq_len, num_segments, seed)

    def _build_cpts(self) -> None:
        rng = self._rng_cpt
        V   = self.vocab_size

        # ── Flat evidence embeddings: (K1, V) ────────────────────────────────
        # Drawn independently from N(0, I) — no block/parent structure.
        # Each Z^(1) state defines its own arbitrary vocabulary cluster.
        E1_ev        = rng.standard_normal((self.k1, V)) * self.embedding_scale
        self.E1_ev   = E1_ev
        self.emit_ev_flat = _softmax(E1_ev, axis=-1)             # (K1, V)

        # ── Flat query embeddings: (K1, V) ───────────────────────────────────
        # Also drawn independently — no relationship to E1_ev or to Z^(0).
        # The query token reveals only Z^(1); no cross-segment inference helps.
        E1_q         = rng.standard_normal((self.k1, V)) * self.embedding_scale
        self.E1_q    = E1_q
        self.emit_q_flat  = _softmax(E1_q, axis=-1)              # (K1, V)

        # ── Uniform marginal (for Bayes CE) ──────────────────────────────────
        # Under the unique-permutation sampling, the marginal of Z^(1) at any
        # segment position is uniform over K1 states.
        self.pi1_marginal = np.ones(self.k1) / self.k1           # (K1,)

    def _sample_sequence(self, rng: np.random.Generator) -> tuple:
        n_ev   = self.seg_len - 2
        tokens = np.empty(self.seg_len * self.num_segments, dtype=np.int32)

        # Sample S unique Z^(1) states: random permutation, take first S.
        z1s = rng.permutation(self.k1)[: self.num_segments].astype(np.int32)

        for s in range(self.num_segments):
            z1    = int(z1s[s])
            start = s * self.seg_len

            # Evidence: flat Z^(1)-only emission
            ev_tokens = rng.choice(
                self.vocab_size, size=n_ev, p=self.emit_ev_flat[z1]
            )
            query_cue = np.array([self.query_token_id], dtype=np.int32)

            # Query: flat Z^(1)-only emission (independent table from evidence)
            q_token = np.array(
                [rng.choice(self.vocab_size, p=self.emit_q_flat[z1])],
                dtype=np.int32,
            )
            tokens[start : start + self.seg_len] = np.concatenate(
                [ev_tokens, query_cue, q_token]
            )

        return tokens, None, z1s

    def mutual_information(self) -> dict[str, float]:
        """
        I_Z0_Xq  : None  (no global latent)
        I_Z1_Xev : I(Z^(1); x_ev) under the uniform marginal
        I_Z0_Z1  : None
        I_S1_S2  : 0.0 exactly — Z^(1) states are drawn independently across
                   sequences (uniqueness is within-sequence only), so query
                   tokens carry zero cross-segment mutual information.
        """
        joint_ev = self.pi1_marginal[:, None] * self.emit_ev_flat  # (K1, V)
        return {
            "I_Z0_Xq":  None,
            "I_Z1_Xev": _mi_discrete(joint_ev),
            "I_Z0_Z1":  None,
            "I_S1_S2":  0.0,   # exact: no shared latent across segments
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test-set factories (separate per DGP type)
# ─────────────────────────────────────────────────────────────────────────────

def build_hierarchical_test_set(
    train_dgp: HierarchicalDGP,
    test_size: int,
) -> dict:
    """
    Build an in-distribution test set for HierarchicalDGP.

    Uses the same CPTs as train_dgp but a fresh sampling RNG (seed+50),
    so sequences are independent of those seen during training.

    Returns a sample dict: tokens, z0, z1, loss_mask, meta.
    """
    seed = train_dgp.seed
    dgp  = HierarchicalDGP(
        vocab_size        = train_dgp.vocab_size,
        seq_len           = train_dgp.seq_len,
        num_segments      = train_dgp.num_segments,
        k0                = train_dgp.k0,
        k1                = train_dgp.k1,
        pi1_concentration = train_dgp.pi1_concentration,
        embedding_scale   = train_dgp.embedding_scale,
        embedding_noise   = train_dgp.embedding_noise,
        alpha_query       = train_dgp.alpha_query,
        seed              = seed,
    )
    dgp._rng_sample = np.random.default_rng(
        None if seed is None else seed + 50
    )
    data = dgp.sample(test_size)
    data["meta"] = {
        "name":        "Hierarchical — in-distribution",
        "description": "Same CPTs as hierarchical training DGP; fresh samples.",
        "dgp_type":    "hierarchical",
        "k0":          train_dgp.k0,
        "k1":          train_dgp.k1,
    }
    return data


def build_flat_test_set(
    train_flat_dgp: FlatDGP,
    test_size:      int,
) -> dict:
    """
    Build an in-distribution test set for FlatDGP.

    Uses the same CPTs as train_flat_dgp but a fresh sampling RNG (seed+50).

    Returns a sample dict: tokens, z0 (None), z1, loss_mask, meta.
    """
    seed = train_flat_dgp.seed
    dgp  = FlatDGP(
        vocab_size      = train_flat_dgp.vocab_size,
        seq_len         = train_flat_dgp.seq_len,
        num_segments    = train_flat_dgp.num_segments,
        k1              = train_flat_dgp.k1,
        embedding_scale = train_flat_dgp.embedding_scale,
        seed            = seed,
    )
    dgp._rng_sample = np.random.default_rng(
        None if seed is None else seed + 50
    )
    data = dgp.sample(test_size)
    data["meta"] = {
        "name":        "Flat — in-distribution",
        "description": (
            "Independent Z^(1) states per segment (unique within sequence), "
            "no global latent, independent embeddings."
        ),
        "dgp_type":    "flat",
        "k1":          train_flat_dgp.k1,
    }
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Bayes-optimal cross-entropy
# ─────────────────────────────────────────────────────────────────────────────

def bayes_optimal_cross_entropy(
    dgp:              "_BaseDGP",
    context_segments: int = 0,
) -> dict[str, float]:
    """
    Compute the Bayes-optimal cross-entropy for a given DGP.

    HierarchicalDGP
    ---------------
    Evidence tokens:
        p(x_ev | context) = Σ_{z1} p(z1|z0_posterior) * emit_ev[z1, x]
        The Z^(0) posterior sharpens across segments, so evidence CE
        decreases as context_segments grows.

    Query token:
        p(x_q | context, z1) = Σ_{z0,z1} post[z0] * p(z1|context) * emit_q[z0,z1,x]
        Depends jointly on both latents; benefits from cross-segment Z^(0)
        inference AND within-segment Z^(1) inference.

    FlatDGP
    -------
    Evidence tokens:
        p(x_ev) = (1/K1) Σ_{z1} emit_ev_flat[z1, x]
        No context helps — Z^(1) states are i.i.d. across sequences.
        Within a sequence they are unique but still independent of each other.

    Query token:
        p(x_q | z1) = emit_q_flat[z1, x]
        Only within-segment Z^(1) signal; no cross-segment inference possible.
        Marginal (before within-segment update):
            p(x_q) = (1/K1) Σ_{z1} emit_q_flat[z1, x]

    Parameters
    ----------
    dgp              : HierarchicalDGP or FlatDGP
    context_segments : preceding complete segments (HierarchicalDGP only)

    Returns
    -------
    dict: H_evidence, H_query, H_total (nats + bits), post_z0, context_segments
    """
    if isinstance(dgp, HierarchicalDGP):
        return _bayes_ce_hierarchical(dgp, context_segments)
    elif isinstance(dgp, FlatDGP):
        return _bayes_ce_flat(dgp)
    else:
        raise TypeError(f"Unsupported DGP type: {type(dgp)}")


# ─────────────────────────────────────────────────────────────────────────────
# Bayes CE internals
# ─────────────────────────────────────────────────────────────────────────────

def _posterior_z0_after_s_segments(dgp: HierarchicalDGP, s: int) -> np.ndarray:
    """
    Compute E[p(Z^(0) | s complete segments)] analytically.

    With the joint query emission emit_q[z0, z1, x]:
        - Evidence tokens x_ev ~ emit_ev[z1], z1 ~ pi1[z0]
        - Query token     x_q  ~ emit_q[z0, z1]  (joint)

    The expected log-likelihood of one full segment given z0 is:

        ell(z0) = Σ_{z1} pi1[z0,z1] * [
                    n_ev * Σ_x emit_ev[z1,x] * log emit_ev[z1,x]      (evidence)
                  + Σ_x emit_q[z0,z1,x] * log emit_q[z0,z1,x]         (query, joint)
                  ]
               = -Σ_{z1} pi1[z0,z1] * n_ev * H_ev[z1]
                 - Σ_{z1} pi1[z0,z1] * H_q[z0,z1]

    where H_ev[z1]    = H(emit_ev[z1])
          H_q[z0, z1] = H(emit_q[z0, z1])

    The posterior after s segments:
        log p(z0 | s segments) ∝ log pi0[z0] + s * ell(z0)

    Returns (K0,) normalised posterior.
    """
    n_ev = dgp.seg_len - 2

    # H_ev[z1]: entropy of evidence emission                          (K1,)
    H_ev = np.array([_safe_entropy(dgp.emit_ev[z1]) for z1 in range(dgp.k1)])

    # H_q[z0, z1]: entropy of joint query emission                    (K0, K1)
    H_q = np.array([
        [_safe_entropy(dgp.emit_q[z0, z1]) for z1 in range(dgp.k1)]
        for z0 in range(dgp.k0)
    ])

    # Expected evidence log-likelihood per segment given z0
    ell_ev = -n_ev * dgp.pi1.dot(H_ev)          # (K0,)

    # Expected query log-likelihood per segment given z0
    # Σ_{z1} pi1[z0,z1] * (-H_q[z0,z1])
    ell_q = -(dgp.pi1 * H_q).sum(axis=1)        # (K0,)

    ell = ell_ev + ell_q                         # (K0,)

    log_post = np.log(dgp.pi0 + 1e-300) + s * ell
    log_post -= log_post.max()
    post      = np.exp(log_post)
    post     /= post.sum()
    return post                                  # (K0,)


def _bayes_ce_hierarchical(dgp: HierarchicalDGP, context_segments: int) -> dict:
    """
    Bayes-optimal CE for HierarchicalDGP with joint query emission.

    Evidence token CE:
        p(z1 | z0_posterior) = Σ_{z0} post[z0] * pi1[z0, z1]   (K1,)
        p(x_ev | context)    = Σ_{z1} p(z1|post) * emit_ev[z1, x]

    Query token CE:
        The query depends jointly on (Z^(0), Z^(1)).  After the current
        segment's evidence, the Bayes predictor has a posterior over both.

        We compute the expected updated Z^(0) posterior after observing Z^(1)
        (marginalised over Z^(1) uncertainty), then compute:

        p(x_q | context, segment_ev) =
            Σ_{z0,z1} E_post[z0] * p(z1|context) * emit_q[z0, z1, x]

        where E_post[z0] is the expected Z^(0) posterior after the Z^(1)
        update, and p(z1|context) is the marginal Z^(1) distribution.
        These are treated as approximately independent for the expectation.
    """
    n_ev    = dgp.seg_len - 2
    seg_len = dgp.seg_len

    # Z^(0) posterior from context_segments complete preceding segments
    post_z0 = _posterior_z0_after_s_segments(dgp, context_segments)  # (K0,)

    # ── Evidence CE ──────────────────────────────────────────────────────────
    p_z1_given_context = post_z0 @ dgp.pi1           # (K1,)
    p_ev = p_z1_given_context @ dgp.emit_ev           # (V,)
    H_evidence = _safe_entropy(p_ev)

    # ── Query CE ─────────────────────────────────────────────────────────────
    # Updated Z^(0) posterior after seeing current segment's Z^(1):
    # For each z1: updated_post[z1, z0] ∝ post_z0[z0] * pi1[z0, z1]
    updated_unnorm = post_z0[np.newaxis, :] * dgp.pi1.T   # (K1, K0)
    row_sums       = updated_unnorm.sum(axis=1, keepdims=True)
    row_sums       = np.where(row_sums > 0, row_sums, 1.0)
    updated_post   = updated_unnorm / row_sums               # (K1, K0)

    # Expected updated z0 posterior: Σ_{z1} p(z1|context) * updated_post[z1]
    expected_post_z0 = p_z1_given_context @ updated_post    # (K0,)

    # Predictive query distribution with joint emit_q[z0, z1, x]:
    # p(x_q) = Σ_{z0,z1} expected_post_z0[z0] * p_z1_given_context[z1] * emit_q[z0,z1,x]
    p_q = np.einsum(
        "z,v,zvx->x",
        expected_post_z0,
        p_z1_given_context,
        dgp.emit_q,
    )                                                        # (V,)
    H_query = _safe_entropy(p_q)

    H_total = (n_ev * H_evidence + 0.0 + H_query) / seg_len

    return {
        "H_evidence":       H_evidence,
        "H_query":          H_query,
        "H_total":          H_total,
        "H_evidence_bits":  H_evidence / np.log(2),
        "H_query_bits":     H_query    / np.log(2),
        "H_total_bits":     H_total    / np.log(2),
        "post_z0":          post_z0,
        "context_segments": context_segments,
    }


def _bayes_ce_flat(dgp: FlatDGP) -> dict:
    """
    Bayes-optimal CE for the new FlatDGP.

    Evidence tokens:
        Uniform marginal over K1 states (unique-permutation sampling gives
        uniform marginal at each segment position).
        p(x_ev) = (1/K1) Σ_{z1} emit_ev_flat[z1, x]
        No context helps — Z^(1) draws are independent across sequences,
        and uniqueness within a sequence provides no cross-segment signal
        (knowing z1_s tells you nothing about z1_{s'} for s' ≠ s, because
        the joint distribution is a uniform permutation, not correlated).

    Query token:
        p(x_q | z1) = emit_q_flat[z1, x]  (independent of evidence table)
        Marginal before within-segment update:
        p(x_q) = (1/K1) Σ_{z1} emit_q_flat[z1, x]
        No cross-segment inference possible; H_query is a constant.
    """
    n_ev    = dgp.seg_len - 2
    seg_len = dgp.seg_len

    # Evidence CE: uniform marginal over K1 flat evidence states
    p_ev       = dgp.emit_ev_flat.mean(axis=0)           # (V,)
    H_evidence = _safe_entropy(p_ev)

    # Query CE: uniform marginal over K1 flat query states
    p_q     = dgp.emit_q_flat.mean(axis=0)               # (V,)
    H_query = _safe_entropy(p_q)

    H_total = (n_ev * H_evidence + 0.0 + H_query) / seg_len

    return {
        "H_evidence":       H_evidence,
        "H_query":          H_query,
        "H_total":          H_total,
        "H_evidence_bits":  H_evidence / np.log(2),
        "H_query_bits":     H_query    / np.log(2),
        "H_total_bits":     H_total    / np.log(2),
        "post_z0":          None,
        "context_segments": None,
    }
