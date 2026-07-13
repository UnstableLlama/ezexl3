# Design note — EMA checkpoint blending ("rolling persona taps")

**Status:** shape only, deliberately unbuilt. Depends on online/interactive
preference training (pineapple proposal #4, also deferred). Captured
2026-07-12 during the preference-training design discussion.

## The idea (UnstableLlama)

Instead of accumulating discrete `/save` checkpoints during long-running
preference training, maintain a small fixed set of **rolling EMA taps** at
different timescales, and expose the blend weights as **inference
hyperparameters**:

```
serve( 25% base  +  25% EMA_month  +  25% EMA_week  +  25% today )
```

The blend is a recency knob: the slow tap holds the stable long-term
persona, the fast tap carries recent teaching, the base anchor guards
general capability.

## Why this is sound (prior art)

- **EMA / Polyak averaging / SWA** — weight averaging along a training
  trajectory beats last-checkpoint on noisy objectives; hand-labeled
  preference data is maximally noisy.
- **WiSE-FT** (Wortsman et al. 2021) — linear interpolation between a
  fine-tuned model and its base recovers general/robust capability while
  retaining most of the tuning. The `25% base` term is exactly this.
- **Model soups** — averaging checkpoints from one trajectory is the
  best-behaved case of weight merging (no permutation misalignment).
- **Multi-timescale EMA** — standard in other domains (fast/slow EMA
  crossovers); here it is a low-pass filter bank over the training
  trajectory.

## The load-bearing fact

The fork's inference LoRA (`exllamav3/model/lora.py`) already supports
**multiple simultaneous adapters, each with its own `lora_scaling`**.
Because checkpoints are LoRA adapters over a frozen base:

```
W_effective = W_base + 0.25·Δ_month + 0.25·Δ_week + 0.25·Δ_today
```

"25 % base" is not a fourth artifact — it is the delta scales summing to
0.75. So the serve-side blend needs **no merging, no model surgery**: load
N adapter snapshots with scales. Different blends are load-time (or
potentially runtime) hyperparameters. This works today.

## Storage model — answers "checkpoints get messy"

Rotating-tap scheme, RRD-style. Storage is O(taps), not O(training steps):

| tap        | update rule                                | half-life  |
|------------|--------------------------------------------|------------|
| `live`     | the training state itself                  | —          |
| `ema_fast` | `ema ← (1−λ_f)·ema + λ_f·live` per step    | ~hours–day |
| `ema_slow` | `ema ← (1−λ_s)·ema + λ_s·live` per step    | ~week      |
| `ema_glacial` | same, smaller λ                         | ~month     |

Each tap is one adapter-sized file (MBs at typical rank), periodically
snapshotted to disk. "1 week ago" above really means "EMA with ~1-week
half-life", which is better than a literal point-in-time snapshot (it is
itself an average, hence smoother).

## The two real design decisions

### 1. Factor-space vs delta-space EMA

An adapter is `Δ = B·A` (rank r). EMA on `A` and `B` **separately** is not
EMA on `Δ` (the product is bilinear, not linear, in the factors).

- **Factor-space** (EMA the A/B matrices): cheap, same-rank output,
  standard practice in diffusion-LoRA trainers. Valid ONLY within one
  continuous training run (factors evolve smoothly from a shared init).
  **Breaks across runs with fresh inits — PiSSA re-init specifically**
  gives unaligned factor bases; averaging them is garbage.
- **Delta-space** (EMA the materialized `B·A`): always mathematically
  correct, but a blended delta has rank ≤ Σ rᵢ. Store as concatenated
  adapters (rank grows) or SVD-recompress back to rank r (lossy,
  measurable).

**Proposed resolution:** factor-space EMA *inside* one training run (the
taps live in the trainer, one run = one trajectory, so it is valid);
delta-space composition (the multi-adapter scaled load) *across* runs and
at serve time, which the inference path gives us for free. SVD
recompression only if we ever need to fold a blend back into a single
adapter file.

### 2. What is "the base" under PiSSA?

PiSSA init modifies the effective starting point (step-0 policy ≠ raw
quantized base; `qlora_train_pref.py` already notes the analogous `qerr`
reference mismatch). Decide whether the blend's base anchor is:
  (a) the raw quantized base (simple, slightly wrong under PiSSA), or
  (b) base + step-0 adapter (exact, one more adapter slot).
Likely (a) with a printed note, same convention the pref trainer uses.

## Open questions (for when we build it)

- Runtime re-blend: `lora_scaling` is set at load; do we want scales
  mutable between generations without reload? (Chat-UI slider dream.)
- Per-module vs global blend weights (probably global; per-module is
  merge-paper territory, diminishing returns).
- Tap snapshot cadence to disk vs in-VRAM EMA accumulators (VRAM cost of
  3 extra adapter copies is small at LoRA sizes; disk snapshot each N
  steps for crash safety).
- Whether `ema_glacial` should decay toward base rather than toward
  `live` (a "forget slowly toward factory settings" prior).

## Explicit non-goals for now

- Building any of this before Phase 1 (rating UI + dataset editor +
  offline run launcher) and pineapple's #4 (online training queue) exist.
- Cross-model or cross-base blending. Same base, same trajectory family
  only.
