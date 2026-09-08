# Handoff — 2026-08-01: CPU offload controls in the chat UI

On branch `dev` (commits `018b552` + this one). Wraps exllamav3 1.3.0's
CPU-offload features in the chat load panel.

## Intent

UnstableLlama asked to survey turboderp upstream for new CPU-offloading
features and expose them in the chat UI. Upstream v1.3.0 (on both
`upstream/master` and `upstream/dev`) added three, all via
`model_init.add_args`:

| Arg | Effect |
|---|---|
| `-mcl` / `-mclt` | Routed experts of the first N block-sparse MoE layers run on CPU, weights in system RAM |
| `-ccs` | Second-tier CPU KV cache (`generator/cpu_cache.py`, commit `7eca0eb`) |
| `-dmcl` / `-dmclt` | Same as `-mcl` for the draft model / MTP head |

All three are exposed, in one collapsible **CPU Offload** group.

## What landed

### Engine (`ezexl3/chat/inference.py`)
- `ChatEngine` gained a `cpu_offload` dict (ctor + `load_model`), stored as
  `self._cpu_offload`.
- `_build_model_args()` appends `-mcl`/`-mclt`/`-ccs` **only when the running
  parser exposes them**, reusing the existing `-ambs` capability-sniff
  precedent. Threads are never sent without a non-zero layer count.
- `cpu_offload_support()` (lru_cached) reports `{moe, moe_threads, cache}` by
  building the parser and inspecting `option_strings`.
- `_apply_draft_cpu_offload()` — we load draft models ourselves rather than
  through `model_init`'s draft args, so `-dmcl` would never reach them. It
  writes `infer_params` directly, mirroring `model_init`:
  `draft_moe_cpu_*` for the MTP head (shared config, component != `"text"`),
  plain `moe_cpu_*` for a standalone draft model (own config, own text
  component). See `block_sparse_mlp.py:916` for the dispatch.

### Server (`ezexl3/chat/server.py`)
- `_parse_cpu_offload()` coerces the panel's block to non-negative numbers.
- Support flags ride on `/api/gpus` and `get_status()` alongside `cpu_cores`.
- `_moe_offload_eligible()` reads the model's `model.safetensors.index.json`
  (no weights loaded) and returns `{moe, mul1}`; `/api/browse` includes it.
- The draft-reload path carries `engine._cpu_offload` across so toggling a
  draft source doesn't silently drop the offload settings.

### UI (`static/index.html`, `js/model.js`, `style.css`)
- Collapsible group: MoE layers, worker threads, CPU KV cache GB, draft MoE
  layers/threads. Auto-expands when anything is configured; persists to
  `ui.json` under `cpu_offload` on successful load.
- Controls disable themselves with an explanation on pre-1.3.0 builds.
- `updateOffloadEligibility()` shows a green note for mul1 experts, an amber
  warning for non-mul1 MoE models, and nothing for dense models.

## Verification

Real end-to-end load through `python -m ezexl3 chat` (not a stub):
**Laguna-S-2.1, 59.8 GB, onto 48 GB of total VRAM** — impossible without
offload, so the successful load is the proof.

- VRAM: GPU0 22,302 + GPU1 9,253 MiB = 30.8 GB
- System RAM: spawned worker (child of the server) at 29.4 GB RSS
- Load 2:03; generated coherent text at 46 tokens / 26 s ≈ 1.8 tok/s

Also verified: args land correctly on the real 1.3.0 parser (12/6/24.0) and
are skipped without crashing on a simulated 1.1.0; UI render, capability
gating, persistence round-trip, eligibility warning across three real models.

## Gotchas worth remembering

- **mul1 only.** `load_cpu_offload` (`block_sparse_mlp.py:554`) requires a
  per-expert `.mul1` tensor and otherwise prints *"experts are not mul1, CPU
  offload skipped"* to the server console only. Locally, `Laguna-S-2.1` is
  mul1; the Gemma4 26B-A4B exl3 quant is `mcg` and is **not** eligible. This
  is what the new amber warning exists to surface.
- **spawn, not fork.** `moe_cpu_host.py:326` uses
  `multiprocessing.get_context("spawn")`. A standalone script that loads a
  model must have an `if __name__ == "__main__":` guard or the child
  re-imports it and recursively re-loads (symptom: `ConnectionResetError`
  from `conn.recv()`). `python -m ezexl3` is **safe** — CPython's
  `spawn.py:256` exempts module names ending in `.__main__`.
- MoE offload asserts layer-split mode; it cannot combine with `-tp`. ezexl3
  chat never passes `-tp`, so no conflict today.
- Throughput cost is real (~1.8 tok/s here). This is a "make it fit at all"
  feature, not a speed feature.

## TODO / not done

- The dashboard's own model-load path doesn't expose these knobs — chat only.
- `-ccs` (CPU KV cache) is wired and reaches the parser but was **not**
  exercised under load; only the MoE path was proven end-to-end.
- Draft-side offload (`_apply_draft_cpu_offload`) is code-verified against
  upstream's semantics but never run — it needs a MoE draft model or an MTP
  head with mul1 experts.
- No thread-count tuning was explored; 0 (auto = half of cores) was used
  throughout.
