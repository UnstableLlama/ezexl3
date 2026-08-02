# Changelog

All notable changes to ezexl3 are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **chat: Laguna prompt format** (`laguna`, auto-detected from the model
  name): Poolside's Laguna-S-2.1 wire format —
  `<system>…</system>\n<user>…</user>\n<assistant>…</assistant>\n`, stopping
  on `</assistant>`. The load-bearing detail is that *every* assistant turn
  opens with a think block — an open `<think>` when thinking is on, a bare
  `</think>` when off — and the no-think render drops reasoning spans from
  history. Note the format emits no BOS and sets `add_bos` False even though
  the template opens with `〈|EOS|〉`: this tokenizer's `TemplateProcessing`
  post-processor prepends that token on every encode, so writing it here as
  well would tokenize to `[2, 2]`. Verified against the real model's
  tokenizer to land exactly one.
- **chat: `jinja` prompt format**: renders through the loaded model's own
  chat template instead of a hardcoded format — the same file an inference
  server applies. Resolved the way HF transformers does:
  `chat_template.jinja`, then `chat_template.json`, then the
  `chat_template` key of `tokenizer_config.json` (named-template lists
  supported). Stored replies are split back into `reasoning_content` and
  `content` at `</think>` so reasoning templates render history correctly,
  and `enable_thinking` follows the Think toggle. The renderer supports
  HF's `{% generation %}` tag, which plain jinja2 rejects outright and
  which Laguna's template uses. A leading BOS written by the template is
  dropped when the tokenizer is going to prepend one anyway (detected by
  probing, not assumed) — otherwise Llama 3, Gemma, Mistral and Laguna
  would all get a doubled BOS from render-then-encode. Opt-in only:
  auto-detect never selects it, so existing models keep their hardcoded
  format unless you explicitly pick `jinja`.
- **chat: CPU offload controls** (load panel → collapsible *CPU Offload*):
  exposes exllamav3 1.3.0's MoE expert offload (`-mcl`/`-mclt`), the
  second-tier CPU KV cache (`-ccs`), and the draft-model/MTP equivalents.
  Expert weights live in system RAM in a spawned worker process, trading
  speed for VRAM so a model larger than total VRAM can load at all.
  Verified end-to-end: Laguna-S-2.1 (59.8 GB) loaded on 48 GB of VRAM with
  24 of 48 layers offloaded — 30.8 GB VRAM + 29.4 GB RAM, generating at
  ~1.8 tok/s on 8 cores. Each knob is gated on the running exllamav3
  actually exposing it, so older builds load exactly as before and the
  controls grey out with an explanation. Because exllamav3 only offloads
  **mul1**-codebook experts (others silently fall back to GPU), the panel
  reads the selected model's safetensors index and says up front whether
  its experts are eligible. Settings persist in `ui.json`. Note: MoE
  offload is layer-split only and cannot combine with tensor-parallel.
- **chat: Off capture mode (new default)**: the top-bar preference-capture
  toggle gains an Off position for normal chat — no rating controls, no
  duels, nothing written. Unknown or unset persisted `ratings_mode` values
  fall back to Off.
- **chat: DPO duel judgment controls**: each duel candidate now carries
  ▲ chosen / ▼ rejected / ✗ failed marks (▲/▼ exclusive across the duo,
  ✗ allowed on one or both). A Regenerate button replaces just the ✗
  candidates with fresh generations — the kept candidate's text and mark
  survive — and a Commit button (enabled once one ▲ and one ▼ are set)
  writes the chosen/rejected pair and continues from the chosen reply.
  Replaces the one-click "Prefer A/B" buttons; Skip still continues
  without recording.
- **chat: DPO generation prompts (contrastive spoofing)**: sidebar
  System Prompt A/B fields override the system prompt at generation time
  for duel candidates A and B (blank = main prompt), so one candidate can
  be biased toward the target behavior and the other away from it —
  RLCD-style contrastive pairs with a human judge. The dataset's `prompt`
  column always keeps the main system prompt; the generation prompts are
  recorded as `chosen_system`/`rejected_system` metadata columns the
  trainer ignores (null when unspoofed). `/api/chat` gains an optional
  `system_prompts` list (per-candidate, validated against `n`);
  regenerating a ✗ candidate re-uses its own slot's prompt; a "sys" tag
  on a candidate shows the prompt it was generated under.
- **MTP tensor quantization** (`ezexl3 mtp` + dashboard MTP tab): wraps
  exllamav3's new `util/convert_mtp.py` (dev branch). Quantizes just the
  MTP tensors from a base checkpoint into a standalone `.safetensors`
  file that can be dropped alongside a legacy quant's weights to enable
  MTP speculative decoding. Skips if the output file already exists.
- **chat: GPT-OSS (harmony) prompt format** ported from exllamav3 dev,
  with model-name auto-detection (gpt-oss / gpt_oss / gptoss).
- **chat: preference-data capture (KTO / DPO)**: 👍/👎 buttons on assistant
  messages write KTO rows; 👍×👎 among sibling regenerations auto-generates
  DPO pairs, and a ⚖ button records explicit preferred-sibling pairs.
  Rows land in `<datasets_dir>/<name>.{kto,dpo}.jsonl` in exactly the
  column format the exllamav3 fork's `training/qlora_train_pref.py` reads
  with its default keys (prompt / completion / label, prompt / chosen /
  rejected) — a collected dataset trains with zero conversion. The prompt
  column stores the full conversation history as {role, content} turns
  (the trainer currently uses system + last user turn; full history is
  kept for future multi-turn training). Dataset name and directory are
  configurable in the sidebar (persisted to `ui.json`); ratings are keyed
  by conversation-tree node id, so re-rating updates in place and marks
  restore across sessions. New endpoints `GET /api/ratings`,
  `POST /api/rate`; new module `ezexl3/chat/ratings.py`; hand-edited or
  externally appended JSONL lines are preserved verbatim.

### Changed
- **Vendored eval scripts pinned to dev for exllamav3 v1.0.0 prep**:
  `model_diff.py`, `ppl.py`, and `eval_perf.py` re-vendored from upstream
  **dev** (was master) to track the v1.0.0 staging branch. `model_diff.py`
  gains cache-quant simulation and sweep modes (`-cq`, `-cca`, `-cqs`,
  `-cqsf`), `-l/--length`, `-nr/--no_reconstruct`, and `main()` now returns
  the KL divergence; `ppl.py` gains MXFP4/gpt-oss dequant handling and
  `-hf_d/--hf_device`; `eval_perf.py` picks up an inert warmup-loop wrapper.
  `VENDOR_MANIFEST.json` source URLs and hashes updated to dev, with a
  `note` marking these as a temporary pin — revert to master once v1.0.0
  merges. The remaining five vendored scripts are byte-identical on dev.
- **KL divergence orientation**: model A is now the quant and model B the
  base, so the reported KL(A, B) is KL(quant ‖ base) — per turboderp's
  guidance. Previously the direction was reversed. KL values measured
  before this change are not directly comparable (KL is asymmetric);
  delete the affected rows from the measurement DB and re-run `measure`
  if you need consistent numbers within one model's table.

## [0.1.0] - 2026-04-16

First tagged release. `ezexl3` wraps the exllamav3 quantize + measure +
report workflow into a single command and ships a dashboard + chat UI
on top of it.

### Added
- **Dashboard** (`ezexl3 ui`): web dashboard on port 8801 with SSE-streamed
  terminal, GPU auto-detection, clickable subcommand forms, resizable
  split panel, and a KL / PPL tab with live measurement table + SVG
  graph.
- **Evals tab**: live perf chart (dual-axis, dark-mode) beside the
  measurement tables, live catbench grid, trimmed to KL / PPL / perf /
  catbench.
- **Chat UI** (`ezexl3 chat`): lightweight web chat with a branching
  conversation tree, regeneration, message editing, sibling navigation,
  GPU picker, configurable cache size / quantization.
- **Full pipeline** (`ezexl3 repo`): interleaved quantize → verify per
  BPW (halts on error), multi-GPU queue with KL + PPL running in
  parallel, optimized-BPW workflow with automatic integer-neighbor
  back-fill, HuggingFace-ready README with embedded SVG graph.
- **Single-stage commands**: `quantize`, `measure`, `readme`, `evals`,
  `upload`, `chat`, `ui`.
- **Template system**: `basic`, `fire`, `green`, `punk`, plus custom
  templates via `--template`.
- **Catbench** (`-cb`): SVG kitten generation at every BPW with VRAM
  pre-flight, multi-sample selection, batch grid assembly in the final
  README; `-cb N` selects sample count (default 3).
- **Measurement DB**: KL divergence + PPL over 200k tokens, perf detail
  DB (`perf_db`), long-context eval, sequential multi-GPU perf runs,
  model-name confirmation gate with graph regeneration.
- **Headless + passthrough**: `--no-prompt` / `-np`, `--no-verify` / `-nv`
  legacy batch mode, `--quant-args --` and `--measure-args --`
  passthrough.
- **Chat prompt formats**: `gemma4`, `qwen35`, plus upstream drift tests
  that fail loudly if vendored templates diverge.

### Changed
- Measurement CSV exports only populated columns.
- Quant and measure queues sort numerically instead of lexicographically.
- UI upper panel starts at 90 %, snaps to 50 % on Run / Create / Upload.
- Upload dry-run defaults to on; locks spawn unlocked; Measure re-locks
  `MODEL`; Resume button removed.
- `Data` tab renamed to `KL / PPL`; evals chart laid out beside tables.

### Fixed
- Perf cache sized at 2 × `max_length` end-to-end so the final gen
  iteration (`past_len = max_length`, +100 forward tokens) no longer
  overflows (#255, #256).
- Per-length perf results collapse onto a single updating line;
  exceptions now surface in the failure summary (#254).
- `measure`: `TypeError` from missing `prompt_for_model_name` in the
  `repo.py` wrapper (#253).
- `repo_measure`: eval done-message accuracy, interim CSV export,
  partial-result persistence on error, no more silent failures
  (#236, #250).
- `quantize`: `calibration_data` import path for newer exllamav3 (#243).
- `exl3`: top-level `exllamav3` imports guarded against namespace-
  package installs (#245).
- SSE stream regressions introduced by the `repo.py` refactor (#235).

### Known limitations
- The `eval_perf.measure_prefill` / `measure_generate` monkey patch in
  `ezexl3/perf_runner.py` is pragmatic but fragile — every exllamav3
  bump needs a diff-and-re-apply review. Will be revisited if upstream
  grows a proper progress hook.
- UI JS is pre-module (every file is window-global); migration to ES
  modules is deferred to a later release.
- `ezexl3/repo.py` is a thin re-export wrapper whose hand-written
  signatures have drifted once (#253); consolidating to
  `from .repo_measure import …` is a candidate for 0.2.
- Several `32768` defaults (evals.py, perf_runner.py, cli.py, UI JS) are
  not yet centralised behind a shared constant.
