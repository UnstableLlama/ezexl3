# ezexl3

**ezexl3** is a simplified interface for exllamav3: quantize, verify, benchmark, visualize, upload, and chat. One pip install, one CLI.

```bash
pip install ezexl3
```

or for custom templates, use a local editable install

```bash
git clone https://github.com/UnstableLlama/ezexl3/
cd ezexl3
pip install -e .
```


Requires a local installation of [exllamav3](https://github.com/turboderp-org/exllamav3).

---

## Quick Start

### Dashboard
```bash
ezexl3 ui
```
Launches a web dashboard on port 8801. Every CLI subcommand is a clickable form with live terminal output via SSE streaming. The Results tab shows the live KL/PPL measurement table alongside qbench's own charts (KL and PPL vs bpw, the per-token KL histograms, the noise-floor comparison). GPU auto-detection. Boolean arguments exposed as toggles. This is the easiest way to use ezexl3.

<p align="center">
  <img src="docs/ezUI1.png" width="65%" />

</p>

The Evals tab shows perf measurements (prefill and generation tokens/s across context lengths) on a dual-axis chart, and the catbench gallery if you ran one. Switch between BPWs with the dropdown.

<p align="center">
  <img src="docs/performance.png" width="65%" />
</p>

### Chat
```bash
ezexl3 chat
```
Launches a lightweight chat web interface for testing quantized models. Browse to your model in the file picker, select GPUs, click load. Branching conversation tree with regeneration, message editing, and sibling navigation. Exllama native, based on chat.py and the generator. No CLI flags needed.

Supports multi-GPU (`-d 0,1`), configurable sequence length (cache is sized 2x behind the scenes), and cache quantization (`-cq 6,6`). Auto-detects prompt format from the model name. Useful for spot-checking quant quality at different BPW levels before uploading.

Speculative decoding is available through the draft model field (DFlash or any smaller draft model), or via the MTP checkbox for models with a built-in MTP head (Qwen3.5+, requires an exllamav3 build with MTP support) — equivalent to exllamav3's `--mtp` flag.

For PLE models with a hashed n-gram embedding table (e.g. Qwen3.8-Flash-Next), the **N-gram table in RAM** checkbox (CLI: `-ngr`) loads the table fully into system RAM instead of streaming rows from disk per token — tens of GB of RAM in exchange for avoiding per-token disk reads. Equivalent to exllamav3's `-ngr` flag; needs exllamav3 >= 1.4.5.

<p align="center">
  <img src="docs/chat.png" width="65%" />
</p>

### CLI Pipeline
Run the full pipeline from the command line:
```bash
ezexl3 repo -m /path/to/base_model -b 2,2.5,3,4,5,6 -d 0,1 -t basic
```

---

## What the pipeline does

ezexl3 wraps the exllamav3 quantization and evaluation workflow into a single command that:
- Interleaves quantize → verify per BPW: each BPW is quantized then immediately verified (KL + PPL) before proceeding, halting on error
- Multi-GPU acceleration for both quantization and verification. KL and PPL run in parallel on 2+ GPUs
- Supports optimized BPWs (2.1 bpw, 3.5 bpw etc.)
- Measures KL divergence + PPL @ 200k tokens, recording data to CSV
- Optional perf measurement (prefill and generation tokens/s across context lengths) with its own SQLite database
- Generates a HuggingFace-ready `README.md` with your measurements using customizable templates
- Embeds an SVG graph from the measurement CSV in the README
- Optional catbench integration. Generates SVG kitten drawings at each BPW and assembles them into a grid
- Optional HuggingFace upload, with metadata locks and a dry-run preview before any repos are created
- Checkpoints and resumes intelligently

```
model → [quantize → verify KL+PPL] per BPW → optimize → evals → graph → README → upload
```

---

### Single-stage subcommands
If you only want to run specific stages:
```bash
# Quantize only
ezexl3 quantize -m /path/to/base_model -b 2,2.5,3,4,5,6 -d 0,1

# Quantize with optimized target (automatically ensures integer neighbors)
ezexl3 repo -m /path/to/base_model -b 4.07 -d 0

# Measure only
ezexl3 measure -m /path/to/base_model -b 2,3,4,5,6 -d 0,1

# Compare quants against the BF16 reference (exllamav3 qbench: cached
# reference logits, self-noise floor, KLD mean/median/p90 + plots).
# BPWs auto-detected; results and plots land in <model>/qbench/
ezexl3 qbench -m /path/to/base_model

# Generate README only (from existing CSV)
ezexl3 readme -m /path/to/base_model -t fire

# Quantize just the MTP tensors (adds speculative decoding to legacy quants)
# -hq raises the bitrate of select MTP layers, matching the integrated conversion's -hq
ezexl3 mtp -m /path/to/base_model -mb 4 -d 0

# Upload to HuggingFace (dry-run by default)
ezexl3 upload -m /path/to/base_model

(but really everything is checkpointed so it usually doesn't hurt to just run the "repo" command every time)
```

### Per-BPW Paint Flags
The dashboard exposes four paint buttons that toggle quantization flags on individual BPW tokens. Click a button, then click a BPW in the parsed-token row to apply it:

- `-hq` — high-quality boost, useful on low BPWs where the head needs the extra precision
- `-sc` — self-calibrated quants (see below); works on any BPW, integer or decimal
- `-opt` — opt-in optimized fractional pipeline (only applies to fractional BPWs)
- `-pm` — global MoE speedup, applies to all BPWs at once
<p align="center">
  <img src="docs/args2.png" width="45%" />
</p>
The same flags work from the CLI via `--quant-args`, but the dashboard is faster for mixing them across BPWs.

Head and vision bitrates are plain numeric options rather than paints:

- `-hb N` / **Head Bits** box — output head layer bitrate, 1-8 (exllamav3 default: 6)
- `-vb N` / **Vision Bits** box — vision tower bitrate, 1-8, or 16 to store unquantized (vision models only)

(The old `-hb8` paint flag still works from the CLI, but `-hb` takes precedence when both are given.)

For PLE models with a hashed n-gram embedding table (e.g. Qwen3.8-Flash-Next), the dashboard's collapsed **N-gram** block (and the matching CLI flags) controls how the table is quantized (exllamav3 >= 1.4.5 required; ignored by models without a table):

- `-ngb N` / **N-gram Bits** box — bits per weight for the n-gram table, 1-8 (exllamav3 default: target BPW rounded)
- `-ngf FILE` / **N-gram File** box — reuse a pre-quantized table (from exllamav3's `util/convert_ngram.py`) instead of quantizing it again; handy when producing several BPWs that should share one table

### Self-Calibrated Quants (`-sc`)
BPWs painted with `-sc` are built through exllamav3's experimental optimization pipeline
(exllamav3 >= 1.4.3 required) instead of the uniform allocation:

1. the model generates its own in-domain calibration trace (using an existing >= 5 bpw quant
   as the sampler when one is available, otherwise the unquantized model),
2. an existing integer quant (lowest available) is probed for per-tensor quantization error,
3. per-tensor sensitivity is measured on the unquantized model with shaped noise injection,
4. a per-tensor bitrate recipe is compiled for each `-sc` BPW, and
5. each BPW is converted with the recipe (`-rcp`) and the self-sampled calibration data (`-cd`).

Every stage writes plain files under `<model>/selfcal/` and is skipped on re-runs when its
output already exists, so interrupted pipelines resume. Sensitivity measurement checks free
VRAM against a conservative estimate of model weights, cached states, and Hessian/noise
workspace. It loads the full model when there is room; otherwise it streams one unquantized
module at a time, caching layer inputs, reference logits, and shaping factors in system RAM.
The largest module plus workspace must still fit in VRAM. Each perturbation replays the
remaining modules, so streaming adds weight-loading overhead. The estimate is a heuristic;
an OOM during automatic full-model loading also triggers streaming, but an OOM later during
measurement still stops the run. Standalone `sc_measure.py` accepts `--load-mode
auto|resident|streaming`, or `--streaming` / `--no-streaming` to force either mode.
MoE sensitivity measurement remains unvalidated.

With multiple GPUs selected by `-d`, sensitivity measurement runs one independent worker
per selected GPU (any GPU count). Modules are assigned by estimated suffix-replay cost,
keeping each module's Hessian capture together; each worker independently selects resident
or streaming mode using its GPU's free VRAM. A coordinator atomically merges completed
tensors into `selfcal/noise_attrib.json`. Worker checkpoints in
`selfcal/noise_attrib.json.workers/` survive interruption and are reused even if the next
run selects a different number of GPUs, including switching back to one GPU.

Workers each keep their own reference logits and activation caches, so system RAM use
grows with worker count. Disk bandwidth, CPU work, and unequal GPU speeds can limit
scaling; this does not pool GPU memory or guarantee an N-fold speedup. Select fewer GPUs
with `-d` when RAM or storage bandwidth is the bottleneck. An already-running measurement
continues with its original worker count; stop and rerun to use a new selection.

```bash
# 2.5 and 3.14 bpw self-calibrated, 4 and 6 bpw standard, 4-bit head everywhere
ezexl3 repo -m /path/to/base_model -b 2.5,3.14,4,6 -sc 2.5,3.14 -hb 4 -d 0,1
```

### QBench (`ezexl3 qbench`)
Runs exllamav3's qbench harness (the replacement for `compare_q.py`) against an ezexl3 model
directory: the BF16 reference's logits are computed once and cached to disk, then every quant
streams through the same test data against the cache — adding one BPW later only measures that
BPW. A second reference pass with BF16-rounding noise measures the model's *self-noise floor*:
the KLD any lossless-equivalent kernel change would show, which contextualizes small
differences between quants. KLD is reported as mean/median/p90 plus buckets by reference
confidence (the mean is dominated by tokens where the reference itself is undecided; the
median and high-confidence buckets isolate actual quantization damage).

```bash
# Auto-detects <bpw>/ quant subdirs; writes results + plots to <model>/qbench/
ezexl3 qbench -m /path/to/base_model

# Explicitly evaluate WikiText instead of generating an evaluation trace
ezexl3 qbench -m /path/to/base_model -b 3,4,5 --dataset wiki2 --template chat
```

By default, qbench generates a **separate evaluation trace** with the vendored upstream
`eval/qbench_prompts.py`, following the calibration/evaluation split in
[Turbo's model card](https://huggingface.co/turboderp/Qwen3.8-27B-exl3). It uses the highest
completed quant at 5 bpw or above (the same selection rule as SC), or the base model if none exists. Generation uses upstream
defaults: a target of 20,000 response tokens, a 4,096-token cap per turn, and a 0.3 tool
conversation fraction. These are the script defaults, not a claim about Turbo's exact run.
The base-model fallback requires a full generation load. Trace generation makes every GPU
supplied to `measure -d` available for automatically splitting the donor and KV cache
(for example, `-d 0,1,2,3` makes all four available);
qbench's streamed comparison passes use the first GPU. The log lists the trace-generation
GPUs explicitly.

The trace is saved to `<model>/qbench/qbench_prompts_gen.json` and reused across all quants
and later runs. An existing `<model>/qbench_prompts_gen.json` is also recognized. SC's
`selfcal/cal_trace.json` is not automatically reused. Logs announce generation/reuse, the
donor, the actual trace path and token counts, or the selected text dataset. Use `--trace`
to supply another evaluation JSON; trace mode scores response positions and ignores
`--rows`, `--length`, and `--template`. Failed trace generation stops the run and leaves no
completed trace; it does not silently fall back to WikiText.

The generated `<model>/qbench/project.yml` is reused on later runs; old generated default
WikiText settings migrate automatically to the separate eval trace. Custom test settings
are preserved, and explicit `--dataset`/`--trace` overrides update the test source without
requiring `--regen`. Changing test data requires new measurements; the old cache is retained.
`--regen` rewrites the project but reuses the evaluation trace. You can add GGUF entries
(`engine: llamacpp`) or HF checkpoints
(`engine: transformers`) for cross-format comparisons — cached results make each addition
cheap. Outputs: `qb_results.json` plus PPL/KLD scatter, KLD spread, and per-token KLD
histogram plots. The README also includes `qb_kld_hist_combined.png`: all quants' raw
per-token KL distributions on one log-axis plot with the noise-floor distribution, like
Turbo's bottom chart. It uses cached per-token measurements and requires the noise-floor
pass. Requires Transformers >= 5.0.0 (including `TokenizersBackend`), seaborn and pyyaml
(installed with ezexl3), and a recent exllamav3. For an existing environment reporting
`Tokenizer class TokenizersBackend does not exist`, run
`python -m pip install -U "transformers>=5.0.0" "seaborn>=0.13"` with that environment's Python.

Checkpointing is automatic, per completed model pass (including BF16 and the noise floor),
under `<model>/qbench/logit_cache/qbench/`. Rerun the same command after an interruption;
completed passes print `Cached`, while an interrupted pass restarts. Keep this directory:
`qb_results.json` and the measurement CSV alone cannot replace the keyed cache. ezexl3's
README metadata updates do not invalidate it. Older entries keyed to that metadata's
timestamp may require a one-time remeasurement after upgrading.

If reference logits have been evicted or deleted, saved measurements still regenerate
reports without inference. Adding a new quant then rebuilds the BF16 logits as needed,
while reusing completed quant and noise-floor measurements. Model changes or changes to
the project's test settings invalidate the corresponding cache; `--regen` rewrites the
project settings but does not clear matching cached measurements.

### Template System
You can customize the generated README by providing a template name via `--template` or `-t`.
Templates are stored in the `/ezexl3/templates/` directory — just use the short name:

```bash
ezexl3 repo -m /path/to/base_model -t fire -b 2,3,4,5,6 -d 0,1
```

If no template is specified, it defaults to `basic`.

**Easily generate your own custom template with AI assistance!**

Copy and paste any template from `/ezexl3/templates/` into your favorite LLM (Gemini, Claude, ChatGPT) along with this example prompt, followed by your own description:

```bash
Take this template, keep the main layout and variables, and modify it aesthetically based on my following prompts. Preserve all of the labels and title strings, only change the aesthetic, not the words or numbers:

*Make it dark and understated, high contrast, professional, metallic.*
```
Then save the result in `/ezexl3/templates/` and use it with `-t yourname`.
<p align="center">
  <img src="ezexl3/templates/basicTemplate.png" width="35%" />
  <img src="ezexl3/templates/punkTemplate.png" width="35%" />
  <img src="ezexl3/templates/fireTemplate.png" width="45%" />
  <img src="ezexl3/templates/greenTemplate.png" width="45%" />
</p>

###  Catbench
SVG Catbench is available as a measurement option via the `-cb` flag. It runs catbench inference at every BPW level (including optimized fractionals), extracts SVGs, and assembles them into a grid in the final README.

```bash
ezexl3 repo -m /path/to/base_model -b 2,3,4,5,6,8 -d 0,1 -t punk -cb
```

- `-cb` alone runs 3 samples per BPW (default), `-cb 5` runs 5
- Catbench runs as a batch pass after KL/PPL/perf complete, using the multi-GPU queue
- VRAM pre-flight check before each catbench load — skips gracefully if model won't fit, automatically uses multi-GPU for large models
- Best valid SVG is selected from N samples for the grid
- SVG extraction and grid assembly happen in a batch pass after all inference completes
- Catbench results are checkpointed like everything else — rerunning skips completed samples
- bf16 baseline included when VRAM allows

###  HuggingFace Upload
The Upload tab (or `ezexl3 upload`) creates HuggingFace repos for your quants. Defaults to dry-run mode so you see exactly what repo names will be created before anything is published.

```bash
# Preview what would be created
ezexl3 upload -m /path/to/base_model

# Actually create and upload
ezexl3 upload -m /path/to/base_model --no-dry-run
```

- Single mode (default): one standalone repo per BPW, named `MODEL-exl3-BPW`. Recommended.
- Branched mode: one repo with each BPW as a separate branch. Note that HuggingFace's download counter does not count branches — branched repos show only the main branch's downloads. Standalone repos preserve your download numbers.
- Metadata fields (Author, Model Name, Repo Link, Quantized By) lock during the README write phase so the values can't drift mid-pipeline.
- Preflight check verifies your HF token before any repos are created.

### Inference Evaluation with WebUI
ezexl3 includes a lightweight chat web interface for quickly testing quantized models. Exllama native, based on chat.py and the generator.

```bash
ezexl3 chat -m /path/to/quantized_model -d 0
```

### Advanced: Passthrough Flags
You can pass custom arguments directly to the underlying quantization (`multiConvert`) or measurement scripts using the `--quant-args` and `--measure-args` flags.

**Important**: These flags require a double-dash `--` delimiter to separate the passthrough block from the rest of the arguments.

```bash
# Pass custom calibration dataset to quantization
ezexl3 repo -m /path/to/model -b 4.0 --quant-args -- -pm

# Pass custom rows/device settings to measurement
ezexl3 repo -m /path/to/model -b 4.0 --measure-args -- -r 200 -d 0
```

Common Use Cases:
- **Quantization**: `-pm` (MoE speedup)
- **Measurement**: `-r` / `--rows` (number of rows for PPL)

Note: passthrough blocks consume remaining args until another passthrough block starts, so keep normal CLI flags (like `--no-readme`) before `--measure-args -- ...`

###  `--no-verify` (Legacy Batch Mode)
By default, ezexl3 interleaves quantization with KL/PPL verification per BPW. Use `--no-verify` (or `-nv`) to revert to the old batch pipeline (all quants first, then all measurements):

```bash
ezexl3 repo -m /path/to/model -b 2,3,4,5,6 -d 0,1 --no-verify
```

This is useful if you're confident in your quantization setup and want to let everything run unattended without per-BPW halting.

### Optimized BPW workflow

If you request an optimized BPW (for example `4.07`), ezexl3 executes the following order:

1. Detect optimized targets and remove them from the initial integer quant queue.
2. Ensure required neighboring integers exist in the quant queue (`4` and `5` for `4.07`).
3. Quantize each integer BPW one at a time, verifying KL+PPL immediately after each (halts on error). With 2+ GPUs, KL and PPL run in parallel during verification.
4. Run exllamav3 `util/measure.py` in a dynamic multi-GPU queue for required integer pairs (resume-safe: skips if `measurements/<low>-<high>_measurement.json` exists), with terminal logs when jobs are assigned and completed per GPU.
5. Run exllamav3 `util/optimize.py` to build the optimized output directory.
6. Verify each optimized BPW with KL+PPL measurement (halts on error).

To locate exllamav3 utility scripts, ezexl3 uses bundled vendored copies (no manual path configuration needed).

###  Headless Mode
For automated pipelines, use the `--no-prompt` (or `-np`) flag to skip interactive metadata collection for the README. It will use sensible defaults based on the model directory name and your environment.

```bash
ezexl3 repo -m /path/to/model -b 4.0 --no-prompt
```
