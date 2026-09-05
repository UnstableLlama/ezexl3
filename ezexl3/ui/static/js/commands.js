// ── Command schema definitions ──────────────────────────────────
// Each command maps to a CLI subcommand with typed fields that
// drive dynamic form generation in forms.js.

const COMMANDS = {
  repo: {
    label: "Repo",
    subtitle: "Full Pipeline",
    description: "Quantize \u2192 Measure KL+PPL \u2192 README",
    hasMetadata: true,
    groups: {
      ngram: { label: "N-gram", help: "Hashed n-gram embedding tables (PLE models, e.g. Qwen3.8-Flash-Next). Requires exllamav3 \u2265 1.4.5; ignored by models without a table." },
    },
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "BF16/base model directory" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "1-8, comma separated, decimals ok (e.g. 2,3,4.5,6)",
        bpwPaintFlags: [
          { name: "hq", flag: "-hq", label: "-hq", color: "#4a90d9", tooltip: "use on low bpws" },
          { name: "sc", flag: "-sc", label: "-sc", color: "#9b6dd6", isGlobal: true, tooltip: "self-calibrated quants (all bpws, slow) — traces from a 6bpw+ quant if one exists, else the much slower bf16 model" },
          { name: "pm", flag: "-pm", label: "-pm", color: "#ffffff", isGlobal: true, tooltip: "use on MoEs" },
        ]
      },
      // Only meaningful while -sc is on, so it stays hidden until then and
      // renders directly under the BPW row rather than down in Options.
      { name: "sc_donor", flag: "-scd", type: "path", label: "SC Trace Generation Model",
        help: "Model the -sc calibration trace is sampled from. Must be 6bpw+, else defaults to slower bf16. Leave empty to auto-pick the highest quant ≥ 5 bpw in the model directory.",
        afterField: "bpws", showWhen: { field: "bpws", globalFlag: "sc" } },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1", help: "GPUs used, comma separated (e.g. 0,1)" },
      { name: "device_ratios", flag: "-r", type: "csv", label: "Device Ratios", placeholder: "1,1", help: "VRAM split ratio per device when GPUs are uneven" },
      { name: "head_bits", flag: "-hb", type: "number", default: "6", label: "Head Bits", help: "Output head layer bitrate, 1-8 (exllamav3 default: 6)", toggleable: true },
      { name: "vision_bits", flag: "-vb", type: "number", default: "6", label: "Vision Bits", help: "Vision tower bitrate, 1-8, or 16 = unquantized (vision models only). exllamav3 default: 6 on validated towers (Gemma 4, GLM-4V, Qwen3-VL, Step3.7, Muse/Glimmer), 16 = unquantized on every other vision model.", toggleable: true },
      { name: "mtp_bits", flag: "-mb", type: "number", default: "4", label: "MTP Bits", help: "4 recommended", toggleable: true },
      { name: "ngram_bits", flag: "-ngb", type: "number", default: "4", label: "N-gram Bits", help: "Bits per weight for the hashed n-gram embedding table, 1-8 (exllamav3 default: target BPW rounded)", toggleable: true, group: "ngram" },
      { name: "ngram_file", flag: "-ngf", type: "text", label: "N-gram File", placeholder: "/path/to/ngram_embedding.safetensors", help: "Pre-quantized n-gram table (from exllamav3 util/convert_ngram.py) reused instead of quantizing the table", group: "ngram" },
      { name: "template", flag: "-t", type: "template", label: "Template", help: "README template style" },
      // Boolean flags
      { name: "no_verify", flag: "-nv", type: "boolean", label: "No Verify", help: "Batch mode: all quants then all measures" },
      { name: "no_cleanup", flag: "-nc", type: "boolean", label: "Keep Work Dirs", help: "Keep w-* working directories and logs" },
      { name: "no_readme", flag: "--no-readme", type: "boolean", label: "Skip README" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs", help: "Skip per-GPU log files" },
      { name: "no_prompt", flag: "-np", type: "boolean", label: "Headless", help: "Use defaults instead of prompting for README metadata" },
      { name: "no_graph", flag: "-ng", type: "boolean", label: "No Charts", help: "Skip embedding the qbench charts in the README" },
      { name: "no_measurement", flag: "-nm", type: "boolean", label: "No Measurement", help: "Skip qbench measurement and charts entirely" },
      // Evals section — KL and PPL are on by default (inverted: emit flag when OFF).
      // Both come from a single qbench run, which also draws the README charts.
      { name: "no_kl", flag: "--no-kl", type: "boolean", label: "KL Divergence", help: "KL divergence, measured by qbench", section: "evals", defaultOn: true, invertFlag: true },
      { name: "no_ppl", flag: "--no-ppl", type: "boolean", label: "Perplexity", help: "Perplexity, measured by qbench", section: "evals", defaultOn: true, invertFlag: true },
      { name: "perf", flag: "-perf", type: "number", label: "Performance", placeholder: "32768", help: "Inference performance benchmark (max length)", toggleable: true, section: "evals" },
      { name: "catbench", flag: "-cb", type: "number", label: "Catbench", placeholder: "3", help: "cat attempts per BPW", toggleable: true, section: "evals" },
    ],
  },

  quantize: {
    label: "Quantize",
    subtitle: "Quantize Only",
    description: "Run quantization without measurement or README",
    groups: {
      ngram: { label: "N-gram", help: "Hashed n-gram embedding tables (PLE models, e.g. Qwen3.8-Flash-Next). Requires exllamav3 ≥ 1.4.5; ignored by models without a table." },
    },
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "BF16/base model directory" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "1-8, comma separated, decimals ok (e.g. 2,3,4.5,6)",
        bpwPaintFlags: [
          { name: "hq", flag: "-hq", label: "-hq", color: "#4a90d9", tooltip: "use on low bpws" },
          { name: "sc", flag: "-sc", label: "-sc", color: "#9b6dd6", isGlobal: true, tooltip: "self-calibrated quants (all bpws, slow) — traces from a 6bpw+ quant if one exists, else the much slower bf16 model" },
          { name: "pm", flag: "-pm", label: "-pm", color: "#ffffff", isGlobal: true, tooltip: "use on MoEs" },
        ]
      },
      // Only meaningful while -sc is on, so it stays hidden until then and
      // renders directly under the BPW row rather than down in Options.
      { name: "sc_donor", flag: "-scd", type: "path", label: "SC Trace Generation Model",
        help: "Model the -sc calibration trace is sampled from. Must be 6bpw+, else defaults to slower bf16. Leave empty to auto-pick the highest quant ≥ 5 bpw in the model directory.",
        afterField: "bpws", showWhen: { field: "bpws", globalFlag: "sc" } },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1", help: "GPUs used, comma separated (e.g. 0,1)" },
      { name: "device_ratios", flag: "-r", type: "csv", label: "Device Ratios", placeholder: "1,1", help: "VRAM split ratio per device when GPUs are uneven" },
      { name: "head_bits", flag: "-hb", type: "number", default: "6", label: "Head Bits", help: "Output head layer bitrate, 1-8 (exllamav3 default: 6)", toggleable: true },
      { name: "vision_bits", flag: "-vb", type: "number", default: "6", label: "Vision Bits", help: "Vision tower bitrate, 1-8, or 16 = unquantized (vision models only). exllamav3 default: 6 on validated towers (Gemma 4, GLM-4V, Qwen3-VL, Step3.7, Muse/Glimmer), 16 = unquantized on every other vision model.", toggleable: true },
      { name: "mtp_bits", flag: "-mb", type: "number", default: "4", label: "MTP Bits", help: "4 recommended", toggleable: true },
      { name: "ngram_bits", flag: "-ngb", type: "number", default: "4", label: "N-gram Bits", help: "Bits per weight for the hashed n-gram embedding table, 1-8 (exllamav3 default: target BPW rounded)", toggleable: true, group: "ngram" },
      { name: "ngram_file", flag: "-ngf", type: "text", label: "N-gram File", placeholder: "/path/to/ngram_embedding.safetensors", help: "Pre-quantized n-gram table (from exllamav3 util/convert_ngram.py) reused instead of quantizing the table", group: "ngram" },
      { name: "out_template", flag: "--out-template", type: "text", default: "{model}/{bpw}", label: "Output Template", help: "Fields: {model}, {model_name}, {bpw}" },
      { name: "w_template", flag: "--w-template", type: "text", default: "{model}/w-{bpw}", label: "Work Dir Template", help: "Fields: {model}, {model_name}, {bpw}" },
      { name: "dry", flag: "--dry", type: "boolean", label: "Dry Run", help: "Print commands without executing" },
      { name: "continue_on_error", flag: "--continue-on-error", type: "boolean", label: "Continue on Error" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs" },
    ],
  },

  // Every eval lives here, including KL/PPL — those come from qbench, whose
  // test-data and reference knobs sit in the collapsed "KL / PPL (qbench)"
  // subsection. Leaving a knob toggled off means "same as the repo pipeline".
  measure: {
    label: "Evals",
    subtitle: "Measure",
    description: "Measure KL + PPL and run the a-la-carte evals",
    groups: {
      qbench: { label: "KL / PPL (qbench)", help: "How the KL+PPL run is set up. Defaults: 10 rows × 2048 tokens of wiki2, scored against a cached BF16 reference. Toggling any of these on re-measures BPWs already in the table." },
    },
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "Model directory with quantized outputs" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6" },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs" },
      { name: "no_cleanup", flag: "-nc", type: "boolean", label: "Keep Temp Files" },
      // qbench knobs — folded in from the old QBench tab
      // These carry real defaults rather than placeholders: --rows/--length/
      // --cache-gb all require a value, and a toggled-on field with an empty
      // box would emit the bare flag.
      { name: "qb_rows", flag: "--rows", type: "number", default: "10", label: "Test Rows", help: "Number of test rows", toggleable: true, group: "qbench" },
      { name: "qb_length", flag: "--length", type: "number", default: "2048", label: "Row Length", help: "Tokens per row", toggleable: true, group: "qbench" },
      { name: "qb_dataset", flag: "--dataset", type: "select", choices: ["wiki2", "openwebtext"], default: "wiki2", label: "Dataset", toggleable: true, group: "qbench" },
      { name: "qb_template", flag: "--template", type: "select", choices: ["none", "chat", "assistant"], default: "none", label: "Chat Template", help: "Apply the model's chat template to test rows", toggleable: true, group: "qbench" },
      { name: "qb_trace", flag: "--trace", type: "text", label: "Test Trace", placeholder: "(optional) qbench_prompts.py JSON", help: "In-domain test trace; replaces dataset/rows/length", group: "qbench" },
      { name: "qb_ref_engine", flag: "--ref-engine", type: "select", choices: ["exllamav3", "transformers"], default: "exllamav3", label: "Reference Engine", help: "Engine for the BF16 reference pass (transformers needs transformers+accelerate)", toggleable: true, group: "qbench" },
      { name: "qb_cache_gb", flag: "--cache-gb", type: "number", default: "50", label: "Logit Cache (GB)", help: "Cache size limit; oldest entries evicted", toggleable: true, group: "qbench" },
      { name: "qb_no_noise_floor", flag: "--no-noise-floor", type: "boolean", label: "Skip Noise Floor", help: "Faster, but disables histogram plots and the floor line", toggleable: true, group: "qbench" },
      { name: "qb_regen", flag: "--regen", type: "boolean", label: "Regenerate Project", help: "Rewrite qbench/project.yml instead of reusing it (cached results survive)", toggleable: true, group: "qbench" },
      // Evals section — KL and PPL are on by default (inverted: emit flag when OFF).
      // Both come from a single qbench run, which also draws the README charts.
      { name: "no_kl", flag: "--no-kl", type: "boolean", label: "KL Divergence", help: "KL divergence, measured by qbench", section: "evals", defaultOn: true, invertFlag: true },
      { name: "no_ppl", flag: "--no-ppl", type: "boolean", label: "Perplexity", help: "Perplexity, measured by qbench", section: "evals", defaultOn: true, invertFlag: true },
      { name: "perf", flag: "-perf", type: "number", label: "Performance", placeholder: "32768", help: "Inference performance benchmark (max length)", toggleable: true, section: "evals" },
      { name: "catbench", flag: "-cb", type: "number", label: "Catbench", placeholder: "3", help: "cat attempts per BPW", toggleable: true, section: "evals" },
    ],
  },

  readme: {
    label: "README",
    subtitle: "Generate README",
    description: "Generate HuggingFace README from existing CSV",
    hasMetadata: true,
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "Directory with measurement CSV" },
      { name: "bpws", flag: "-b", type: "csv", label: "BPWs", placeholder: "2,3,4,5,6", help: "BPWs (required for single mode, auto-detected otherwise)" },
      { name: "mode", flag: "--mode", type: "select", choices: ["single", "branched"], default: "single", label: "Mode", help: "Single: per-BPW READMEs with cross-linked repos. Branched: single README" },
      { name: "template", flag: "-t", type: "template", label: "Template", help: "README template style" },
      { name: "no_prompt", flag: "-np", type: "boolean", label: "Headless", help: "Use defaults for metadata" },
      { name: "no_graph", flag: "-ng", type: "boolean", label: "No Graph" },
      { name: "no_measurement", flag: "-nm", type: "boolean", label: "No Measurement", help: "Omit the qbench charts from the README" },
      { name: "catbench", flag: "-cb", type: "boolean", label: "Catbench", help: "Include the SVG Catbench grid in the README" },
    ],
  },

  upload: {
    label: "Upload",
    subtitle: "HuggingFace",
    description: "Create repos and upload to HuggingFace",
    twoActions: true,
    hasMetadata: true,
    metadataFields: ["MODEL_DIR", "MODEL", "USER"],
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "Directory with quantized models" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "BPWs to upload" },
      { name: "mode", flag: "--mode", type: "select", choices: ["single", "branched"], default: "single", label: "Mode", help: "Single: separate repo per BPW. Branched: one repo with branches" },
      // Boolean flags
      { name: "dry_run", flag: "-dr", type: "boolean", label: "Dry Run", help: "Preview repos without contacting HuggingFace", defaultOn: true },
      { name: "private", flag: "--private", type: "boolean", label: "Private Repos", help: "Create private HuggingFace repos" },
      { name: "small_only", flag: "--small-only", type: "boolean", label: "Small Files Only", help: "Exclude *.safetensors, *.bin, *.pt, *.ckpt" },
    ],
  },
};
