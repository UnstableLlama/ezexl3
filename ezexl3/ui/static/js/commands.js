// ── Command schema definitions ──────────────────────────────────
// Each command maps to a CLI subcommand with typed fields that
// drive dynamic form generation in forms.js.

const COMMANDS = {
  repo: {
    label: "Repo",
    subtitle: "Full Pipeline",
    description: "Quantize \u2192 Measure KL+PPL \u2192 README",
    hasMetadata: true,
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "BF16/base model directory" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "Target bits per weight",
        bpwPaintFlags: [
          { name: "hq", flag: "-hq", label: "-hq", color: "#4a90d9" },
          { name: "hb8", flag: "-hb8", label: "-hb 8", color: "#43a047" },
        ]
      },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1", help: "GPU device indices" },
      { name: "device_ratios", flag: "-r", type: "csv", label: "Device Ratios", placeholder: "1,1", help: "VRAM ratios per device (optional)" },
      { name: "template", flag: "-t", type: "template", label: "Template", help: "README template style" },
      { name: "layers", flag: "-l", type: "select", choices: ["1", "2", "3"], default: "2", label: "Optimization Depth", help: "Layer depth for optimization", toggleable: true },
      // Boolean flags
      { name: "no_verify", flag: "-nv", type: "boolean", label: "No Verify", help: "Batch mode: all quants then all measures" },
      { name: "no_cleanup", flag: "-nc", type: "boolean", label: "Keep Work Dirs", help: "Keep w-* working directories and logs" },
      { name: "no_readme", flag: "--no-readme", type: "boolean", label: "Skip README" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs", help: "Skip per-GPU log files" },
      { name: "no_prompt", flag: "-np", type: "boolean", label: "Headless", help: "Use defaults instead of prompting for README metadata" },
      { name: "no_graph", flag: "-ng", type: "boolean", label: "No Graph", help: "Skip SVG graph generation" },
      { name: "no_measurement", flag: "-nm", type: "boolean", label: "No Measurement", help: "Skip KL/PPL measurement entirely" },
      // Evals section — KL and PPL are on by default (inverted: emit flag when OFF)
      { name: "no_kl", flag: "--no-kl", type: "boolean", label: "KL Divergence", help: "KL divergence measurement", section: "evals", defaultOn: true, invertFlag: true },
      { name: "no_ppl", flag: "--no-ppl", type: "boolean", label: "Perplexity", help: "Perplexity measurement", section: "evals", defaultOn: true, invertFlag: true },
      { name: "catbench", flag: "-cb", type: "number", label: "Catbench", placeholder: "3", help: "SVG Catbench samples per BPW", toggleable: true, section: "evals" },
      { name: "diversity", flag: "-div", type: "number", label: "Diversity", placeholder: "50", help: "Output diversity eval (N samples)", toggleable: true, section: "evals" },
      { name: "humaneval", flag: "-he", type: "number", label: "HumanEval", placeholder: "200", help: "Code generation eval (N samples/task)", toggleable: true, section: "evals" },
      { name: "ifbench", flag: "-ifb", type: "number", label: "IFBench", placeholder: "16384", help: "Instruction following eval (max tokens)", toggleable: true, section: "evals" },
      { name: "longctx", flag: "-lctx", type: "boolean", label: "Long Context", help: "Long context understanding eval", section: "evals" },
      { name: "mmlu", flag: "-mmlu", type: "number", label: "MMLU", placeholder: "5", help: "Knowledge benchmark (N fewshot examples)", toggleable: true, section: "evals" },
      { name: "perf", flag: "-perf", type: "number", label: "Perf", placeholder: "32768", help: "Inference performance benchmark (max length)", toggleable: true, section: "evals" },
    ],
  },

  quantize: {
    label: "Quantize",
    subtitle: "Quantize Only",
    description: "Run quantization without measurement or README",
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "BF16/base model directory" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "Target bits per weight",
        bpwPaintFlags: [
          { name: "hq", flag: "-hq", label: "-hq", color: "#4a90d9" },
          { name: "hb8", flag: "-hb8", label: "-hb 8", color: "#43a047" },
        ]
      },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1" },
      { name: "device_ratios", flag: "-r", type: "csv", label: "Device Ratios", placeholder: "1,1" },
      { name: "out_template", flag: "--out-template", type: "text", default: "{model}/{bpw}", label: "Output Template", help: "Fields: {model}, {model_name}, {bpw}" },
      { name: "w_template", flag: "--w-template", type: "text", default: "{model}/w-{bpw}", label: "Work Dir Template", help: "Fields: {model}, {model_name}, {bpw}" },
      { name: "layers", flag: "-l", type: "select", choices: ["1", "2", "3"], default: "2", label: "Optimization Depth", help: "Layer depth for optimization", toggleable: true },
      { name: "dry", flag: "--dry", type: "boolean", label: "Dry Run", help: "Print commands without executing" },
      { name: "continue_on_error", flag: "--continue-on-error", type: "boolean", label: "Continue on Error" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs" },
    ],
  },

  measure: {
    label: "Measure",
    subtitle: "Measure Only",
    description: "Run KL divergence + perplexity measurement",
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "Model directory with quantized outputs" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6" },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs" },
      { name: "no_cleanup", flag: "-nc", type: "boolean", label: "Keep Temp Files" },
      // Evals section — KL and PPL are on by default (inverted: emit flag when OFF)
      { name: "no_kl", flag: "--no-kl", type: "boolean", label: "KL Divergence", help: "KL divergence measurement", section: "evals", defaultOn: true, invertFlag: true },
      { name: "no_ppl", flag: "--no-ppl", type: "boolean", label: "Perplexity", help: "Perplexity measurement", section: "evals", defaultOn: true, invertFlag: true },
      { name: "catbench", flag: "-cb", type: "number", label: "Catbench", placeholder: "3", help: "SVG Catbench samples per BPW", toggleable: true, section: "evals" },
      { name: "diversity", flag: "-div", type: "number", label: "Diversity", placeholder: "50", help: "Output diversity eval (N samples)", toggleable: true, section: "evals" },
      { name: "humaneval", flag: "-he", type: "number", label: "HumanEval", placeholder: "200", help: "Code generation eval (N samples/task)", toggleable: true, section: "evals" },
      { name: "ifbench", flag: "-ifb", type: "number", label: "IFBench", placeholder: "16384", help: "Instruction following eval (max tokens)", toggleable: true, section: "evals" },
      { name: "longctx", flag: "-lctx", type: "boolean", label: "Long Context", help: "Long context understanding eval", section: "evals" },
      { name: "mmlu", flag: "-mmlu", type: "number", label: "MMLU", placeholder: "5", help: "Knowledge benchmark (N fewshot examples)", toggleable: true, section: "evals" },
      { name: "perf", flag: "-perf", type: "number", label: "Perf", placeholder: "32768", help: "Inference performance benchmark (max length)", toggleable: true, section: "evals" },
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
      { name: "mode", flag: "--mode", type: "select", choices: ["branched", "single"], default: "branched", label: "Mode", help: "Branched: single README. Single: per-BPW READMEs with cross-linked repos" },
      { name: "template", flag: "-t", type: "template", label: "Template", help: "README template style" },
      { name: "no_prompt", flag: "-np", type: "boolean", label: "Headless", help: "Use defaults for metadata" },
      { name: "no_graph", flag: "-ng", type: "boolean", label: "No Graph" },
      { name: "no_measurement", flag: "-nm", type: "boolean", label: "No Measurement", help: "Remove KL/PPL columns" },
    ],
  },

  upload: {
    label: "Upload",
    subtitle: "HuggingFace",
    description: "Create repos and upload to HuggingFace",
    twoActions: true,
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "Directory with quantized models" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "BPWs to upload" },
      { name: "mode", flag: "--mode", type: "select", choices: ["branched", "single"], default: "branched", label: "Mode", help: "Branched: one repo with branches. Single: separate repo per BPW" },
      // Boolean flags
      { name: "private", flag: "--private", type: "boolean", label: "Private Repos", help: "Create private HuggingFace repos" },
      { name: "small_only", flag: "--small-only", type: "boolean", label: "Small Files Only", help: "Exclude *.safetensors, *.bin, *.pt, *.ckpt" },
    ],
  },
};
