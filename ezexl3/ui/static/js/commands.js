// ── Command schema definitions ──────────────────────────────────
// Each command maps to a CLI subcommand with typed fields that
// drive dynamic form generation in forms.js.

const COMMANDS = {
  repo: {
    label: "Repo",
    subtitle: "Full Pipeline",
    description: "Quantize \u2192 Measure KL+PPL \u2192 README",
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "BF16/base model directory" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "Target bits per weight" },
      { name: "hq", flag: "-hq", type: "boolean", label: "-hq", toggleable: true },
      { name: "devices", flag: "-d", type: "csv", default: "0", label: "CUDA Devices", placeholder: "0,1", help: "GPU device indices" },
      { name: "device_ratios", flag: "-r", type: "csv", label: "Device Ratios", placeholder: "1,1", help: "VRAM ratios per device (optional)" },
      { name: "template", flag: "-t", type: "template", label: "Template", help: "README template style" },
      { name: "layers", flag: "-l", type: "select", choices: ["1", "2", "3"], default: "2", label: "Optimization Depth", help: "Layer depth for optimization", toggleable: true },
      { name: "catbench", flag: "-cb", type: "number", label: "Catbench Samples", placeholder: "3", help: "SVG Catbench samples per BPW", toggleable: true },
      // Boolean flags
      { name: "no_verify", flag: "-nv", type: "boolean", label: "No Verify", help: "Batch mode: all quants then all measures" },
      { name: "no_cleanup", flag: "-nc", type: "boolean", label: "Keep Work Dirs", help: "Keep w-* working directories and logs" },
      { name: "no_readme", flag: "--no-readme", type: "boolean", label: "Skip README" },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs", help: "Skip per-GPU log files" },
      { name: "no_prompt", flag: "-np", type: "boolean", label: "Headless", help: "Use defaults instead of prompting for README metadata" },
      { name: "no_graph", flag: "-ng", type: "boolean", label: "No Graph", help: "Skip SVG graph generation" },
      { name: "no_measurement", flag: "-nm", type: "boolean", label: "No Measurement", help: "Skip KL/PPL measurement entirely" },
    ],
  },

  quantize: {
    label: "Quantize",
    subtitle: "Quantize Only",
    description: "Run quantization without measurement or README",
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "BF16/base model directory" },
      { name: "bpws", flag: "-b", type: "csv", required: true, label: "BPWs", placeholder: "2,3,4,5,6", help: "Target bits per weight" },
      { name: "hq", flag: "-hq", type: "boolean", label: "-hq", toggleable: true },
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
      { name: "catbench", flag: "-cb", type: "number", label: "Catbench Samples", placeholder: "3", help: "SVG Catbench samples per BPW", toggleable: true },
      { name: "no_logs", flag: "--no-logs", type: "boolean", label: "No Logs" },
      { name: "no_cleanup", flag: "-nc", type: "boolean", label: "Keep Temp Files" },
    ],
  },

  readme: {
    label: "README",
    subtitle: "Generate README",
    description: "Generate HuggingFace README from existing CSV",
    fields: [
      { name: "models", flag: "-m", type: "path", required: true, label: "Model Directory", help: "Directory with measurement CSV" },
      { name: "template", flag: "-t", type: "template", label: "Template", help: "README template style" },
      { name: "no_prompt", flag: "-np", type: "boolean", label: "Headless", help: "Use defaults for metadata" },
      { name: "no_graph", flag: "-ng", type: "boolean", label: "No Graph" },
      { name: "no_measurement", flag: "-nm", type: "boolean", label: "No Measurement", help: "Remove KL/PPL columns" },
    ],
  },
};
