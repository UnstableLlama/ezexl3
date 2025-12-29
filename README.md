# ezexl3

**ezexl3** is a simple, single-command EXL3 repo generator.

It wraps the EXL3 quantization and evaluation workflow into a tool that:
- runs batch quantization safely (resume / skip supported),
- measures quality (PPL + KL) reproducibly,
- scales across multiple GPUs without CSV corruption,
- and produces HF-ready artifacts with minimal effort.

This is designed for people who *want the results of EXL3 quantization* without having to wire everything together themselves.

---

## Usage

### 1. Build a full repository
Run the entire pipeline (quantize -> measure -> report):
```bash
ezexl3 repo -m /path/to/base_model -b 2,3,4,5,6 -d 0,1
```

### 2. Standalone subcommands
If you only want to run specific stages:
```bash
# Quantize only
ezexl3 quantize -m /path/to/base_model -b 2,3,4,5,6 -d 0,1

# Measure only
ezexl3 measure -m /path/to/base_model -b 2,3,4,5,6 -d 0,1
```

## Key Features

- **Instant Startup**: CLI responds in ~0.05s by deferring heavy ML imports.
- **Asynchronous Measurement**: GPU workers pull from a dynamic queue and report results in real-time.
- **Auto-Merge**: Measurement results are merged into a canonical CSV as they arrive.
- **Resume Support**: Quantization skips already-finished BPWs and resumes partial jobs.

## Development Status

- ✅ Quantization: Stable, supports resume and multi-GPU ratios.
- ✅ Measurement: Stable, multi-GPU safe, asynchronous reporting.
- 🟡 Reporting: Stub implemented; logic for automated README generation pending.
- 🟡 Cleanup: Stub implemented; `--cleanup` flag logic pending.
