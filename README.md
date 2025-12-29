# ezexl3

**ezexl3** is a simple, single-command EXL3 repo generator.

It wraps the EXL3 quantization and evaluation workflow into a tool that:
- runs batch quantization easily (resume / skip supported),
- measures quality (PPL + K/L) efficiently, data recorded to .csv,
- (todo) and produces HF-ready artifacts with minimal effort.

This is designed for people who *want the results of EXL3 quantization* without having to wire everything together themselves.

---

## Usage

### 1. Quantize a full repository
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
- **Quantization Queuing**: Run multiple quantizations at a time with one command.
- **Auto-Measure**: Autotmaitcally measure and record K/L div @ -r 10 and ppl -r 100, layer by layer, memory effiently.
- **Resume Support**: Quantization skips already-finished BPWs and resumes partial jobs.

## Development Status

- ✅ Quantization: Stable, supports resume and multi-GPU ratios.
- ✅ Measurement: Stable, multi-GPU safe, asynchronous reporting.
- 🟡 Reporting: Stub implemented; logic for automated README generation pending.
- 🟡 Cleanup: Stub implemented; `--cleanup` flag logic pending.
