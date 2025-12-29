# ezexl3

**ezexl3** is a simple, single-command EXL3 repo generator.

It wraps the EXL3 quantization and evaluation workflow into a tool that:
- Runs batch quantization easily (resume / skip supported).
- Measures quality (PPL + K/L) efficiently, recording data to CSV.
- Automatically generates HuggingFace-ready `README.md` with your measurements using customizable templates.

All with one command.

---

## Usage

### 1. Quantize a full repository
Run the entire pipeline (quantize -> measure -> README):
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

# Generate README only (from existing CSV)
ezexl3 readme -m /path/to/base_model -t fire
```

### 3. Template System
You can customize the generated README by providing a template name via `--template` or `-t`. Templates are stored in the `templates/` directory.

The system is flexible with naming. For example, `-t fire` will search for:
- `templates/fire.md`
- `templates/fireTemplateREADME.md`
- `templates/fireREADME.md`
- `templates/fireTemplate.md`

If no template is specified, it defaults to `basicTemplateREADME.md`.

## Key Features
- **Quantization Queuing**: Run multiple quantizations at a time with one command.
- **Auto-Measure**: Automatically measure and record K/L div and PPL asynchronously.
- **Resume Support**: Skip already-finished BPWs and resume partial jobs.
- **Flexible README Templates**: Automated high-quality README generation for HuggingFace uploads.
- **Automatic Cleanup**: Default cleanup of temporary working directories and logs.

## Development Status
- ✅ Quantization: Stable, supports resume and multi-GPU ratios.
- ✅ Measurement: Stable, sharded multi-GPU execution.
- ✅ README Generation: Flexible template system with standalone subcommand.
- ✅ Cleanup: Automatic cleanup (toggle with `--no-cleanup`).
