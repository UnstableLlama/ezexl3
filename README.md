ezexl3 is a small CLI tool that turns a BF16 model directory into a ready-to-upload EXL3 quantized repo using a single command.

It handles:

-batch quantization across multiple bitrates

-automatic resume if something crashes

-automated measurements

-clean output layout

Quality quantizations quickly!

git add pyproject.toml ezexl3/ README.md LICENSE
git commit -m "Initial ezexl3 CLI with integrated EXL3 quantization" \
            -m "- Add pip-installable ezexl3 CLI with subcommands
- Integrate EXL3 quantization via native Python wrapper
- Call exllamav3 conversion APIs directly (no subprocesses)
- Preserve EXL3 skip/resume semantics via config.json and args.json
- Support multi-model, multi-BPW batch quantization
- Implement passthrough argument blocks (--quant-args -- ...)
- Add dry-run mode for safe inspection"

