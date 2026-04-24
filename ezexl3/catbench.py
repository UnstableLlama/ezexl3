# catbench.py - SVG Catbench for ezexl3
# Prompts models to draw a kitten using matplotlib, extracts SVG output.
# Runnable as: python -m ezexl3.catbench

import sys
import os
import re
import argparse
import subprocess
import tempfile
import time

CATBENCH_PROMPT = "Write a python script that draws a cute kitten using matplotlib."
DEFAULT_MAX_NEW_TOKENS = 4096


def build_prompt_and_stops(mode: str, tokenizer, config, user_name: str = "User", bot_name: str = "Assistant"):
    """Frame CATBENCH_PROMPT in the given chat template and collect stop conditions.

    Returns (input_ids, stop_conditions, resolved_mode).
    """
    from ezexl3.chat.templates import prompt_formats

    pf_cls = prompt_formats.get(mode, prompt_formats["chatml"])
    pf = pf_cls(user_name, bot_name)
    system_prompt = pf.default_system_prompt(False)
    frm_context = pf.format(system_prompt, [(CATBENCH_PROMPT, None)], False)

    input_ids = tokenizer.encode(
        frm_context,
        add_bos=pf.add_bos(),
        encode_special_tokens=True,
    )

    stops = [sc for sc in pf.stop_conditions(tokenizer) if sc is not None]
    if getattr(config, "eos_token_id_list", None):
        for s in config.eos_token_id_list:
            if s is not None and s not in stops:
                stops.append(s)
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None and tok_eos not in stops:
        stops.append(tok_eos)

    return input_ids, stops, mode

# ---------------------------------------------------------------------------
# VRAM pre-flight
# ---------------------------------------------------------------------------

def _safetensors_size_gib(model_dir: str) -> float:
    """Total .safetensors size in GiB (non-recursive)."""
    total = 0
    if not os.path.isdir(model_dir):
        return 0.0
    for fn in os.listdir(model_dir):
        if fn.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(model_dir, fn))
    return total / (1024 ** 3)


def check_vram_fit(model_dir: str, device: int, headroom_gib: float = 2.0):
    """Check if a model's safetensors fit on a single GPU.

    Returns (fits, model_gib, available_gib).
    """
    import torch

    model_gib = _safetensors_size_gib(model_dir)
    props = torch.cuda.get_device_properties(device)
    total_gib = props.total_memory / (1024 ** 3)
    allocated_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
    available_gib = total_gib - allocated_gib

    fits = (model_gib + headroom_gib) <= available_gib
    return fits, model_gib, available_gib


def check_multi_gpu_fit(model_dir: str, devices: list, headroom_gib: float = 2.0):
    """Check if a model fits across multiple GPUs combined.

    Returns (fits, model_gib, total_available_gib).
    """
    import torch

    model_gib = _safetensors_size_gib(model_dir)
    total_available = 0.0
    for d in devices:
        props = torch.cuda.get_device_properties(d)
        total_gib = props.total_memory / (1024 ** 3)
        allocated_gib = torch.cuda.memory_allocated(d) / (1024 ** 3)
        total_available += total_gib - allocated_gib

    fits = (model_gib + headroom_gib) <= total_available
    return fits, model_gib, total_available


# ---------------------------------------------------------------------------
# SVG extraction
# ---------------------------------------------------------------------------

_SVG_BLOCK_RE = re.compile(r"(<svg[\s\S]*?</svg>)", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```")
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def extract_svg(text: str) -> str | None:
    """Extract SVG content from a model response.

    Multi-pass approach:
      1. Strip <think> blocks
      2. Look for raw <svg>...</svg> tags
      3. Extract fenced code blocks and run matplotlib ones
      4. If still nothing, strip code fences and try running the
         remaining text as bare Python (handles models that emit
         code without proper fencing)
    """
    # Strip think blocks first
    text = _THINK_BLOCK_RE.sub("", text)

    # Direct SVG in response
    m = _SVG_BLOCK_RE.search(text)
    if m:
        return m.group(1)

    # Try extracting and running fenced Python code blocks
    for code_match in _CODE_BLOCK_RE.finditer(text):
        code = code_match.group(1)
        if "matplotlib" not in code and "plt" not in code:
            continue
        svg = _run_matplotlib_code(code)
        if svg:
            return svg

    # Last resort: strip fences and try running as bare Python
    bare = text.replace("```python", "").replace("```", "").strip()
    if ("matplotlib" in bare or "plt" in bare) and bare != text.strip():
        svg = _run_matplotlib_code(bare)
        if svg:
            return svg

    return None


def _run_matplotlib_code(code: str) -> str | None:
    """Execute matplotlib code in a subprocess to produce SVG."""
    # Strip plt.show() — it clears the figure before our savefig
    code = re.sub(r"plt\.show\(\)", "", code)

    # Auto-call defined functions that aren't invoked at module level
    func_defs = re.findall(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    func_calls = ""
    for fn in func_defs:
        # Check if fn() appears at the start of a line (not indented = module level call)
        if not re.search(rf"^{re.escape(fn)}\s*\(", code, re.MULTILINE):
            func_calls += f"{fn}()\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = os.path.join(tmpdir, "output.svg")

        # Patch the code to save to our SVG path
        wrapper = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            f"{code}\n"
            f"{func_calls}"
            f"plt.savefig({svg_path!r}, format='svg', bbox_inches='tight')\n"
            "plt.close('all')\n"
        )
        script_path = os.path.join(tmpdir, "render.py")
        with open(script_path, "w") as f:
            f.write(wrapper)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=30,
                cwd=tmpdir,
            )
            if result.returncode == 0 and os.path.exists(svg_path):
                with open(svg_path, "r") as f:
                    return f.read()
        except subprocess.TimeoutExpired:
            pass
        except OSError:
            pass

    return None


# ---------------------------------------------------------------------------
# Core catbench runner
# ---------------------------------------------------------------------------


def run_catbench(args) -> list:
    """Run catbench against a single model, saving N raw .txt samples.

    SVG extraction is handled later in batch by the orchestrator.

    Prints progress markers for the parent process to parse:
      CATBENCH_MODEL_LOADED   — model loaded
      CATBENCH_SAMPLE_DONE i/n — sample i of n complete

    Returns list of saved .txt paths.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    # Defensive: if exllamav3 ends up as a PEP-420 namespace package, the
    # top-level re-exports aren't available. Fall back to explicit submodules.
    try:
        from exllamav3 import Generator, Job
    except ImportError:
        from exllamav3.generator import Generator, Job
    try:
        from exllamav3 import model_init
    except ImportError:
        import exllamav3.model_init as model_init
    from exllamav3.util.memory import free_mem

    n_samples = args.n_samples
    output_dir = args.output_dir
    label = args.label
    max_new_tokens = args.max_new_tokens

    torch.set_grad_enabled(False)
    os.makedirs(output_dir, exist_ok=True)

    # Determine file naming
    if label == "bf16" or label == "base":
        file_prefix = "bf16"
    else:
        try:
            val = float(label)
            file_prefix = f"{val:.2f}bpw"
        except (ValueError, TypeError):
            file_prefix = label

    print(f" -- Loading model from: {args.model_dir}", flush=True)

    # Load model via model_init (handles device mapping, cache sizing, multi-GPU)
    model, config, cache, tokenizer = model_init.init(args)

    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
    )

    print("CATBENCH_MODEL_LOADED", flush=True)

    # Resolve prompt-format mode (explicit --mode wins, else infer from path).
    from ezexl3.chat.templates import infer_mode_from_path

    mode = getattr(args, "mode", None) or infer_mode_from_path(args.model_dir)
    print(f" -- Prompt format: {mode}", flush=True)

    input_ids, stop_conditions, _ = build_prompt_and_stops(
        mode, tokenizer, config,
    )

    print(f" -- Cache: {cache.max_num_tokens} tokens", flush=True)
    print(f" -- Stop conditions: {stop_conditions}", flush=True)
    print(f" -- Input tokens: {input_ids.shape[-1]}", flush=True)
    print(f" -- Max new tokens: {max_new_tokens}", flush=True)

    saved_paths = []

    for i in range(1, n_samples + 1):
        # First sample is canonical (no suffix); subsequent get _1, _2, ...
        if i == 1:
            txt_path = os.path.join(output_dir, f"{file_prefix}.txt")
        else:
            txt_path = os.path.join(output_dir, f"{file_prefix}_{i - 1}.txt")

        # Skip if .txt already exists (inference already done for this sample)
        if os.path.exists(txt_path):
            saved_paths.append(txt_path)
            print(f" -- Sample {i}: already exists, skipping", flush=True)
            print(f"CATBENCH_SAMPLE_DONE {i}/{n_samples}", flush=True)
            continue

        print(f"CATBENCH_SAMPLE_START {i}/{n_samples}", flush=True)

        # Run inference
        job = Job(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions if stop_conditions else None,
        )
        generator.enqueue(job)

        token_count = 0
        t_start = time.time()
        response_chunks = []
        r = None
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                chunk = r.get("text", "")
                if chunk:
                    response_chunks.append(chunk)
                token_ids = r.get("token_ids")
                if token_ids is not None:
                    prev = token_count
                    token_count += token_ids.shape[-1]
                    if token_count // 100 > prev // 100:
                        elapsed = time.time() - t_start
                        tps = token_count / elapsed if elapsed > 0 else 0
                        print(f"CATBENCH_TOKENS {token_count} {tps:.1f}", flush=True)

        elapsed = time.time() - t_start
        tps = token_count / elapsed if elapsed > 0 else 0
        eos_reason = r.get("eos_reason", "unknown") if r else "unknown"
        print(f" -- Sample {i}: {token_count} tokens in {elapsed:.1f}s ({tps:.1f} t/s), stopped: {eos_reason}", flush=True)

        response = "".join(response_chunks)

        # Save raw response as .txt (always), then attempt SVG extraction
        # inline so the dashboard can show results live. The end-of-phase
        # batch pass in repo.py remains as an idempotent safety net that
        # re-normalizes numbering in case per-sample extraction failed.
        with open(txt_path, "w") as f:
            f.write(response)
        saved_paths.append(txt_path)
        print(f" -- Sample {i}: saved .txt ({len(response)} chars)", flush=True)

        svg_content = extract_svg(response)
        if svg_content:
            # First successful extraction for this prefix → canonical name.
            # Subsequent successes get _1, _2, … appended.
            canonical = os.path.join(output_dir, f"{file_prefix}.svg")
            if not os.path.exists(canonical):
                svg_path = canonical
            else:
                idx = 1
                while os.path.exists(
                    os.path.join(output_dir, f"{file_prefix}_{idx}.svg")
                ):
                    idx += 1
                svg_path = os.path.join(output_dir, f"{file_prefix}_{idx}.svg")
            with open(svg_path, "w") as f:
                f.write(svg_content)
            print(
                f" -- Sample {i}: SVG extracted → {os.path.basename(svg_path)} "
                f"({len(svg_content)} chars)",
                flush=True,
            )
        else:
            print(f" -- Sample {i}: no SVG extracted (batch pass will retry)", flush=True)

        print(f"CATBENCH_SAMPLE_DONE {i}/{n_samples}", flush=True)

    # Cleanup
    del generator, cache, model, config
    free_mem()

    print(f" -- Catbench complete: {n_samples} samples saved", flush=True)
    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from exllamav3 import model_init
    except ImportError:
        import exllamav3.model_init as model_init

    parser = argparse.ArgumentParser(description="Run SVG Catbench on a model")
    model_init.add_args(parser, cache=True, add_sampling_args=False)
    parser.add_argument("-n", "--n_samples", type=int, default=3, help="Number of samples per model")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for SVGs")
    parser.add_argument("-l", "--label", type=str, required=True, help="BPW label for file naming")
    parser.add_argument("-maxr", "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens per response")
    parser.add_argument("-pf", "--mode", type=str, default=None,
                        help="Chat prompt format (e.g. qwen35, chatml, gemma4). Auto-detected from model dir if omitted.")
    _args = parser.parse_args()
    run_catbench(_args)
