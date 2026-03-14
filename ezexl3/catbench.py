# catbench.py - SVG Catbench for ezexl3
# Prompts models to draw a kitten using matplotlib, extracts SVG output.
# Runnable as: python -m ezexl3.catbench

import sys
import os
import re
import argparse
import shutil
import subprocess
import tempfile
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CATBENCH_PROMPT = "Write a python script that draws a cute kitten using matplotlib."
DEFAULT_MAX_NEW_TOKENS = 4096

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


def _clean_response(text: str) -> str:
    """Strip think tags and code fences from model response."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.replace("```python", "").replace("```", "")
    return text.strip()


def extract_svg(text: str) -> str | None:
    """Extract SVG content from a model response.

    Looks for raw <svg>...</svg> blocks first, then tries to execute
    Python code blocks that use matplotlib to produce SVG output.
    """
    # Direct SVG in response
    m = _SVG_BLOCK_RE.search(text)
    if m:
        return m.group(1)

    # Try extracting and running Python code blocks
    for code_match in _CODE_BLOCK_RE.finditer(text):
        code = code_match.group(1)
        if "matplotlib" not in code and "plt" not in code:
            continue
        svg = _run_matplotlib_code(code)
        if svg:
            return svg

    return None


def _run_matplotlib_code(code: str) -> str | None:
    """Execute matplotlib code in a subprocess to produce SVG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = os.path.join(tmpdir, "output.svg")

        # Patch the code to save to our SVG path
        wrapper = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            f"{code}\n"
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
        except (subprocess.TimeoutExpired, Exception):
            pass

    return None


# ---------------------------------------------------------------------------
# Core catbench runner
# ---------------------------------------------------------------------------


def run_catbench(args) -> list:
    """Run catbench against a single model, producing N SVG samples.

    Uses exllamav3's model_init for model loading (same as chat.py).

    Prints progress markers for the parent process to parse:
      CATBENCH_MODEL_LOADED   — model loaded, 80% of work done
      CATBENCH_SAMPLE_DONE i/n — sample i of n complete

    Returns list of paths to saved SVGs.
    """
    import torch
    from exllamav3 import Generator, Job, model_init
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

    # Build prompt - use simple format since we just need code generation
    prompt = CATBENCH_PROMPT
    input_ids = tokenizer.encode(prompt, add_bos=True)

    # Get stop conditions from config
    stop_conditions = []
    if hasattr(config, "eos_token_id_list") and config.eos_token_id_list:
        stop_conditions = [s for s in config.eos_token_id_list if s]
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
        if tokenizer.eos_token_id not in stop_conditions:
            stop_conditions.append(tokenizer.eos_token_id)

    print(f" -- Cache: {cache.max_num_tokens} tokens", flush=True)
    print(f" -- Stop conditions: {stop_conditions}", flush=True)
    print(f" -- Input tokens: {input_ids.shape[-1]}", flush=True)
    print(f" -- Max new tokens: {max_new_tokens}", flush=True)

    saved_paths = []
    canonical_path = os.path.join(output_dir, f"{file_prefix}.svg")
    first_svg_saved = False

    for i in range(1, n_samples + 1):
        sample_path = os.path.join(output_dir, f"{file_prefix}_{i}.svg")
        txt_path = os.path.join(output_dir, f"{file_prefix}_{i}.txt")

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

        # Clean up model response and extract SVG
        cleaned = _clean_response(response)
        svg_content = extract_svg(cleaned)
        if svg_content:
            with open(sample_path, "w") as f:
                f.write(svg_content)
            saved_paths.append(sample_path)
            print(f" -- Sample {i}: SVG saved ({len(svg_content)} chars)", flush=True)

            # First successful SVG becomes canonical
            if not first_svg_saved:
                shutil.copy2(sample_path, canonical_path)
                first_svg_saved = True
        else:
            # Save raw response for debugging
            with open(txt_path, "w") as f:
                f.write(response)
            print(f" -- Sample {i}: No SVG extracted, raw response saved", flush=True)

        print(f"CATBENCH_SAMPLE_DONE {i}/{n_samples}", flush=True)

    # Cleanup
    del generator, cache, model, config
    free_mem()

    if saved_paths:
        print(f" -- Catbench complete: {len(saved_paths)}/{n_samples} SVGs saved", flush=True)
    else:
        print(f" -- Catbench complete: no SVGs extracted", flush=True)

    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from exllamav3 import model_init

    parser = argparse.ArgumentParser(description="Run SVG Catbench on a model")
    model_init.add_args(parser, cache=True, add_sampling_args=False)
    parser.add_argument("-n", "--n_samples", type=int, default=3, help="Number of samples per model")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for SVGs")
    parser.add_argument("-l", "--label", type=str, required=True, help="BPW label for file naming")
    parser.add_argument("-maxr", "--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max tokens per response")
    _args = parser.parse_args()
    run_catbench(_args)
