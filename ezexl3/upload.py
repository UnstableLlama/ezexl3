"""Upload quantized models to HuggingFace Hub."""

import os
import re
import shutil
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional


LARGE_FILE_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.ckpt"]

# Shared artifacts that live in the model root (not in per-BPW subdirs)
SHARED_ARTIFACT_PATTERNS = [
    "*.csv", "*.svg", "*.db",
    "catbench/",
    "evals/",
]


def check_hf_auth() -> str:
    """Verify HuggingFace authentication. Returns username or exits."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().whoami()
        username = info.get("name", "")
        if not username:
            print("🔴 HuggingFace token found but no username returned.")
            sys.exit(1)
        print(f"✅ Authenticated as: {username}")
        return username
    except Exception:
        print("🔴 Not logged in to HuggingFace.")
        print("   Run: hf login")
        sys.exit(1)


def _format_bpw(bpw: str) -> str:
    """Format a BPW string to standard label like '4.00bpw'."""
    try:
        return f"{float(bpw):.2f}bpw"
    except ValueError:
        return bpw


def _get_ignore_patterns(small_only: bool) -> List[str]:
    """Return file patterns to exclude during upload."""
    if small_only:
        return list(LARGE_FILE_PATTERNS)
    return []


def _find_shared_artifacts(model_dir: str) -> List[str]:
    """Find shared artifact files/dirs in the model root."""
    found = []
    for item in os.listdir(model_dir):
        path = os.path.join(model_dir, item)
        if item == "catbench" and os.path.isdir(path):
            found.append(item)
        elif item == "evals" and os.path.isdir(path):
            found.append(item)
        elif item.endswith((".csv", ".svg")):
            found.append(item)
    return found


# ── Bandwidth monitoring ──────────────────────────────────────────

class BandwidthMonitor:
    """Background thread that reports network TX throughput."""

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _read_tx_bytes() -> int:
        """Read total TX bytes from /proc/net/dev."""
        total = 0
        try:
            with open("/proc/net/dev") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 10 or ":" not in parts[0]:
                        continue
                    iface = parts[0].rstrip(":")
                    if iface == "lo":
                        continue
                    total += int(parts[9])  # TX bytes is column 9
        except (OSError, ValueError):
            pass
        return total

    def _loop(self):
        prev = self._read_tx_bytes()
        while not self._stop.wait(self._interval):
            curr = self._read_tx_bytes()
            delta = curr - prev
            mb_s = delta / (1024 * 1024 * self._interval)
            if mb_s >= 0.01:
                print(f"\r<<EZEXL3:BANDWIDTH:{mb_s:.1f} MB/s>>", flush=True)
            prev = curr

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None


# ── Repo creation ────────────────────────────────────────────────

def create_repos_branched(
    repo_id: str,
    bpws: List[str],
    private: bool = False,
) -> None:
    """Create a single HF repo with a branch per BPW."""
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"📦 Creating repo: {repo_id} (private={private})")
    api.create_repo(repo_id, private=private, exist_ok=True)

    for bpw in bpws:
        label = _format_bpw(bpw)
        print(f"  🌿 Creating branch: {label}")
        api.create_branch(repo_id, branch=label, exist_ok=True)

    print(f"✅ Repo ready with {len(bpws)} branches")


def create_repos_single(
    user: str,
    model: str,
    bpws: List[str],
    private: bool = False,
) -> None:
    """Create one HF repo per BPW."""
    from huggingface_hub import HfApi

    api = HfApi()
    for bpw in bpws:
        label = _format_bpw(bpw)
        repo_id = f"{user}/{model}-{label}-exl3"
        print(f"📦 Creating repo: {repo_id} (private={private})")
        api.create_repo(repo_id, private=private, exist_ok=True)

    print(f"✅ Created {len(bpws)} repos")


# ── Upload ────────────────────────────────────────────────────────

def upload_branched(
    model_dir: str,
    repo_id: str,
    bpws: List[str],
    small_only: bool = False,
) -> None:
    """Upload to a single repo with branches."""
    from huggingface_hub import HfApi

    api = HfApi()
    ignore = _get_ignore_patterns(small_only)
    model_dir = os.path.abspath(model_dir)

    # Upload main branch: README, graph, CSV, catbench, evals
    readme_path = os.path.join(model_dir, "README.md")
    if os.path.exists(readme_path):
        print("📤 Uploading main branch artifacts...")
        # Collect root-level files (not BPW subdirs)
        artifacts = _find_shared_artifacts(model_dir)
        if os.path.exists(readme_path):
            artifacts.append("README.md")

        # Upload each artifact individually to main
        for item in artifacts:
            path = os.path.join(model_dir, item)
            if os.path.isdir(path):
                print(f"  📁 {item}/")
                api.upload_folder(
                    folder_path=path,
                    path_in_repo=item,
                    repo_id=repo_id,
                    commit_message=f"Upload {item}",
                )
            elif os.path.isfile(path):
                print(f"  📄 {item}")
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=item,
                    repo_id=repo_id,
                    commit_message=f"Upload {item}",
                )

    # Upload each BPW to its branch
    for bpw in bpws:
        label = _format_bpw(bpw)
        bpw_dir = os.path.join(model_dir, bpw)
        if not os.path.isdir(bpw_dir):
            print(f"⚠️  Skipping {label}: directory {bpw_dir} not found")
            continue

        print(f"📤 Uploading {label} to branch...")
        api.upload_folder(
            folder_path=bpw_dir,
            repo_id=repo_id,
            revision=label,
            commit_message=f"Upload {label} quantization",
            ignore_patterns=ignore or None,
        )
        print(f"  ✅ {label} uploaded")


def upload_single(
    model_dir: str,
    user: str,
    model: str,
    bpws: List[str],
    small_only: bool = False,
    template_name: Optional[str] = None,
    include_graph: bool = True,
    include_measurements: bool = True,
    include_catbench: bool = False,
) -> None:
    """Upload to separate per-BPW repos, generating per-BPW READMEs."""
    from huggingface_hub import HfApi

    api = HfApi()
    ignore = _get_ignore_patterns(small_only)
    model_dir = os.path.abspath(model_dir)

    # Generate per-BPW READMEs into each BPW subdirectory
    print("📝 Generating per-BPW READMEs...")
    _generate_single_bitrate_readmes(
        model_dir=model_dir,
        user=user,
        model=model,
        bpws=bpws,
        template_name=template_name,
        include_graph=include_graph,
        include_measurements=include_measurements,
        include_catbench=include_catbench,
    )

    # Copy shared artifacts into each BPW folder, then upload
    artifacts = _find_shared_artifacts(model_dir)

    for bpw in bpws:
        label = _format_bpw(bpw)
        repo_id = f"{user}/{model}-{label}-exl3"
        bpw_dir = os.path.join(model_dir, bpw)
        if not os.path.isdir(bpw_dir):
            print(f"⚠️  Skipping {label}: directory {bpw_dir} not found")
            continue

        # Copy shared artifacts into BPW dir temporarily
        copied = []
        for item in artifacts:
            src = os.path.join(model_dir, item)
            dst = os.path.join(bpw_dir, item)
            if os.path.exists(dst):
                continue  # don't overwrite existing files
            try:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
                copied.append(dst)
            except Exception as e:
                print(f"  ⚠️  Could not copy {item}: {e}")

        print(f"📤 Uploading to {repo_id}...")
        try:
            api.upload_folder(
                folder_path=bpw_dir,
                repo_id=repo_id,
                commit_message=f"Upload {label} quantization",
                ignore_patterns=ignore or None,
            )
            print(f"  ✅ {repo_id} uploaded")
        finally:
            # Clean up copied artifacts
            for dst in copied:
                try:
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                except Exception:
                    pass


# ── Single-bitrate README generation ──────────────────────────────

def _generate_single_bitrate_readmes(
    model_dir: str,
    user: str,
    model: str,
    bpws: List[str],
    template_name: Optional[str] = None,
    include_graph: bool = True,
    include_measurements: bool = True,
    include_catbench: bool = False,
) -> None:
    """Generate a modified README for each BPW in single-bitrate mode.

    Writes README.md into each {model_dir}/{bpw}/ subdirectory.
    """
    from ezexl3.readme import run_readme

    # First, generate the standard README into model_dir so we have a base
    run_readme(
        model_dir,
        template_name=template_name,
        interactive=False,
        include_graph=include_graph,
        include_measurements=include_measurements,
        bpws_hint=bpws,
        include_catbench=include_catbench,
    )

    base_readme = os.path.join(model_dir, "README.md")
    if not os.path.exists(base_readme):
        print("⚠️  Could not generate base README for single-bitrate mode")
        return

    with open(base_readme) as f:
        base_content = f.read()

    # The base README uses links like: USER/MODEL-exl3/tree/X.XXbpw
    # We need to rewrite these for each BPW's standalone repo
    quant_repo_base = f"{user}/{model}-exl3"

    for bpw in bpws:
        label = _format_bpw(bpw)
        content = base_content

        # 1. Update the YAML front matter base_model_relation metadata
        #    The title in <h1> says "AUTHOR / MODEL" — append the BPW
        content = re.sub(
            r'(<h1>)(.*?)(</h1>)',
            rf'\1\2 — {label}\3',
            content,
        )

        # 2. Rewrite data table links:
        #    FROM: href="https://huggingface.co/USER/MODEL-exl3/tree/X.XXbpw"
        #    TO:   href="https://huggingface.co/USER/MODEL-X.XXbpw-exl3"
        #    For the CURRENT bpw row, remove the <a> wrapper
        for other_bpw in bpws:
            other_label = _format_bpw(other_bpw)
            old_href = f"https://huggingface.co/{quant_repo_base}/tree/{other_label}"
            new_href = f"https://huggingface.co/{user}/{model}-{other_label}-exl3"

            if other_bpw == bpw:
                # Current BPW: replace <a> with bold plain text
                content = re.sub(
                    rf'<a class="link-style" href="{re.escape(old_href)}">{re.escape(other_label)}</a>',
                    f"<b>{other_label}</b>",
                    content,
                )
            else:
                # Sibling BPW: update link target
                content = content.replace(old_href, new_href)

        # 3. Rewrite the download command
        #    FROM: hf download USER/MODEL-exl3 --revision "X.XXbpw" --local-dir ./MODEL-exl3-X.XXbpw
        #    TO:   hf download USER/MODEL-X.XXbpw-exl3 --local-dir ./MODEL-X.XXbpw-exl3
        old_download = re.compile(
            rf'hf download {re.escape(quant_repo_base)} --revision "{re.escape(label)}" --local-dir \S+'
        )
        new_download = f"hf download {user}/{model}-{label}-exl3 --local-dir ./{model}-{label}-exl3"
        content = old_download.sub(new_download, content)

        # Also handle the case where DEFAULT_REVISION was used (first bpw label in the template)
        # Replace any remaining download references to the branched repo
        content = re.sub(
            rf'hf download {re.escape(quant_repo_base)} --revision "[^"]*" --local-dir \S+',
            new_download,
            content,
        )

        # Write into BPW subdirectory
        bpw_dir = os.path.join(model_dir, bpw)
        os.makedirs(bpw_dir, exist_ok=True)
        out_path = os.path.join(bpw_dir, "README.md")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"  📄 {label}/README.md")


# ── Main entry point ──────────────────────────────────────────────

def run_upload(
    model_dir: str,
    bpws: List[str],
    mode: str = "branched",
    private: bool = False,
    small_only: bool = False,
    create_only: bool = False,
    template_name: Optional[str] = None,
    include_graph: bool = True,
    include_measurements: bool = True,
    include_catbench: bool = False,
) -> int:
    """Top-level upload orchestrator."""
    from ezexl3.readme import _compute_defaults, get_hf_username

    model_dir = os.path.abspath(model_dir)
    model_name = os.path.basename(model_dir)

    # Auth check
    hf_user = check_hf_auth()

    # Compute metadata
    defaults = _compute_defaults(model_dir)
    user = hf_user
    model = defaults.get("MODEL", model_name)

    # Normalize BPWs
    bpw_labels = [_format_bpw(b) for b in bpws]

    print(f"\n{'='*60}")
    print(f"Upload: {model_name}")
    print(f"Mode: {mode}")
    print(f"BPWs: {', '.join(bpw_labels)}")
    print(f"Private: {private}")
    print(f"Small files only: {small_only}")
    if create_only:
        print(f"Action: Create repos only")
    print(f"{'='*60}\n")

    if mode == "branched":
        repo_id = f"{user}/{model}-exl3"

        # Create
        create_repos_branched(repo_id, bpws, private=private)

        if not create_only:
            # Upload
            monitor = BandwidthMonitor()
            monitor.start()
            try:
                upload_branched(model_dir, repo_id, bpws, small_only=small_only)
            finally:
                monitor.stop()

    elif mode == "single":
        # Create
        create_repos_single(user, model, bpws, private=private)

        if not create_only:
            # Upload
            monitor = BandwidthMonitor()
            monitor.start()
            try:
                upload_single(
                    model_dir, user, model, bpws,
                    small_only=small_only,
                    template_name=template_name,
                    include_graph=include_graph,
                    include_measurements=include_measurements,
                    include_catbench=include_catbench,
                )
            finally:
                monitor.stop()

    print(f"\n✅ Upload complete!")
    return 0
