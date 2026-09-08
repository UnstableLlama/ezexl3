"""Upload quantized models to HuggingFace Hub."""

import os
import shutil
import sys
import threading
import time
from typing import List, Optional


LARGE_FILE_PATTERNS = ["*.safetensors", "*.bin", "*.pt", "*.ckpt"]


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
    """Find shared artifact files/dirs in the model root.

    The qbench charts are matched by exact name rather than by extension:
    they are copied up from qbench/, which also holds the logit cache, and a
    blanket *.png sweep would pick up unrelated images from the source
    checkpoint.
    """
    from ezexl3.qbench import README_CHARTS

    found = []
    for item in os.listdir(model_dir):
        path = os.path.join(model_dir, item)
        if item == "catbench" and os.path.isdir(path):
            found.append(item)
        elif item == "evals" and os.path.isdir(path):
            found.append(item)
        elif item.endswith((".csv", ".svg")) or item in README_CHARTS:
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
        # Clear the progress line so terminal output isn't stuck
        print("", flush=True)


def _upload_folder_graceful(api, folder_path: str, repo_id: str, commit_message: str,
                            revision: Optional[str] = None,
                            ignore_patterns: Optional[List[str]] = None) -> bool:
    """Upload a folder, retrying without README.md if HF rejects metadata."""
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            revision=revision,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
        )
        return True
    except Exception as e:
        err = str(e)
        if "Invalid metadata in README.md" in err or "base_model" in err:
            # HF rejected the README frontmatter — retry without it
            print(f"  ⚠️  README.md has invalid HF metadata, uploading without it")
            retry_ignore = list(ignore_patterns or []) + ["README.md"]
            try:
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    revision=revision,
                    commit_message=commit_message,
                    ignore_patterns=retry_ignore,
                )
                return True
            except Exception as e2:
                print(f"  🔴 Failed even without README: {e2}")
                return False
        else:
            raise


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
        try:
            api.create_branch(repo_id, branch=label, exist_ok=True)
        except Exception as e:
            print(f"  ⚠️  Could not create branch {label}: {e}")

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
    created = 0
    for bpw in bpws:
        label = _format_bpw(bpw)
        repo_id = f"{user}/{model}-exl3-{label}"
        print(f"📦 Creating repo: {repo_id} (private={private})")
        try:
            api.create_repo(repo_id, private=private, exist_ok=True)
            created += 1
        except Exception as e:
            print(f"  ⚠️  Could not create {repo_id}: {e}")

    print(f"✅ Created {created}/{len(bpws)} repos")


# ── Upload ────────────────────────────────────────────────────────

def upload_branched(
    model_dir: str,
    repo_id: str,
    bpws: List[str],
    small_only: bool = False,
) -> None:
    """Upload to a single repo with branches.

    Uploads root-level artifacts (README, graph, CSV, catbench, evals)
    to main, then each BPW subdirectory to its branch.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    ignore = _get_ignore_patterns(small_only)
    model_dir = os.path.abspath(model_dir)

    # Upload main branch: README + any shared artifacts
    artifacts = _find_shared_artifacts(model_dir)
    readme_path = os.path.join(model_dir, "README.md")
    if os.path.exists(readme_path):
        artifacts.append("README.md")

    if artifacts:
        print("📤 Uploading main branch artifacts...")
        for item in artifacts:
            path = os.path.join(model_dir, item)
            try:
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
            except Exception as e:
                err = str(e)
                if item == "README.md" and ("Invalid metadata" in err or "base_model" in err):
                    print(f"  ⚠️  README.md has invalid HF metadata — skipped")
                    print(f"       Fix the base_model field in README frontmatter and re-upload")
                else:
                    print(f"  ⚠️  Failed to upload {item}: {e}")
    else:
        print("ℹ️  No shared artifacts found in model root")

    # Upload each BPW to its branch
    for bpw in bpws:
        label = _format_bpw(bpw)
        bpw_dir = os.path.join(model_dir, bpw)
        if not os.path.isdir(bpw_dir):
            print(f"⚠️  Skipping {label}: directory {bpw_dir} not found")
            continue

        print(f"📤 Uploading {label} to branch...")
        try:
            ok = _upload_folder_graceful(
                api, bpw_dir, repo_id,
                commit_message=f"Upload {label} quantization",
                revision=label,
                ignore_patterns=ignore or None,
            )
            if ok:
                print(f"  ✅ {label} uploaded")
        except Exception as e:
            print(f"  🔴 Failed to upload {label}: {e}")


def upload_single(
    model_dir: str,
    user: str,
    model: str,
    bpws: List[str],
    small_only: bool = False,
) -> None:
    """Upload to separate per-BPW repos.

    Expects READMEs already generated in each BPW subdirectory
    (via `ezexl3 readme --mode single`). Copies shared artifacts
    (CSV, SVG, catbench, evals) into each BPW folder temporarily
    for the upload, then cleans up.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    ignore = _get_ignore_patterns(small_only)
    model_dir = os.path.abspath(model_dir)
    artifacts = _find_shared_artifacts(model_dir)

    for bpw in bpws:
        label = _format_bpw(bpw)
        repo_id = f"{user}/{model}-exl3-{label}"
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
                continue
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
            ok = _upload_folder_graceful(
                api, bpw_dir, repo_id,
                commit_message=f"Upload {label} quantization",
                ignore_patterns=ignore or None,
            )
            if ok:
                print(f"  ✅ {repo_id} uploaded")
        except Exception as e:
            print(f"  🔴 Failed to upload {repo_id}: {e}")
        finally:
            # Clean up temporarily copied artifacts
            for dst in copied:
                try:
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                except Exception:
                    pass


# ── Main entry point ──────────────────────────────────────────────

def run_upload(
    model_dir: str,
    bpws: List[str],
    mode: str = "branched",
    private: bool = False,
    small_only: bool = False,
    create_only: bool = False,
    dry_run: bool = False,
) -> int:
    """Top-level upload orchestrator. Agnostic to folder contents."""
    from ezexl3.readme import _compute_defaults, _read_saved_metadata

    model_dir = os.path.abspath(model_dir)
    model_name = os.path.basename(model_dir)

    # Auth check (ensures the user is logged in to HF; the namespace used for
    # the upload can still be overridden via saved metadata — e.g. pushing to
    # an org the authenticated user belongs to). Skipped in dry-run so users
    # can preview repo layouts without logging in.
    authed_user = "" if dry_run else check_hf_auth()

    # Prefer saved README metadata (MODEL + USER) over computed defaults so
    # the dashboard's upload metadata panel drives the repo naming. This is
    # the same .ezexl3_readme_meta.json that the README tab writes.
    saved = _read_saved_metadata(model_dir) or {}
    defaults = _compute_defaults(model_dir)
    model = (saved.get("MODEL") or "").strip() or defaults.get("MODEL", model_name)
    hf_user = (saved.get("USER") or "").strip() or authed_user or "<USER>"
    if authed_user and hf_user != authed_user:
        print(f"ℹ️  Uploading under namespace '{hf_user}' (authenticated as '{authed_user}')")

    # Normalize BPWs
    bpw_labels = [_format_bpw(b) for b in bpws]

    print(f"\n{'='*60}")
    if dry_run:
        print(f"🧪 DRY RUN — no repos will be created, no files will be uploaded")
    print(f"Upload: {model_name}")
    print(f"Mode: {'BRANCHED (single repo, branches per BPW)' if mode == 'branched' else 'SINGLE (separate repo per BPW)'}")
    print(f"BPWs: {', '.join(bpw_labels)}")
    print(f"Private: {private}")
    print(f"Small files only: {small_only}")
    if create_only:
        print(f"Action: Create repos only")

    # Show exactly what repos will be created
    if mode == "branched":
        repo_id = f"{hf_user}/{model}-exl3"
        print(f"\nRepo: https://huggingface.co/{repo_id}")
        print(f"Branches: {', '.join(bpw_labels)}")
    else:
        print(f"\nRepos:")
        for label in bpw_labels:
            print(f"  https://huggingface.co/{hf_user}/{model}-exl3-{label}")
    print(f"{'='*60}\n")

    if dry_run:
        print("🧪 Dry run complete. Re-run without --dry-run to create and upload.")
        return 0

    # Pre-flight: make sure the model directory and every BPW subdir the
    # user asked for actually exist before we touch HuggingFace. Prevents
    # the footgun where repos get created on HF and then every BPW is
    # skipped because the model_dir was stale or wrong.
    if not create_only:
        if not os.path.isdir(model_dir):
            print(f"🔴 Model directory does not exist: {model_dir}")
            return 1
        missing = [b for b in bpws if not os.path.isdir(os.path.join(model_dir, b))]
        if missing:
            print(f"🔴 Missing BPW subdirectories in {model_dir}:")
            for b in missing:
                print(f"     {os.path.join(model_dir, b)}")
            print(f"   Aborting BEFORE creating any HuggingFace repos.")
            print(f"   Check the Model Directory — it should contain a subfolder per BPW.")
            return 1

    if mode == "branched":
        repo_id = f"{hf_user}/{model}-exl3"
        create_repos_branched(repo_id, bpws, private=private)

        if not create_only:
            monitor = BandwidthMonitor()
            monitor.start()
            try:
                upload_branched(model_dir, repo_id, bpws, small_only=small_only)
            finally:
                monitor.stop()

    elif mode == "single":
        create_repos_single(hf_user, model, bpws, private=private)

        if not create_only:
            monitor = BandwidthMonitor()
            monitor.start()
            try:
                upload_single(model_dir, hf_user, model, bpws, small_only=small_only)
            finally:
                monitor.stop()

    print(f"\n✅ Upload complete!")
    return 0
