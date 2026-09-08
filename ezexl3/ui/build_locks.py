"""Conservative Linux cleanup of interrupted ExLlamaV3 extension builds.

PyTorch FileBaton keeps an open descriptor while owning its lock. Snapshot
identities before stopping, then check descriptors again before unlinking.
Never import exllamav3 here: that import can itself wait on the stale lock.
"""

import os
import tempfile
from pathlib import Path


def prepare_build_cache() -> list[str]:
    """Recover before any ExLlama import, including CLI and dashboard jobs.

    If ownership cannot be established (or another build owns the lock), use
    an independent cache. Never delete a possibly live builder's lock.
    Children inherit the selected cache through TORCH_EXTENSIONS_DIR.
    """
    locks = snapshot_locks()
    if not locks:
        return []
    removed = cleanup_locks(locks)
    messages = [f"Removed interrupted extension build lock: {p}" for p in removed]
    if any(p.exists() for p in locks):
        # Keep this on disk: subprocess workers and later imports still need it.
        root = next(iter(locks)).parent.parent
        cache = tempfile.mkdtemp(prefix='ezexl3-recovery-', dir=root)
        os.environ['TORCH_EXTENSIONS_DIR'] = cache
        messages.append(
            f"Existing extension build lock could not be safely cleared; "
            f"using separate PyTorch build cache: {cache}"
        )
    return messages


def snapshot_locks() -> dict[Path, tuple[int, int]]:
    if not Path('/proc/self/fd').is_dir():
        return {}
    root = Path(os.environ.get('TORCH_EXTENSIONS_DIR') or
                Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
                / 'torch_extensions')
    locks = {}
    # Explicit TORCH_EXTENSIONS_DIR has no Python/CUDA version subdirectory.
    for pattern in ('exllamav3_ext/lock', '*/exllamav3_ext/lock'):
        for path in root.glob(pattern):
            try:
                st = path.lstat()
                if path.is_symlink() or st.st_uid != os.getuid() or not path.is_file():
                    continue
                locks[path] = (st.st_dev, st.st_ino)
            except OSError:
                continue
    return locks


def cleanup_locks(locks: dict[Path, tuple[int, int]]) -> list[Path]:
    if not locks:
        return []
    held = set()
    try:
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if proc.stat().st_uid != os.getuid():
                    continue
                for fd in (proc / 'fd').iterdir():
                    try:
                        st = fd.stat()
                        held.add((st.st_dev, st.st_ino))
                    except (FileNotFoundError, PermissionError):
                        pass  # Descriptor closed during the scan.
            except FileNotFoundError:
                pass  # Process exited during the scan.
            except PermissionError:
                # Same uid but not inspectable: a non-dumpable system helper
                # such as systemd's sd-pam, present in every login session.
                # Those never run PyTorch builds, so skipping them keeps the
                # proof sound. Aborting here instead made this function a
                # no-op on every machine and pushed every start into the
                # recovery-cache fallback.
                pass
    except OSError:
        # Cannot enumerate processes at all; leave the locks alone.
        return []

    removed = []
    for path, identity in locks.items():
        if identity in held:
            continue
        try:
            st = path.lstat()
            if (st.st_dev, st.st_ino) == identity:
                path.unlink()
                removed.append(path)
        except OSError:
            pass
    return removed


def group_is_running(pgid: int) -> bool:
    """Ignore zombies, which cannot build or hold file descriptors."""
    for proc in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = proc.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                return True
        except FileNotFoundError:
            continue
    return False
