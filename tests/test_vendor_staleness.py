"""
Vendor staleness checks for eval scripts vendored from exllamav3.

Offline test: verifies vendored files match the hashes recorded in
VENDOR_MANIFEST.json (catches accidental local edits).

Online test: fetches upstream scripts from GitHub and warns if they
have changed since we last vendored (marks as xfail, not hard failure).
"""

import hashlib
import json
import os

import pytest

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "ezexl3", "vendor")
_MANIFEST_PATH = os.path.join(_VENDOR_DIR, "VENDOR_MANIFEST.json")


def _load_manifest():
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


def _file_sha256(path: str) -> str:
    """Compute SHA-256 of a file's content, skipping the attribution header
    (first 3 lines starting with #, plus the blank line after)."""
    with open(path, "rb") as f:
        lines = f.readlines()
    # Skip the attribution header we prepended during vendoring.
    # The header is: comment, comment, comment, blank line.
    # Original content starts after that.
    idx = 0
    while idx < len(lines) and lines[idx].startswith(b"#"):
        idx += 1
    if idx < len(lines) and lines[idx].strip() == b"":
        idx += 1
    original_content = b"".join(lines[idx:])
    return hashlib.sha256(original_content).hexdigest()


class TestVendorManifestIntegrity:
    """Offline: vendored files match manifest hashes (no network needed)."""

    def test_manifest_exists(self):
        assert os.path.isfile(_MANIFEST_PATH), "VENDOR_MANIFEST.json is missing"

    def test_all_vendored_files_present(self):
        manifest = _load_manifest()
        for filename in manifest:
            path = os.path.join(_VENDOR_DIR, filename)
            assert os.path.isfile(path), f"Vendored file missing: {filename}"

    def test_vendored_hashes_match_manifest(self):
        """Ensure vendored files haven't been accidentally modified."""
        manifest = _load_manifest()
        for filename, meta in manifest.items():
            path = os.path.join(_VENDOR_DIR, filename)
            actual_hash = _file_sha256(path)
            assert actual_hash == meta["sha256"], (
                f"{filename}: local hash {actual_hash[:16]}... != manifest {meta['sha256'][:16]}... "
                f"(file was modified locally? Re-vendor from upstream to fix)"
            )


class TestVendorUpstreamStaleness:
    """Online: checks if upstream scripts have changed since we vendored.

    These tests are expected to fail (xfail) when upstream updates occur.
    They serve as a notification, not a hard gate.
    """

    @staticmethod
    def _fetch_upstream(url: str, timeout: int = 15) -> bytes:
        import urllib.request
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.read()

    @pytest.mark.parametrize("filename", _load_manifest().keys() if os.path.isfile(_MANIFEST_PATH) else [])
    def test_upstream_unchanged(self, filename):
        """Fetch upstream and compare hash. Warns (xfail) if changed."""
        manifest = _load_manifest()
        meta = manifest[filename]
        url = meta["source"]

        try:
            upstream_data = self._fetch_upstream(url)
        except Exception as e:
            pytest.skip(f"Network unavailable or URL failed: {e}")

        upstream_hash = hashlib.sha256(upstream_data).hexdigest()

        if upstream_hash != meta["sha256"]:
            pytest.xfail(
                f"UPSTREAM CHANGED: {filename} has been updated in exllamav3. "
                f"Local manifest hash: {meta['sha256'][:16]}..., "
                f"upstream hash: {upstream_hash[:16]}... "
                f"Re-vendor this script to pick up changes."
            )
