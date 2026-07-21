"""
Tests for vendored chat prompt format templates.

Offline tests verify structural contracts (every registered format is usable).
Online test fetches upstream chat_templates.py from exllamav3 and warns if
new format keys have been added that we haven't vendored yet.
"""

import re
import unittest

import pytest

from ezexl3.chat.templates import PromptFormat, prompt_formats


# ---------------------------------------------------------------------------
# Offline: structural contracts (no network)
# ---------------------------------------------------------------------------

class TestPromptFormatContracts(unittest.TestCase):
    """Every registered prompt format must satisfy the PromptFormat interface."""

    def test_all_formats_are_prompt_format_subclasses(self):
        for key, cls in prompt_formats.items():
            self.assertTrue(
                issubclass(cls, PromptFormat),
                f"prompt_formats[{key!r}] = {cls.__name__} is not a PromptFormat subclass",
            )

    def test_all_formats_have_description(self):
        for key, cls in prompt_formats.items():
            self.assertTrue(
                cls.description,
                f"prompt_formats[{key!r}] has empty description",
            )

    def test_all_formats_can_produce_output(self):
        """Instantiate each format and verify .format() returns a non-empty string."""
        messages = [("Hello", None)]
        for key, cls in prompt_formats.items():
            pf = cls("User", "Assistant")
            system = pf.default_system_prompt(think=False)
            output = pf.format(system, messages, think=False)
            self.assertIsInstance(output, str, f"{key}: format() did not return str")
            self.assertTrue(len(output) > 0, f"{key}: format() returned empty string")

    def test_all_formats_declare_stop_conditions(self):
        """stop_conditions() must return a non-empty list."""

        class FakeTokenizer:
            eos_token_id = 0

            def single_id(self, token_str):
                return hash(token_str) & 0xFFFF

        tok = FakeTokenizer()
        for key, cls in prompt_formats.items():
            pf = cls("User", "Assistant")
            stops = pf.stop_conditions(tok)
            self.assertIsInstance(stops, list, f"{key}: stop_conditions() not a list")
            self.assertTrue(len(stops) > 0, f"{key}: stop_conditions() empty")


# ---------------------------------------------------------------------------
# Offline: auto-detect heuristic coverage
# ---------------------------------------------------------------------------

class TestAutoDetectCoverage(unittest.TestCase):
    """The auto-detect heuristic should cover every registered format."""

    def test_every_format_key_reachable_from_heuristic_or_explicit(self):
        """Every key in prompt_formats should be either:
        - directly reachable from the auto-detect heuristic in templates.py, OR
        - 'raw' (intentionally not auto-detected).

        This catches formats added to prompt_formats but forgotten in _MODE_HINTS.
        """
        from ezexl3.chat.templates import _MODE_HINTS

        reachable_modes = {mode for _hint, mode in _MODE_HINTS}
        # Also include the fallback used when no hint matches
        reachable_modes.add("chatml")

        # Formats that are intentionally not auto-detected:
        #   - "raw": model-agnostic chatlog simulator
        #   - "mistral": legacy v1/v2 [INST] format, superseded by "mistral3";
        #     the "mistral" hint string routes to mistral3 for all modern models.
        #   - "gemma4-nothink": manual alternative to "gemma4" (same models,
        #     thinking suppressed); auto-detect must keep picking "gemma4".
        exempt = {"raw", "mistral", "gemma4-nothink"}

        missing = set(prompt_formats.keys()) - reachable_modes - exempt
        self.assertEqual(
            missing, set(),
            f"Format keys registered in prompt_formats but unreachable from "
            f"the auto-detect heuristic (add to _MODE_HINTS in templates.py): {missing}",
        )


# ---------------------------------------------------------------------------
# Online: upstream drift detection (network required)
# ---------------------------------------------------------------------------

_UPSTREAM_URL = (
    "https://raw.githubusercontent.com/turboderp-org/exllamav3/"
    "master/examples/chat_templates.py"
)

# Regex to extract keys from the upstream `prompt_formats = { ... }` dict.
# Matches lines like:  "gemma4": PromptFormat_gemma4,
_FORMAT_KEY_RE = re.compile(r'["\'](\w+)["\']\s*:\s*PromptFormat_')


def _fetch_upstream_format_keys():
    """Fetch upstream chat_templates.py and extract prompt_formats keys."""
    import urllib.request
    resp = urllib.request.urlopen(_UPSTREAM_URL, timeout=15)
    source = resp.read().decode("utf-8")
    return set(_FORMAT_KEY_RE.findall(source))


class TestUpstreamTemplateDrift:
    """Fetch upstream chat_templates.py and flag new formats we're missing.

    These tests xfail (not hard-fail) when drift is detected — they serve
    as a notification that we need to vendor new templates.
    """

    def test_no_missing_upstream_formats(self):
        """All format keys in upstream chat_templates.py should exist locally."""
        try:
            upstream_keys = _fetch_upstream_format_keys()
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        local_keys = set(prompt_formats.keys())
        missing = upstream_keys - local_keys

        if missing:
            pytest.xfail(
                f"UPSTREAM DRIFT: {len(missing)} new format(s) in exllamav3 "
                f"chat_templates.py not yet vendored: {sorted(missing)}. "
                f"Update ezexl3/chat/templates.py to add them."
            )

    def test_no_removed_upstream_formats(self):
        """Detect if upstream has removed formats we still carry."""
        try:
            upstream_keys = _fetch_upstream_format_keys()
        except Exception as e:
            pytest.skip(f"Network unavailable: {e}")

        local_keys = set(prompt_formats.keys())
        # 'raw' is our addition, not upstream
        extra = (local_keys - upstream_keys) - {"raw"}

        if extra:
            pytest.xfail(
                f"LOCAL EXTRAS: {len(extra)} format(s) exist locally but not "
                f"upstream: {sorted(extra)}. May be intentional or may need cleanup."
            )
