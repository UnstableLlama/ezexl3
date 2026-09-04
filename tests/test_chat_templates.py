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
            if cls.requires_model_dir:
                continue  # needs a real model dir; covered by TestJinjaFormat
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
# Offline: Laguna (Poolside)
# ---------------------------------------------------------------------------

class TestLagunaFormat(unittest.TestCase):
    """Pins the wire format against poolside/Laguna-S-2.1's chat_template.jinja
    (verified byte-for-byte against a render of the real template).

    The load-bearing detail: EVERY assistant turn opens with a think block —
    an open <think> when thinking, a bare </think> when not — and the no-think
    render drops any reasoning span from history.
    """

    SYS = "S."

    def _fmt(self, ctx, think):
        pf = prompt_formats["laguna"]("User", "Assistant")
        out = pf.format(self.SYS, ctx, think)
        if think:
            out += pf.thinktag()[0]  # the prefill the caller appends
        return out

    def test_generation_prompt_thinking(self):
        self.assertEqual(
            self._fmt([("hi", None)], True),
            "<system>S.</system>\n<user>hi</user>\n<assistant><think>")

    def test_generation_prompt_no_thinking(self):
        self.assertEqual(
            self._fmt([("hi", None)], False),
            "<system>S.</system>\n<user>hi</user>\n<assistant></think>")

    def test_history_turn_always_carries_a_think_block(self):
        self.assertEqual(
            self._fmt([("hi", "yo"), ("more", None)], True),
            "<system>S.</system>\n<user>hi</user>\n"
            "<assistant><think></think>yo</assistant>\n"
            "<user>more</user>\n<assistant><think>")

    def test_history_reasoning_span_is_rewrapped(self):
        self.assertEqual(
            self._fmt([("hi", "musing</think>yo"), ("more", None)], True),
            "<system>S.</system>\n<user>hi</user>\n"
            "<assistant><think>musing</think>yo</assistant>\n"
            "<user>more</user>\n<assistant><think>")

    def test_no_think_drops_history_reasoning(self):
        self.assertEqual(
            self._fmt([("hi", "musing</think>yo"), ("more", None)], False),
            "<system>S.</system>\n<user>hi</user>\n"
            "<assistant></think>yo</assistant>\n"
            "<user>more</user>\n<assistant></think>")

    def test_bos_is_left_to_the_tokenizer(self):
        """Laguna's tokenizer carries a TemplateProcessing post-processor that
        prepends 〈|EOS|〉 (id 2, which doubles as BOS) on EVERY encode. The
        chat template writes it literally too, so a format that emitted it
        would tokenize to [2, 2]. Neither the literal nor add_bos, therefore
        — verified against the real tokenizer to land exactly one id 2.
        """
        pf = prompt_formats["laguna"]("User", "Assistant")
        out = pf.format("S.", [("hi", None)], think=True)
        self.assertFalse(out.startswith("〈|EOS|〉"))
        self.assertFalse(pf.add_bos())

    def test_autodetect(self):
        from ezexl3.chat.templates import infer_mode
        self.assertEqual(infer_mode("Laguna-S-2.1-exl3"), "laguna")


# ---------------------------------------------------------------------------
# Offline: the 'jinja' format (renders the model's own chat template)
# ---------------------------------------------------------------------------

# Exercises the pieces real templates need: a {% generation %} block (an HF
# extension plain jinja2 rejects outright), separate reasoning_content, and
# an enable_thinking switch on the generation prompt.
_FIXTURE_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}<sys>{{ m.content }}</sys>{{ '\\n' }}"
    "{%- elif m.role == 'user' -%}<u>{{ m.content }}</u>{{ '\\n' }}"
    "{%- else -%}{%- generation -%}"
    "<a>{{ m.reasoning_content | default('') }}</think>{{ m.content }}</a>"
    "{{ '\\n' }}{%- endgeneration -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "<a>{% if enable_thinking %}<think>{% else %}</think>{% endif %}"
    "{%- endif -%}"
)


class TestJinjaFormat(unittest.TestCase):
    """The 'jinja' format resolves and renders a model dir's own template."""

    def _model_dir(self, tmp, **files):
        import os
        for name, body in files.items():
            with open(os.path.join(tmp, name.replace("__", ".")), "w",
                      encoding="utf-8") as f:
                f.write(body)
        return tmp

    def test_renders_chat_template_jinja(self):
        import tempfile
        from ezexl3.chat.templates import PromptFormat_jinja

        with tempfile.TemporaryDirectory() as tmp:
            self._model_dir(tmp, chat_template__jinja=_FIXTURE_TEMPLATE)
            pf = PromptFormat_jinja("User", "Assistant")
            pf.set_special({"model_dir": tmp})
            out = pf.format("Be nice.", [("hi", None)], think=True)
        self.assertEqual(out, "<sys>Be nice.</sys>\n<u>hi</u>\n<a><think>")

    def test_reasoning_span_becomes_reasoning_content(self):
        """Stored replies read '<reasoning></think><content>'; the template
        wants those as separate fields."""
        import tempfile
        from ezexl3.chat.templates import PromptFormat_jinja

        with tempfile.TemporaryDirectory() as tmp:
            self._model_dir(tmp, chat_template__jinja=_FIXTURE_TEMPLATE)
            pf = PromptFormat_jinja("User", "Assistant")
            pf.set_special({"model_dir": tmp})
            out = pf.format("", [("hi", "musing</think>hello"), ("more", None)],
                            think=False)
        self.assertEqual(
            out, "<u>hi</u>\n<a>musing</think>hello</a>\n<u>more</u>\n<a></think>")

    def test_falls_back_to_tokenizer_config(self):
        import json
        import tempfile
        from ezexl3.chat.templates import PromptFormat_jinja

        with tempfile.TemporaryDirectory() as tmp:
            self._model_dir(tmp, tokenizer_config__json=json.dumps(
                {"chat_template": _FIXTURE_TEMPLATE}))
            pf = PromptFormat_jinja("User", "Assistant")
            pf.set_special({"model_dir": tmp})
            out = pf.format("", [("hi", None)], think=False)
        self.assertEqual(out, "<u>hi</u>\n<a></think>")

    def test_missing_template_raises_readable_error(self):
        import tempfile
        from ezexl3.chat.templates import PromptFormat_jinja

        with tempfile.TemporaryDirectory() as tmp:
            pf = PromptFormat_jinja("User", "Assistant")
            pf.set_special({"model_dir": tmp})
            with self.assertRaises(ValueError) as cm:
                pf.format("", [("hi", None)], think=False)
        self.assertIn("No chat template found", str(cm.exception))

    def test_strips_bos_the_tokenizer_will_add(self):
        """A template that writes BOS + a tokenizer that prepends it on encode
        would double it; the literal one is dropped."""
        import tempfile
        import torch
        from ezexl3.chat.templates import PromptFormat_jinja

        class AutoBosTokenizer:
            bos_token = "<s>"
            bos_token_id = 1

            def encode(self, text, add_bos=False, encode_special_tokens=False):
                return torch.tensor([[self.bos_token_id, 42]])

        with tempfile.TemporaryDirectory() as tmp:
            self._model_dir(tmp, chat_template__jinja="<s>" + _FIXTURE_TEMPLATE)
            pf = PromptFormat_jinja("User", "Assistant")
            pf.set_special({"model_dir": tmp, "tokenizer": AutoBosTokenizer()})
            out = pf.format("", [("hi", None)], think=False)
        self.assertEqual(out, "<u>hi</u>\n<a></think>")

    def test_keeps_bos_when_tokenizer_does_not_add_one(self):
        import tempfile
        import torch
        from ezexl3.chat.templates import PromptFormat_jinja

        class PlainTokenizer:
            bos_token = "<s>"
            bos_token_id = 1

            def encode(self, text, add_bos=False, encode_special_tokens=False):
                return torch.tensor([[42]])

        with tempfile.TemporaryDirectory() as tmp:
            self._model_dir(tmp, chat_template__jinja="<s>" + _FIXTURE_TEMPLATE)
            pf = PromptFormat_jinja("User", "Assistant")
            pf.set_special({"model_dir": tmp, "tokenizer": PlainTokenizer()})
            out = pf.format("", [("hi", None)], think=False)
        self.assertEqual(out, "<s><u>hi</u>\n<a></think>")

    def test_no_thinktag_prefill(self):
        """add_generation_prompt already emits the prefill; the caller must
        not append its own think tag on top."""
        from ezexl3.chat.templates import PromptFormat_jinja

        self.assertEqual(
            PromptFormat_jinja("User", "Assistant").thinktag(), (None, None))


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
        #   - "muse-nothink": same deal for "muse" — the prefilled empty
        #     to=self turn is an opt-in, so auto-detect must keep picking
        #     "muse" for Muse Glimmer models.
        #   - "jinja": renders the model's own chat template; deliberately
        #     opt-in, so auto-detect must never select it by model name.
        exempt = {"raw", "mistral", "gemma4-nothink", "muse-nothink", "jinja"}

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
        # 'raw' and 'jinja' are our additions, not upstream
        extra = (local_keys - upstream_keys) - {"raw", "jinja"}

        if extra:
            pytest.xfail(
                f"LOCAL EXTRAS: {len(extra)} format(s) exist locally but not "
                f"upstream: {sorted(extra)}. May be intentional or may need cleanup."
            )
