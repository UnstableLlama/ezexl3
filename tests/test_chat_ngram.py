"""
Tests for n-gram draft support in the chat engine.

N-gram drafting is exllamav3's draft-model-free speculative decoding
(Generator's ngram_match_min kwarg, backed by a suffix automaton): drafts
are produced by matching recent output against prior context, so there is
no extra model to load and no VRAM cost. It is mutually exclusive with a
draft model directory (DFlash etc.) and MTP drafting.

Mocks torch and exllamav3 so no GPU or model download is required.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock heavy dependencies BEFORE any project imports
# ---------------------------------------------------------------------------

if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.cuda.is_available.return_value = False
    _mock_torch.cuda.device_count.return_value = 0
    sys.modules["torch"] = _mock_torch

if "exllamav3" not in sys.modules:
    sys.modules["exllamav3"] = MagicMock()

from ezexl3.chat import inference  # noqa: E402
from ezexl3.chat.inference import ChatEngine  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_engine(model_dir=None):
    """Bare ChatEngine with just the attributes draft loading needs."""
    engine = ChatEngine.__new__(ChatEngine)
    engine.model_dir = model_dir
    engine.model_name = os.path.basename(model_dir) if model_dir else ""
    engine.model = MagicMock()
    engine.config = MagicMock()
    engine.cache = MagicMock()
    engine.cache.max_num_tokens = 4096
    engine.tokenizer = MagicMock()
    engine.draft_model = None
    engine.draft_cache = None
    engine.draft_config = None
    engine.draft_model_dir = None
    engine.draft_model_name = ""
    engine.use_mtp = False
    engine.ngram_min = 0
    return engine


class TestCreateGeneratorNgram(unittest.TestCase):

    def _generator_kwargs(self, engine):
        MockGenerator = MagicMock()
        with patch.object(inference, "_import_generator", return_value=MockGenerator):
            engine._create_generator()
        return MockGenerator.call_args.kwargs

    def test_ngram_min_passed_when_set(self):
        engine = make_engine("/fake/model")
        engine.ngram_min = 3
        kwargs = self._generator_kwargs(engine)
        self.assertEqual(kwargs["ngram_match_min"], 3)
        self.assertNotIn("draft_model", kwargs)

    def test_ngram_min_omitted_when_zero(self):
        engine = make_engine("/fake/model")
        kwargs = self._generator_kwargs(engine)
        self.assertNotIn("ngram_match_min", kwargs)

    def test_draft_model_takes_precedence(self):
        # Generator asserts on draft_model + ngram; the engine must never
        # pass both even if state is inconsistent.
        engine = make_engine("/fake/model")
        engine.draft_model = MagicMock()
        engine.draft_cache = MagicMock()
        engine.ngram_min = 3
        kwargs = self._generator_kwargs(engine)
        self.assertIn("draft_model", kwargs)
        self.assertNotIn("ngram_match_min", kwargs)


class TestLoadModelValidation(unittest.TestCase):

    def test_load_model_rejects_ngram_plus_draft_dir(self):
        engine = ChatEngine.__new__(ChatEngine)
        engine.generator = None  # not loaded
        with self.assertRaises(ValueError):
            engine.load_model("/m", draft_model_dir="/d", ngram_min=3)

    def test_load_model_rejects_ngram_plus_mtp(self):
        engine = ChatEngine.__new__(ChatEngine)
        engine.generator = None
        with self.assertRaises(ValueError):
            engine.load_model("/m", use_mtp=True, ngram_min=3)


class TestLoadDraftNgram(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine("/fake/model")
        self.engine.generator = MagicMock()  # is_loaded
        self.engine._is_generating = False

    def test_load_draft_ngram_recreates_generator_without_loading_weights(self):
        MockModel = MagicMock()
        with patch.object(inference, "_import_model", return_value=MockModel), \
             patch.object(self.engine, "_create_generator") as mock_create:
            self.engine.load_draft(ngram_min=3)
        self.assertEqual(self.engine.ngram_min, 3)
        self.assertIsNone(self.engine.draft_model)
        MockModel.from_config.assert_not_called()
        mock_create.assert_called_once()

    def test_load_draft_rejects_ngram_plus_dir(self):
        with self.assertRaises(RuntimeError):
            self.engine.load_draft("/some/draft", ngram_min=3)

    def test_load_draft_rejects_ngram_plus_mtp(self):
        with self.assertRaises(RuntimeError):
            self.engine.load_draft(mtp=True, ngram_min=3)

    def test_load_draft_rejects_nothing_selected(self):
        with self.assertRaises(RuntimeError):
            self.engine.load_draft()

    def test_unload_draft_clears_ngram_and_recreates_generator(self):
        self.engine.ngram_min = 3
        with patch.object(self.engine, "_create_generator") as mock_create:
            self.engine.unload_draft()
        self.assertEqual(self.engine.ngram_min, 0)
        mock_create.assert_called_once()

    def test_unload_draft_noop_when_nothing_active(self):
        with patch.object(self.engine, "_create_generator") as mock_create:
            self.engine.unload_draft()
        mock_create.assert_not_called()

    def test_loading_dir_draft_clears_ngram(self):
        self.engine.ngram_min = 3
        with patch.object(inference, "_import_config"), \
             patch.object(inference, "_import_model"), \
             patch.object(inference, "_import_cache"), \
             patch.object(self.engine, "_create_generator"):
            self.engine.load_draft("/some/draft")
        self.assertEqual(self.engine.ngram_min, 0)


class TestRecurrentModelDraftHeadroom(unittest.TestCase):
    """Recurrent (hybrid linear-attn) models need Cache max_history >= draft
    length, fixed at cache creation. Post-load enabling must be refused when
    the cache has no headroom, and load() must request it up front."""

    def setUp(self):
        self.engine = make_engine("/fake/model")
        self.engine.generator = MagicMock()
        self.engine._is_generating = False
        self.engine.model.caps = {"recurrent_states": True}

    def test_load_draft_ngram_refused_without_cache_headroom(self):
        self.engine.cache.max_history = 0
        with self.assertRaises(RuntimeError) as ctx:
            self.engine.load_draft(ngram_min=3)
        self.assertIn("load time", str(ctx.exception))

    def test_load_draft_dir_refused_without_cache_headroom(self):
        self.engine.cache.max_history = 0
        with self.assertRaises(RuntimeError):
            self.engine.load_draft("/some/draft")

    def test_load_draft_ngram_allowed_with_cache_headroom(self):
        self.engine.cache.max_history = 4
        with patch.object(self.engine, "_create_generator"):
            self.engine.load_draft(ngram_min=3)
        self.assertEqual(self.engine.ngram_min, 3)

    def test_load_draft_ngram_allowed_on_non_recurrent_model(self):
        self.engine.model.caps = {}
        self.engine.cache.max_history = 0
        with patch.object(self.engine, "_create_generator"):
            self.engine.load_draft(ngram_min=3)
        self.assertEqual(self.engine.ngram_min, 3)

    def test_needs_load_time_draft(self):
        # Recurrent without headroom: only a load-time draft works
        self.engine.cache.max_history = 0
        self.assertTrue(self.engine.needs_load_time_draft())
        # Recurrent with headroom: hot-loading is fine
        self.engine.cache.max_history = 4
        self.assertFalse(self.engine.needs_load_time_draft())
        # Non-recurrent: always hot-loadable
        self.engine.model.caps = {}
        self.engine.cache.max_history = 0
        self.assertFalse(self.engine.needs_load_time_draft())


class TestEngineInitDraft(unittest.TestCase):
    """ChatEngine can be constructed with a draft source so the CLI can
    load model + draft together (required for recurrent models)."""

    def test_init_stores_draft_model_dir(self):
        engine = ChatEngine(model_dir="/m", draft_model_dir="/d/dflash")
        self.assertEqual(engine.draft_model_dir, os.path.abspath("/d/dflash"))
        self.assertEqual(engine.draft_model_name, "dflash")

    def test_init_stores_mtp_and_ngram(self):
        self.assertTrue(ChatEngine(model_dir="/m", use_mtp=True).use_mtp)
        self.assertEqual(ChatEngine(model_dir="/m", ngram_min=3).ngram_min, 3)

    def test_init_rejects_multiple_draft_sources(self):
        for kwargs in (
            {"draft_model_dir": "/d", "use_mtp": True},
            {"draft_model_dir": "/d", "ngram_min": 3},
            {"use_mtp": True, "ngram_min": 3},
        ):
            with self.assertRaises(ValueError):
                ChatEngine(model_dir="/m", **kwargs)


class TestLoadPassesMinDraftLen(unittest.TestCase):
    """load() must ask model_init for recurrent draft headroom whenever any
    draft source is selected, so hybrid models can speculate at all."""

    def _run_load(self, engine):
        class FakeModelInit:
            captured = {}

            @staticmethod
            def add_args(parser, cache=False):
                parser.add_argument("-m", "--model_dir")
                parser.add_argument("-gs", "--gpu_split")
                parser.add_argument("-cs", "--cache_size", type=int)
                parser.add_argument("-cq", "--cache_quant")

            @staticmethod
            def init(args, min_draft_len=0):
                FakeModelInit.captured["min_draft_len"] = min_draft_len
                model = MagicMock()
                cache = MagicMock()
                cache.max_num_tokens = 4096
                return model, MagicMock(), cache, MagicMock()

        with patch.object(inference, "_import_model_init", return_value=FakeModelInit), \
             patch.object(engine, "_create_generator"), \
             patch.object(engine, "_load_loras"), \
             patch.object(engine, "_auto_detect_mode"):
            engine.load()
        return FakeModelInit.captured

    def _make_load_engine(self):
        engine = make_engine("/fake/model")
        engine._devices = [0]
        engine._device_ratios = None
        engine._cache_size = 32768
        engine._cache_quant = "6,6"
        engine.settings = MagicMock()
        engine.settings.mode = "chatml"
        return engine

    def test_min_draft_len_passed_for_ngram(self):
        engine = self._make_load_engine()
        engine.ngram_min = 3
        captured = self._run_load(engine)
        self.assertEqual(captured["min_draft_len"], 4)

    def test_min_draft_len_passed_for_mtp(self):
        engine = self._make_load_engine()
        engine.use_mtp = True
        with patch.object(ChatEngine, "_load_draft_model"):
            captured = self._run_load(engine)
        self.assertEqual(captured["min_draft_len"], 4)

    def test_min_draft_len_omitted_without_draft_source(self):
        engine = self._make_load_engine()
        captured = self._run_load(engine)
        self.assertEqual(captured["min_draft_len"], 0)


class TestStatusExposesNgram(unittest.TestCase):

    def test_get_status_includes_ngram_min(self):
        engine = make_engine("/fake/model")
        engine.generator = None
        engine._is_generating = False
        engine.context_length = 0
        engine.context = []
        engine.lora_dirs = []
        engine.lora_weights = []
        engine.loras = []
        engine.ngram_min = 3
        status = engine.get_status()
        self.assertEqual(status["ngram_min"], 3)


class TestUiWiring(unittest.TestCase):

    def test_index_html_has_ngram_controls(self):
        src = (REPO_ROOT / "ezexl3" / "chat" / "static" / "index.html").read_text()
        self.assertIn('id="use-ngram-checkbox"', src)
        self.assertIn('id="use-ngram-min"', src)
        self.assertIn('id="draft-ngram-checkbox"', src)
        self.assertIn('id="draft-ngram-min"', src)

    def test_model_js_sends_ngram_min(self):
        src = (REPO_ROOT / "ezexl3" / "chat" / "static" / "js" / "model.js").read_text()
        self.assertIn("ngram_min", src)

    def test_draft_js_sends_ngram_min(self):
        src = (REPO_ROOT / "ezexl3" / "chat" / "static" / "js" / "draft.js").read_text()
        self.assertIn("ngram_min", src)


if __name__ == "__main__":
    unittest.main()
