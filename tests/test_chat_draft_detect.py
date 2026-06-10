"""
Tests for draft model kind auto-detection (DFlash vs MTP vs plain draft).

The chat UI has a single "draft model" field. The engine classifies the
path the user gives it:

  - same directory as the main model  -> "mtp" (built-in MTP head,
    loaded as the model's "mtp" component, exllamav3-dev style)
  - separate dir whose config.json declares DFlashDraftModel -> "dflash"
  - any other model dir -> "draft" (plain speculative decoding)

Mocks torch and exllamav3 so no GPU or model download is required.
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
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


def make_engine(model_dir=None):
    """Bare ChatEngine with just the attributes detection/loading needs."""
    engine = ChatEngine.__new__(ChatEngine)
    engine.model_dir = model_dir
    engine.model_name = os.path.basename(model_dir) if model_dir else ""
    engine.model = MagicMock()
    engine.config = MagicMock()
    engine.cache = MagicMock()
    engine.cache.max_num_tokens = 4096
    engine.draft_model = None
    engine.draft_cache = None
    engine.draft_config = None
    engine.draft_model_dir = None
    engine.draft_model_name = ""
    engine.draft_kind = ""
    return engine


def write_config(directory, arch):
    with open(os.path.join(directory, "config.json"), "w", encoding="utf8") as f:
        json.dump({"architectures": [arch]}, f)


class TestDetectDraftKind(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.tmp.name, "main-model")
        self.draft_dir = os.path.join(self.tmp.name, "draft-model")
        os.makedirs(self.model_dir)
        os.makedirs(self.draft_dir)
        self.engine = make_engine(self.model_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_dir_is_mtp(self):
        self.assertEqual(self.engine._detect_draft_kind(self.model_dir), "mtp")

    def test_same_dir_unnormalized_path_is_mtp(self):
        aliased = os.path.join(self.model_dir, ".", "..", "main-model")
        self.assertEqual(self.engine._detect_draft_kind(aliased), "mtp")

    def test_dflash_arch_is_dflash(self):
        write_config(self.draft_dir, "DFlashDraftModel")
        self.assertEqual(self.engine._detect_draft_kind(self.draft_dir), "dflash")

    def test_other_arch_is_plain_draft(self):
        write_config(self.draft_dir, "Qwen3ForCausalLM")
        self.assertEqual(self.engine._detect_draft_kind(self.draft_dir), "draft")

    def test_missing_config_is_plain_draft(self):
        # Server validates config.json exists before this runs; the
        # classifier itself just falls back to the generic path.
        self.assertEqual(self.engine._detect_draft_kind(self.draft_dir), "draft")

    def test_dflash_config_in_main_dir_still_mtp(self):
        # Same-directory check wins over architecture inspection
        write_config(self.model_dir, "DFlashDraftModel")
        self.assertEqual(self.engine._detect_draft_kind(self.model_dir), "mtp")


class TestLoadDraftModelByKind(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.tmp.name, "main-model")
        self.draft_dir = os.path.join(self.tmp.name, "draft-model")
        os.makedirs(self.model_dir)
        os.makedirs(self.draft_dir)
        self.engine = make_engine(self.model_dir)

        self.MockConfig = MagicMock()
        self.MockModel = MagicMock()
        self.MockCache = MagicMock()
        self.patches = [
            patch.object(inference, "_import_config", return_value=self.MockConfig),
            patch.object(inference, "_import_model", return_value=self.MockModel),
            patch.object(inference, "_import_cache", return_value=self.MockCache),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_mtp_loads_mtp_component_of_main_config(self):
        self.engine.config.model_classes = {"text": object, "mtp": object}
        self.engine.draft_model_dir = self.model_dir
        self.engine._load_draft_model()

        self.assertEqual(self.engine.draft_kind, "mtp")
        # Shares the main model's config, never re-reads the directory
        self.assertIs(self.engine.draft_config, self.engine.config)
        self.MockConfig.from_directory.assert_not_called()
        self.MockModel.from_config.assert_called_once_with(
            self.engine.config, component="mtp"
        )
        self.assertIn("(MTP)", self.engine.draft_model_name)
        self.engine.draft_model.load.assert_called_once()

    def test_mtp_without_mtp_component_raises(self):
        self.engine.config.model_classes = {"text": object}
        self.engine.draft_model_dir = self.model_dir
        with self.assertRaises(RuntimeError) as ctx:
            self.engine._load_draft_model()
        self.assertIn("MTP", str(ctx.exception))
        self.assertEqual(self.engine.draft_kind, "")

    def test_mtp_on_old_exllamav3_config_raises(self):
        # Older exllamav3 Config objects have no model_classes attribute
        self.engine.config = argparse.Namespace()
        self.engine.draft_model_dir = self.model_dir
        with self.assertRaises(RuntimeError):
            self.engine._load_draft_model()

    def test_dflash_loads_from_its_own_directory(self):
        write_config(self.draft_dir, "DFlashDraftModel")
        self.engine.draft_model_dir = self.draft_dir
        self.engine.draft_model_name = "draft-model"
        self.engine._load_draft_model()

        self.assertEqual(self.engine.draft_kind, "dflash")
        self.MockConfig.from_directory.assert_called_once_with(self.draft_dir)
        self.MockModel.from_config.assert_called_once_with(
            self.MockConfig.from_directory.return_value
        )
        self.assertEqual(self.engine.draft_model_name, "draft-model")

    def test_plain_draft_loads_from_its_own_directory(self):
        write_config(self.draft_dir, "LlamaForCausalLM")
        self.engine.draft_model_dir = self.draft_dir
        self.engine._load_draft_model()

        self.assertEqual(self.engine.draft_kind, "draft")
        self.MockConfig.from_directory.assert_called_once_with(self.draft_dir)

    def test_unload_resets_kind(self):
        write_config(self.draft_dir, "DFlashDraftModel")
        self.engine.draft_model_dir = self.draft_dir
        self.engine._load_draft_model()
        self.engine._unload_draft_model()
        self.assertEqual(self.engine.draft_kind, "")
        self.assertIsNone(self.engine.draft_model)


class TestStatusExposesDraftKind(unittest.TestCase):

    def test_get_status_includes_draft_kind(self):
        engine = make_engine("/fake/model")
        engine.generator = None
        engine._is_generating = False
        engine.context_length = 0
        engine.context = []
        engine.lora_dirs = []
        engine.lora_weights = []
        engine.loras = []
        engine.draft_kind = "mtp"
        status = engine.get_status()
        self.assertEqual(status["draft_kind"], "mtp")


class TestBuildModelArgsMtpShim(unittest.TestCase):

    def test_args_namespace_gets_mtp_default(self):
        """exllamav3 dev's init() reads args.mtp unconditionally."""

        class FakeModelInit:
            @staticmethod
            def add_args(parser, cache=False):
                parser.add_argument("-m", "--model_dir")
                parser.add_argument("-gs", "--gpu_split")
                parser.add_argument("-cs", "--cache_size", type=int)
                parser.add_argument("-cq", "--cache_quant")

        with patch.object(inference, "_import_model_init", return_value=FakeModelInit):
            args = inference._build_model_args("/m", [0], None, 32768, "6,6")
        self.assertIs(args.mtp, False)


if __name__ == "__main__":
    unittest.main()
