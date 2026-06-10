"""
Tests for MTP draft support in the chat engine.

MTP drafting follows standard exllamav3 behavior (the --mtp flag): the
MTP head lives inside the main model's checkpoint and is loaded as the
model's "mtp" component sharing the main config. It is an explicit
toggle, mutually exclusive with the draft model directory (DFlash or
any regular draft model).

Also covers the args namespace shim: exllamav3 dev's model_init.init()
reads args.mtp unconditionally, which crashed every chat model load
("'Namespace' object has no attribute 'mtp'") when the namespace was
built without draft model args.

Mocks torch and exllamav3 so no GPU or model download is required.
"""

import argparse
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
    """Bare ChatEngine with just the attributes draft loading needs."""
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
    engine.use_mtp = False
    return engine


class TestLoadDraftModel(unittest.TestCase):

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
        self.engine.use_mtp = True
        self.engine._load_draft_model()

        # Shares the main model's config, never re-reads the directory
        self.assertIs(self.engine.draft_config, self.engine.config)
        self.assertEqual(self.engine.draft_model_dir, self.model_dir)
        self.MockConfig.from_directory.assert_not_called()
        self.MockModel.from_config.assert_called_once_with(
            self.engine.config, component="mtp"
        )
        self.assertIn("(MTP)", self.engine.draft_model_name)
        self.engine.draft_model.load.assert_called_once()

    def test_mtp_without_mtp_component_raises(self):
        self.engine.config.model_classes = {"text": object}
        self.engine.use_mtp = True
        with self.assertRaises(RuntimeError) as ctx:
            self.engine._load_draft_model()
        self.assertIn("MTP", str(ctx.exception))
        self.assertFalse(self.engine.use_mtp)

    def test_mtp_on_old_exllamav3_config_raises(self):
        # Older exllamav3 Config objects have no model_classes attribute
        self.engine.config = argparse.Namespace()
        self.engine.use_mtp = True
        with self.assertRaises(RuntimeError):
            self.engine._load_draft_model()

    def test_dir_draft_loads_from_its_own_directory(self):
        self.engine.draft_model_dir = self.draft_dir
        self.engine.draft_model_name = "draft-model"
        self.engine._load_draft_model()

        self.MockConfig.from_directory.assert_called_once_with(self.draft_dir)
        self.MockModel.from_config.assert_called_once_with(
            self.MockConfig.from_directory.return_value
        )
        self.assertEqual(self.engine.draft_model_name, "draft-model")

    def test_unload_resets_mtp(self):
        self.engine.config.model_classes = {"mtp": object}
        self.engine.use_mtp = True
        self.engine._load_draft_model()
        self.engine._unload_draft_model()
        self.assertFalse(self.engine.use_mtp)
        self.assertIsNone(self.engine.draft_model)
        self.assertIsNone(self.engine.draft_model_dir)


class TestLoadDraftValidation(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine("/fake/model")
        self.engine.generator = MagicMock()  # is_loaded
        self.engine._is_generating = False

    def test_load_draft_rejects_both_dir_and_mtp(self):
        with self.assertRaises(RuntimeError):
            self.engine.load_draft("/some/draft", mtp=True)

    def test_load_draft_rejects_neither(self):
        with self.assertRaises(RuntimeError):
            self.engine.load_draft()

    def test_load_model_rejects_both_dir_and_mtp(self):
        engine = ChatEngine.__new__(ChatEngine)
        engine.generator = None  # not loaded
        with self.assertRaises(ValueError):
            engine.load_model("/m", draft_model_dir="/d", use_mtp=True)


class TestStatusExposesMtp(unittest.TestCase):

    def test_get_status_includes_draft_mtp(self):
        engine = make_engine("/fake/model")
        engine.generator = None
        engine._is_generating = False
        engine.context_length = 0
        engine.context = []
        engine.lora_dirs = []
        engine.lora_weights = []
        engine.loras = []
        engine.use_mtp = True
        status = engine.get_status()
        self.assertIs(status["draft_mtp"], True)


class TestBuildModelArgsMtpShim(unittest.TestCase):

    def test_args_namespace_gets_mtp_default(self):
        """exllamav3 dev's init() reads args.mtp unconditionally; without
        this default every model load fails with
        "'Namespace' object has no attribute 'mtp'"."""

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
