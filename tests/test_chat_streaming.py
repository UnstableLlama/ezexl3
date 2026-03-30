"""
Headless tests for the chat UI server — streaming, context override,
session save/load with tree data.

Mocks torch and exllamav3 so no GPU or model download is required.
Starts the real aiohttp server and exercises every endpoint via HTTP,
verifying the SSE streaming format, context handling, and tree
round-tripping through session save/load.
"""

import json
import sys
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies BEFORE any project imports
# ---------------------------------------------------------------------------

FAKE_TOKENS = ["Hello", ",", " world", "!"]


class FakeJob:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGenerator:
    """Simulates exllamav3.Generator — yields tokens then EOS."""

    def __init__(self, **kwargs):
        self._jobs = []
        self._cancelled = set()

    def enqueue(self, job):
        self._jobs.append(job)

    def num_remaining_jobs(self):
        return len(self._jobs)

    def iterate(self):
        if not self._jobs:
            return
        job = self._jobs[0]
        if id(job) in self._cancelled:
            self._jobs.pop(0)
            return
        for i, tok in enumerate(FAKE_TOKENS):
            is_last = i == len(FAKE_TOKENS) - 1
            result = {"text": tok, "eos": is_last}
            if is_last:
                result.update({
                    "eos_reason": "stop_token",
                    "new_tokens": len(FAKE_TOKENS),
                    "prompt_tokens": 10,
                    "cached_tokens": 5,
                    "time_prefill": 0.01,
                })
            yield result
        self._jobs.pop(0)

    def cancel(self, job):
        self._cancelled.add(id(job))


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_bos=False, encode_special_tokens=False):
        mock_ids = MagicMock()
        mock_ids.shape = (1, max(1, len(text) // 4))
        return mock_ids

    def single_id(self, text):
        return None  # No special stop tokens in test


# Install mocks before any project code is imported
_mock_torch = MagicMock()
_mock_torch.set_grad_enabled = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.cuda.device_count.return_value = 0
_mock_torch.cuda.empty_cache = MagicMock()
sys.modules["torch"] = _mock_torch

_mock_exl = MagicMock()
_mock_exl.Job = FakeJob
_mock_exl.Generator = FakeGenerator
_mock_exl.model_init = MagicMock()
sys.modules["exllamav3"] = _mock_exl

# NOW safe to import project code
from ezexl3.chat.inference import ChatEngine, ChatSettings  # noqa: E402
from ezexl3.chat.server import create_app  # noqa: E402

from aiohttp.test_utils import AioHTTPTestCase  # noqa: E402


# ---------------------------------------------------------------------------
# Build a ChatEngine with mocked internals
# ---------------------------------------------------------------------------

class FakeConfig:
    eos_token_id_list = []


class FakeCache:
    max_num_tokens = 4096


def make_test_engine():
    """Create a ChatEngine that uses fake model objects."""
    engine = ChatEngine.__new__(ChatEngine)
    engine.model_dir = "/fake/model"
    engine._devices = [0]
    engine._device_ratios = None
    engine._cache_size = 4096
    engine._cache_quant = "6,6"
    engine.model = MagicMock()
    engine.config = FakeConfig()
    engine.cache = FakeCache()
    engine.tokenizer = FakeTokenizer()
    engine.generator = FakeGenerator()
    engine.context_length = 4096
    engine.model_name = "FakeModel-0.5B"
    engine.settings = ChatSettings()
    engine.settings.mode = "chatml"
    engine.context = []
    engine._current_job = None
    engine._is_generating = False
    return engine


# ---------------------------------------------------------------------------
# Helper: parse SSE stream from response bytes
# ---------------------------------------------------------------------------

def parse_sse_events(body_bytes):
    """Parse SSE body into list of dicts (or '[DONE]' strings)."""
    events = []
    for line in body_bytes.decode().split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestChatServer(AioHTTPTestCase):
    """Integration tests against the real aiohttp app with mocked model."""

    async def get_application(self):
        self.engine = make_test_engine()
        return create_app(self.engine)

    def _fresh_generator(self):
        """Install a fresh FakeGenerator (each one drains after use)."""
        self.engine.generator = FakeGenerator()

    # -- Status & Settings --------------------------------------------------

    async def test_status_endpoint(self):
        resp = await self.client.request("GET", "/api/status")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["loaded"])
        self.assertEqual(data["model_name"], "FakeModel-0.5B")
        self.assertEqual(data["context_length"], 4096)

    async def test_get_settings(self):
        resp = await self.client.request("GET", "/api/settings")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("temperature", data)
        self.assertIn("mode", data)

    async def test_set_settings(self):
        resp = await self.client.request("POST", "/api/settings",
                                         json={"temperature": 0.5})
        self.assertEqual(resp.status, 200)
        self.assertAlmostEqual(self.engine.settings.temperature, 0.5)

    # -- Streaming chat (the sacred pipeline) --------------------------------

    async def test_chat_streams_tokens(self):
        """Verify the SSE stream yields token events followed by tps, done, [DONE]."""
        self._fresh_generator()
        resp = await self.client.request(
            "POST", "/api/chat",
            json={"message": "Hi"},
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers["Content-Type"], "text/event-stream")

        body = await resp.read()
        events = parse_sse_events(body)

        # Should have token events for each fake token
        token_events = [e for e in events if isinstance(e, dict) and e["type"] == "token"]
        self.assertEqual(len(token_events), len(FAKE_TOKENS))
        full_text = "".join(e["text"] for e in token_events)
        self.assertEqual(full_text, "Hello, world!")

        # Should have tps and done events
        types = [e["type"] for e in events if isinstance(e, dict)]
        self.assertIn("tps", types)
        self.assertIn("done", types)

        # Should end with [DONE]
        self.assertEqual(events[-1], "[DONE]")

        # Context should be updated
        self.assertEqual(len(self.engine.context), 1)
        self.assertEqual(self.engine.context[0][0], "Hi")
        self.assertEqual(self.engine.context[0][1], "Hello, world!")

    async def test_chat_empty_message_rejected(self):
        resp = await self.client.request(
            "POST", "/api/chat",
            json={"message": "   "},
        )
        self.assertEqual(resp.status, 400)

    # -- Context override (the bridge for tree) ------------------------------

    async def test_context_override(self):
        """Verify sending context in the request overrides engine.context."""
        self._fresh_generator()
        prior_context = [["Hello", "Hi there!"], ["How are you?", "I'm good."]]

        resp = await self.client.request(
            "POST", "/api/chat",
            json={"message": "Great!", "context": prior_context},
        )
        await resp.read()

        # After generation, context should have the overridden history + new turn
        self.assertEqual(len(self.engine.context), 3)
        self.assertEqual(self.engine.context[0], ("Hello", "Hi there!"))
        self.assertEqual(self.engine.context[1], ("How are you?", "I'm good."))
        self.assertEqual(self.engine.context[2][0], "Great!")
        self.assertEqual(self.engine.context[2][1], "Hello, world!")

    async def test_context_override_empty(self):
        """Context override with empty list simulates fresh start (regen from root)."""
        self._fresh_generator()
        self.engine.context = [("old", "data")]

        resp = await self.client.request(
            "POST", "/api/chat",
            json={"message": "Fresh start", "context": []},
        )
        await resp.read()

        self.assertEqual(len(self.engine.context), 1)
        self.assertEqual(self.engine.context[0][0], "Fresh start")

    async def test_regeneration_context_flow(self):
        """Simulate regeneration: same user message, context up to branch point."""
        self._fresh_generator()
        prior = [["Hello", "Hi!"]]

        resp = await self.client.request(
            "POST", "/api/chat",
            json={"message": "More", "context": prior},
        )
        body = await resp.read()
        events = parse_sse_events(body)

        full_text = "".join(e["text"] for e in events
                           if isinstance(e, dict) and e["type"] == "token")
        self.assertEqual(full_text, "Hello, world!")
        # Context: [("Hello", "Hi!"), ("More", "Hello, world!")]
        self.assertEqual(len(self.engine.context), 2)
        self.assertEqual(self.engine.context[1][0], "More")

    # -- Stop generation -----------------------------------------------------

    async def test_stop_endpoint(self):
        resp = await self.client.request("POST", "/api/stop")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["ok"])

    # -- Clear context -------------------------------------------------------

    async def test_clear_context(self):
        self.engine.context = [("a", "b"), ("c", "d")]
        resp = await self.client.request("POST", "/api/clear")
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.engine.context), 0)

    # -- Session save/load ---------------------------------------------------

    async def test_session_save(self):
        self.engine.context = [("Hello", "World")]
        self.engine.settings.temperature = 0.42

        resp = await self.client.request("GET", "/api/session/save")
        self.assertEqual(resp.status, 200)
        data = await resp.json()

        self.assertIn("context", data)
        self.assertIn("settings", data)
        self.assertEqual(data["context"], [["Hello", "World"]])
        self.assertAlmostEqual(data["settings"]["temperature"], 0.42)

    async def test_session_load(self):
        session_data = {
            "context": [["Hi", "Hey"], ["Bye", "See ya"]],
            "settings": {"temperature": 0.99, "mode": "llama3"},
        }
        resp = await self.client.request(
            "POST", "/api/session/load", json=session_data,
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.engine.context), 2)
        self.assertEqual(self.engine.context[0], ("Hi", "Hey"))
        self.assertAlmostEqual(self.engine.settings.temperature, 0.99)

    async def test_session_load_with_tree_passthrough(self):
        """Tree data in session JSON is ignored by backend (frontend handles it)."""
        session_data = {
            "context": [["A", "B"]],
            "settings": {"temperature": 0.5},
            "tree": {
                "nodes": {"n1": {"id": "n1", "role": "user", "content": "A"}},
                "rootChildren": ["n1"],
                "activeRootChild": 0,
            },
        }
        resp = await self.client.request(
            "POST", "/api/session/load", json=session_data,
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.engine.context), 1)

    # -- Multiple sequential chats -------------------------------------------

    async def test_sequential_chats_build_context(self):
        """Two sequential chats without context override should accumulate."""
        self._fresh_generator()
        await (await self.client.request(
            "POST", "/api/chat", json={"message": "First"},
        )).read()

        self._fresh_generator()
        await (await self.client.request(
            "POST", "/api/chat", json={"message": "Second"},
        )).read()

        self.assertEqual(len(self.engine.context), 2)
        self.assertEqual(self.engine.context[0][0], "First")
        self.assertEqual(self.engine.context[1][0], "Second")

    async def test_context_override_then_normal_chat(self):
        """Context override followed by normal chat should work correctly."""
        self._fresh_generator()
        await (await self.client.request(
            "POST", "/api/chat",
            json={"message": "Start", "context": [["old", "history"]]},
        )).read()

        self._fresh_generator()
        await (await self.client.request(
            "POST", "/api/chat",
            json={"message": "Continue"},
        )).read()

        self.assertEqual(len(self.engine.context), 3)
        self.assertEqual(self.engine.context[0], ("old", "history"))
        self.assertEqual(self.engine.context[1][0], "Start")
        self.assertEqual(self.engine.context[2][0], "Continue")


# ---------------------------------------------------------------------------
# Unit tests (no server needed)
# ---------------------------------------------------------------------------

class TestSSEFormat(unittest.TestCase):
    """Unit tests for SSE event format parsing (what the frontend receives)."""

    def test_parse_token_events(self):
        raw = (
            b'data: {"type": "token", "text": "Hi"}\n\n'
            b'data: {"type": "token", "text": " there"}\n\n'
            b'data: {"type": "tps", "tps": 42.0, "new_tokens": 2}\n\n'
            b'data: {"type": "done", "eos_reason": "stop_token"}\n\n'
            b"data: [DONE]\n\n"
        )
        events = parse_sse_events(raw)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["type"], "token")
        self.assertEqual(events[0]["text"], "Hi")
        self.assertEqual(events[1]["text"], " there")
        self.assertEqual(events[2]["type"], "tps")
        self.assertEqual(events[3]["type"], "done")
        self.assertEqual(events[4], "[DONE]")

    def test_full_text_reconstruction(self):
        raw = b""
        for tok in FAKE_TOKENS:
            raw += f'data: {{"type": "token", "text": "{tok}"}}\n\n'.encode()
        raw += b"data: [DONE]\n\n"

        events = parse_sse_events(raw)
        full = "".join(e["text"] for e in events
                       if isinstance(e, dict) and e.get("type") == "token")
        self.assertEqual(full, "Hello, world!")


class TestTreeContextBridge(unittest.TestCase):
    """Unit tests for the context override logic (the tree<->backend bridge)."""

    def test_tuple_conversion(self):
        raw_context = [["Hello", "Hi"], ["What?", "Nothing"]]
        converted = [tuple(pair) for pair in raw_context]
        self.assertEqual(converted[0], ("Hello", "Hi"))
        self.assertIsInstance(converted[0], tuple)

    def test_empty_context_override(self):
        converted = [tuple(pair) for pair in []]
        self.assertEqual(converted, [])

    def test_single_turn_context(self):
        converted = [tuple(pair) for pair in [["Q", "A"]]]
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0], ("Q", "A"))


# ---------------------------------------------------------------------------
# Model management tests (no-model startup, browse, load, unload)
# ---------------------------------------------------------------------------

class TestNoModelStartup(AioHTTPTestCase):
    """Tests for starting the server without a model loaded."""

    async def get_application(self):
        self.engine = ChatEngine.__new__(ChatEngine)
        self.engine.model_dir = None
        self.engine._devices = []
        self.engine._device_ratios = None
        self.engine._cache_size = 32768
        self.engine._cache_quant = "6,6"
        self.engine.model = None
        self.engine.config = None
        self.engine.cache = None
        self.engine.tokenizer = None
        self.engine.generator = None
        self.engine.context_length = 0
        self.engine.model_name = ""
        self.engine.settings = ChatSettings()
        self.engine.context = []
        self.engine._current_job = None
        self.engine._is_generating = False
        return create_app(self.engine)

    async def test_status_shows_not_loaded(self):
        resp = await self.client.request("GET", "/api/status")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertFalse(data["loaded"])
        self.assertEqual(data["model_name"], "")
        self.assertEqual(data["context_length"], 0)
        self.assertIn("gpus", data)

    async def test_chat_returns_error_when_unloaded(self):
        resp = await self.client.request(
            "POST", "/api/chat",
            json={"message": "Hello"},
        )
        self.assertEqual(resp.status, 200)
        body = await resp.read()
        events = parse_sse_events(body)
        error_events = [e for e in events if isinstance(e, dict) and e.get("type") == "error"]
        self.assertTrue(len(error_events) > 0)
        self.assertIn("not loaded", error_events[0]["message"].lower())

    async def test_settings_work_without_model(self):
        resp = await self.client.request("GET", "/api/settings")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("temperature", data)

    async def test_gpus_endpoint(self):
        resp = await self.client.request("GET", "/api/gpus")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("gpus", data)
        self.assertIsInstance(data["gpus"], list)

    async def test_model_unload_when_not_loaded(self):
        resp = await self.client.request("POST", "/api/model/unload")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["ok"])

    async def test_model_load_missing_dir(self):
        resp = await self.client.request(
            "POST", "/api/model/load",
            json={"model_dir": "/nonexistent/path/to/model"},
        )
        self.assertEqual(resp.status, 400)
        data = await resp.json()
        self.assertFalse(data["ok"])

    async def test_model_load_empty_dir(self):
        resp = await self.client.request(
            "POST", "/api/model/load",
            json={"model_dir": ""},
        )
        self.assertEqual(resp.status, 400)
        data = await resp.json()
        self.assertFalse(data["ok"])


class TestBrowseEndpoint(AioHTTPTestCase):
    """Tests for the file browser endpoint."""

    async def get_application(self):
        self.engine = ChatEngine.__new__(ChatEngine)
        self.engine.model_dir = None
        self.engine._devices = []
        self.engine._device_ratios = None
        self.engine._cache_size = 32768
        self.engine._cache_quant = "6,6"
        self.engine.model = None
        self.engine.config = None
        self.engine.cache = None
        self.engine.tokenizer = None
        self.engine.generator = None
        self.engine.context_length = 0
        self.engine.model_name = ""
        self.engine.settings = ChatSettings()
        self.engine.context = []
        self.engine._current_job = None
        self.engine._is_generating = False
        return create_app(self.engine)

    async def test_browse_default_path(self):
        """Browse with no path should default to home directory."""
        resp = await self.client.request("GET", "/api/browse")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("current", data)
        self.assertIn("entries", data)
        self.assertIn("is_model", data)
        self.assertIsInstance(data["entries"], list)

    async def test_browse_root(self):
        resp = await self.client.request("GET", "/api/browse?path=/")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["current"], "/")
        self.assertIsNone(data["parent"])

    async def test_browse_nonexistent_path(self):
        resp = await self.client.request("GET", "/api/browse?path=/nonexistent/foobar")
        self.assertEqual(resp.status, 400)

    async def test_browse_entries_have_correct_format(self):
        resp = await self.client.request("GET", "/api/browse?path=/")
        data = await resp.json()
        for entry in data["entries"]:
            self.assertIn("name", entry)
            self.assertIn("type", entry)
            self.assertIn(entry["type"], ("dir", "file"))


class TestEngineUnload(unittest.TestCase):
    """Unit tests for engine unload/load_model methods."""

    def test_unload_resets_state(self):
        engine = make_test_engine()
        self.assertTrue(engine.is_loaded)
        engine.unload()
        self.assertFalse(engine.is_loaded)
        self.assertIsNone(engine.generator)
        self.assertIsNone(engine.model)
        self.assertEqual(engine.context_length, 0)
        self.assertEqual(engine.context, [])
        self.assertEqual(engine.model_name, "")
        self.assertIsNone(engine.model_dir)

    def test_unload_when_not_loaded(self):
        """Unload on an already-unloaded engine should not raise."""
        engine = ChatEngine.__new__(ChatEngine)
        engine.model_dir = None
        engine._devices = []
        engine._device_ratios = None
        engine._cache_size = 32768
        engine._cache_quant = "6,6"
        engine.model = None
        engine.config = None
        engine.cache = None
        engine.tokenizer = None
        engine.generator = None
        engine.context_length = 0
        engine.model_name = ""
        engine.settings = ChatSettings()
        engine.context = []
        engine._current_job = None
        engine._is_generating = False
        engine.unload()  # Should not raise

    def test_detect_gpus_returns_list(self):
        # With mocked torch, cuda may not be "available"
        result = ChatEngine.detect_gpus()
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
