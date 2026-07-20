"""
Tests for the chat prompt queue (batch DPO capture from a JSONL file).

A JSONL of prompts can be opened as a live queue: the UI serves one
prompt at a time as a fresh-conversation DPO duel, and every advance is
checkpointed per file (next unserved 1-based file line) in the ratings
dir, so a re-opened queue resumes where it left off. An explicit start
line overrides the checkpoint.

Mocks torch and exllamav3 so no GPU or model download is required.
"""

import json
import sys
import tempfile
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

from ezexl3.chat.prompt_queue import (  # noqa: E402
    PromptQueue, load_checkpoint, parse_prompt_line, save_checkpoint,
)
from ezexl3.chat.inference import ChatEngine  # noqa: E402
from ezexl3.chat.server import create_app  # noqa: E402

from aiohttp.test_utils import AioHTTPTestCase  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_prompts(lines):
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8")
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

class TestParsePromptLine(unittest.TestCase):

    def test_accepted_shapes(self):
        cases = [
            ('"just a JSON string"', "just a JSON string"),
            ('{"prompt": "from prompt key"}', "from prompt key"),
            ('{"text": "from text key"}', "from text key"),
            ('{"instruction": "do the thing"}', "do the thing"),
            ('{"question": "why?"}', "why?"),
            ('{"message": "hi"}', "hi"),
            # Turn lists use the LAST user turn
            ('{"prompt": [{"role": "system", "content": "s"}, '
             '{"role": "user", "content": "first"}, '
             '{"role": "assistant", "content": "a"}, '
             '{"role": "user", "content": "last"}]}', "last"),
            ('[{"role": "user", "content": "bare turn list"}]',
             "bare turn list"),
            # Non-JSON lines are used verbatim (plain-text prompt lists)
            ("What's 2+2?", "What's 2+2?"),
        ]
        for raw, want in cases:
            self.assertEqual(parse_prompt_line(raw), want, raw)

    def test_blank_lines_are_none(self):
        self.assertIsNone(parse_prompt_line(""))
        self.assertIsNone(parse_prompt_line("   "))

    def test_unusable_json_raises(self):
        for raw in ('""', '42', '{"prompt": 5}', '{"other": "x"}',
                    '[{"role": "assistant", "content": "no user"}]',
                    '{"prompt": []}'):
            with self.assertRaises(ValueError, msg=raw):
                parse_prompt_line(raw)


# ---------------------------------------------------------------------------
# Queue semantics
# ---------------------------------------------------------------------------

class TestPromptQueue(unittest.TestCase):

    def test_entries_keep_file_line_numbers(self):
        path = write_prompts(['"one"', "", '"three"'])
        q = PromptQueue(path)
        self.assertEqual([e["line"] for e in q.entries], [1, 3])
        self.assertEqual(q.status()["total"], 2)
        self.assertEqual(q.status()["prompt"], "one")
        self.assertEqual(q.status()["line"], 1)

    def test_seek_line_lands_on_next_usable_entry(self):
        path = write_prompts(['"one"', "", '"three"', '"four"'])
        q = PromptQueue(path)
        q.seek_line(2)  # line 2 is blank -> first entry at/after it: line 3
        self.assertEqual(q.status()["line"], 3)
        q.seek_line(99)
        self.assertTrue(q.status()["done"])

    def test_advance_is_guarded_by_index(self):
        path = write_prompts(['"a"', '"b"'])
        q = PromptQueue(path)
        self.assertFalse(q.advance(1))   # not the current entry
        self.assertTrue(q.advance(0))
        self.assertFalse(q.advance(0))   # duplicate — no double skip
        self.assertEqual(q.status()["index"], 1)
        self.assertTrue(q.advance(1))
        self.assertTrue(q.status()["done"])
        self.assertIsNone(q.status()["prompt"])
        self.assertFalse(q.advance(2))   # past the end

    def test_next_line_checkpoint_value(self):
        path = write_prompts(['"a"', "", '"b"'])
        q = PromptQueue(path)
        self.assertEqual(q.next_line(), 1)
        q.advance(0)
        self.assertEqual(q.next_line(), 3)
        q.advance(1)
        self.assertEqual(q.next_line(), 4)  # EOF + 1 -> reopens as done

    def test_bad_line_reports_line_number(self):
        path = write_prompts(['"ok"', '{"prompt": 5}'])
        with self.assertRaises(ValueError) as ctx:
            PromptQueue(path)
        self.assertIn("line 2", str(ctx.exception))

    def test_empty_file_raises(self):
        path = write_prompts(["", "   "])
        with self.assertRaises(ValueError):
            PromptQueue(path)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpoints(unittest.TestCase):

    def test_roundtrip_and_isolation_per_path(self):
        root = tempfile.mkdtemp()
        self.assertIsNone(load_checkpoint(root, "/some/file.jsonl"))
        save_checkpoint(root, "/some/file.jsonl", 7)
        save_checkpoint(root, "/other/file.jsonl", 3)
        self.assertEqual(load_checkpoint(root, "/some/file.jsonl"), 7)
        self.assertEqual(load_checkpoint(root, "/other/file.jsonl"), 3)
        save_checkpoint(root, "/some/file.jsonl", 8)  # overwrite
        self.assertEqual(load_checkpoint(root, "/some/file.jsonl"), 8)

    def test_corrupt_checkpoint_file_is_tolerated(self):
        root = Path(tempfile.mkdtemp())
        (root / "queue_checkpoints.json").write_text("not json")
        self.assertIsNone(load_checkpoint(root, "/f.jsonl"))
        save_checkpoint(root, "/f.jsonl", 2)  # recovers by rewriting
        self.assertEqual(load_checkpoint(root, "/f.jsonl"), 2)


# ---------------------------------------------------------------------------
# Route integration
# ---------------------------------------------------------------------------

def make_test_engine():
    engine = ChatEngine.__new__(ChatEngine)
    engine.model_dir = "/fake/FakeModel-0.5B"
    engine.model_name = "FakeModel-0.5B"
    engine.model = MagicMock()
    engine.generator = None
    engine.settings = MagicMock()
    return engine


class TestQueueRoutes(AioHTTPTestCase):

    async def get_application(self):
        self.engine = make_test_engine()
        self.tmpdir = tempfile.mkdtemp()
        self._cfg_patch = patch(
            "ezexl3.chat.server._load_config",
            lambda: {"ratings_dir": self.tmpdir},
        )
        self._cfg_patch.start()
        self.addCleanup(self._cfg_patch.stop)
        self.prompts_path = write_prompts(['"p1"', '"p2"', '"p3"'])
        return create_app(self.engine)

    async def test_no_queue_by_default(self):
        resp = await self.client.request("GET", "/api/queue")
        self.assertEqual(resp.status, 200)
        self.assertEqual(await resp.json(), {"active": False})

    async def test_open_advance_close_flow(self):
        resp = await self.client.request(
            "POST", "/api/queue/open", json={"path": self.prompts_path})
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertTrue(data["active"])
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["prompt"], "p1")
        self.assertEqual(data["index"], 0)

        resp = await self.client.request(
            "POST", "/api/queue/advance", json={"index": 0})
        data = await resp.json()
        self.assertEqual(data["prompt"], "p2")
        # Duplicate advance is a no-op (idempotent)
        resp = await self.client.request(
            "POST", "/api/queue/advance", json={"index": 0})
        data = await resp.json()
        self.assertEqual(data["prompt"], "p2")

        resp = await self.client.request("POST", "/api/queue/close")
        self.assertEqual(await resp.json(), {"active": False})
        resp = await self.client.request("GET", "/api/queue")
        self.assertEqual(await resp.json(), {"active": False})

    async def test_advance_checkpoints_and_reopen_resumes(self):
        await self.client.request(
            "POST", "/api/queue/open", json={"path": self.prompts_path})
        await self.client.request(
            "POST", "/api/queue/advance", json={"index": 0})
        # Checkpoint written under the ratings dir
        ckpt = json.loads(
            (Path(self.tmpdir) / "queue_checkpoints.json").read_text())
        key = str(Path(self.prompts_path).resolve())
        self.assertEqual(ckpt[key]["line"], 2)
        # Close and reopen without start_line -> resumes at the checkpoint
        await self.client.request("POST", "/api/queue/close")
        resp = await self.client.request(
            "POST", "/api/queue/open", json={"path": self.prompts_path})
        data = await resp.json()
        self.assertEqual(data["prompt"], "p2")

    async def test_start_line_overrides_checkpoint(self):
        await self.client.request(
            "POST", "/api/queue/open", json={"path": self.prompts_path})
        await self.client.request(
            "POST", "/api/queue/advance", json={"index": 0})
        resp = await self.client.request(
            "POST", "/api/queue/open",
            json={"path": self.prompts_path, "start_line": 3})
        data = await resp.json()
        self.assertEqual(data["prompt"], "p3")
        self.assertEqual(data["line"], 3)

    async def test_open_rejects_bad_input(self):
        cases = [
            {},                                            # no path
            {"path": "/nonexistent/nope.jsonl"},           # missing file
            {"path": self.prompts_path, "start_line": 0},  # bad line
            {"path": self.prompts_path, "start_line": "x"},
            {"path": self.prompts_path, "start_line": True},
        ]
        for body in cases:
            resp = await self.client.request(
                "POST", "/api/queue/open", json=body)
            self.assertEqual(resp.status, 400, body)

    async def test_open_reports_unparseable_line(self):
        bad = write_prompts(['"ok"', '{"prompt": 5}'])
        resp = await self.client.request(
            "POST", "/api/queue/open", json={"path": bad})
        self.assertEqual(resp.status, 400)
        self.assertIn("line 2", (await resp.json())["error"])

    async def test_advance_requires_open_queue(self):
        resp = await self.client.request(
            "POST", "/api/queue/advance", json={"index": 0})
        self.assertEqual(resp.status, 400)


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------

class TestUiWiring(unittest.TestCase):

    def test_chat_ui_has_queue_controls(self):
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        for needle in ('id="queue-path"', 'id="queue-start-line"',
                       'id="queue-open-btn"', 'id="queue-close-btn"',
                       'id="queue-next-btn"', 'id="queue-skip-btn"',
                       'id="queue-status"', 'js/queue.js'):
            self.assertIn(needle, html)

    def test_queue_js_defines_api(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/queue.js").read_text()
        for name in ("openQueue", "closeQueue", "queueRunCurrent",
                     "queueAdvance", "queueDuelResolved", "queueSkipPrompt"):
            self.assertIn(name, js)
        # Each queue prompt starts a fresh root conversation (no context)
        self.assertIn("runDuel(userNode, [])", js)

    def test_ratings_js_notifies_queue_on_resolve(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        # Both commit and skip hand off to the queue auto-advance hook,
        # passing the resolved duel's user node so manual side-chat
        # duels can't advance the queue.
        self.assertEqual(js.count("queueDuelResolved(userNodeId)"), 2)

    def test_duel_candidate_count_is_parameterized(self):
        # DPO batch size: the sidebar exposes candidates-per-duel (2-4)
        # and the duel path generates n candidates, still judged
        # best-vs-worst (one ▲ + one ▼ pair).
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        self.assertIn('id="ratings-duel-n"', html)
        ratings_js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        self.assertIn("duelCandidateCount", ratings_js)
        self.assertIn("duelLabel", ratings_js)
        chat_js = (REPO_ROOT / "ezexl3/chat/static/js/chat.js").read_text()
        self.assertIn("duelCandidateCount()", chat_js)
        self.assertIn("duelSystemPrompts(n)", chat_js)
        render_js = (REPO_ROOT / "ezexl3/chat/static/js/render.js").read_text()
        self.assertIn("duelLabel(i)", render_js)


if __name__ == "__main__":
    unittest.main()
