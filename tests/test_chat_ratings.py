"""
Tests for chat preference-rating capture (KTO / DPO data collection).

The chat UI writes thumbs ratings and sibling preferences into JSONL
datasets shaped exactly like the rows UnstableLlama/exllamav3's
training/qlora_train_pref.py reads with its default keys
(--prompt-key prompt, --completion-key completion, --label-key label,
--chosen-key chosen, --rejected-key rejected).

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

from ezexl3.chat.ratings import (  # noqa: E402
    RatingsStore, valid_dataset_name, validate_prompt,
)
from ezexl3.chat.inference import ChatEngine  # noqa: E402
from ezexl3.chat.server import create_app  # noqa: E402

from aiohttp.test_utils import AioHTTPTestCase  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

PROMPT = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"},
    {"role": "user", "content": "Second question"},
]


def make_store():
    tmp = tempfile.mkdtemp()
    return RatingsStore(tmp), Path(tmp)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_dataset_names(self):
        for good in ("chat", "my-data", "run_2.kto", "A1"):
            self.assertTrue(valid_dataset_name(good), good)
        for bad in ("", "../etc", "a/b", ".hidden", "a" * 80, "sp ace"):
            self.assertFalse(valid_dataset_name(bad), bad)

    def test_prompt_must_be_turn_list_ending_in_user(self):
        self.assertIsNone(validate_prompt(PROMPT))
        self.assertIsNotNone(validate_prompt([]))
        self.assertIsNotNone(validate_prompt("just a string"))
        self.assertIsNotNone(validate_prompt([{"role": "user"}]))  # no content
        self.assertIsNotNone(validate_prompt(
            [{"role": "wizard", "content": "hi"}]))
        # Final turn assistant -> rejected (completion is stored separately)
        self.assertIsNotNone(validate_prompt(
            [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "a"}]))


# ---------------------------------------------------------------------------
# Store semantics
# ---------------------------------------------------------------------------

class TestRatingsStore(unittest.TestCase):

    def test_kto_upsert_and_clear(self):
        store, root = make_store()
        store.rate_kto("t", "n1", PROMPT, "answer", True, "m")
        store.rate_kto("t", "n1", PROMPT, "answer", False, "m")  # re-rate
        state = store.state("t")
        self.assertEqual(state["kto"], {"n1": False})
        # One row on disk, not two
        rows = [json.loads(l) for l in
                (root / "t.kto.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        # None removes
        store.rate_kto("t", "n1", PROMPT, "answer", None, "m")
        self.assertEqual(store.state("t")["kto"], {})

    def test_kto_row_is_trainer_format(self):
        store, root = make_store()
        store.rate_kto("t", "n1", PROMPT, "the completion", True, "FakeModel")
        row = json.loads((root / "t.kto.jsonl").read_text())
        # Exact default keys qlora_train_pref.py reads
        self.assertEqual(row["prompt"], PROMPT)
        self.assertEqual(row["completion"], "the completion")
        self.assertIs(row["label"], True)
        # Provenance rides along
        self.assertEqual(row["node_id"], "n1")
        self.assertEqual(row["model"], "FakeModel")
        self.assertIn("ts", row)

    def test_auto_pairs_cross_product_and_resync(self):
        store, root = make_store()
        group = [
            {"node_id": "a", "content": "good1", "label": True},
            {"node_id": "b", "content": "good2", "label": True},
            {"node_id": "c", "content": "bad1", "label": False},
            {"node_id": "d", "content": "meh", "label": None},
        ]
        store.sync_dpo_auto("t", PROMPT, group, "m")
        pairs = store.state("t")["dpo"]
        self.assertEqual(len(pairs), 2)  # 2 goods x 1 bad; unrated excluded
        self.assertEqual({(p["chosen"], p["rejected"]) for p in pairs},
                         {("a", "c"), ("b", "c")})
        row = json.loads((root / "t.dpo.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["prompt"], PROMPT)
        self.assertEqual(row["chosen"], "good1")
        self.assertEqual(row["rejected"], "bad1")
        self.assertEqual(row["source"], "auto")

        # Flip "a" to bad and resync: its pairs must vanish, not linger
        group[0]["label"] = False
        store.sync_dpo_auto("t", PROMPT, group, "m")
        pairs = store.state("t")["dpo"]
        self.assertEqual({(p["chosen"], p["rejected"]) for p in pairs},
                         {("b", "c"), ("b", "a")})

    def test_manual_pairs_survive_auto_resync(self):
        store, _ = make_store()
        store.rate_dpo_manual("t", PROMPT, {"node_id": "x", "content": "X"},
                              [{"node_id": "y", "content": "Y"}], "m")
        store.sync_dpo_auto("t", PROMPT, [
            {"node_id": "x", "content": "X", "label": None},
            {"node_id": "y", "content": "Y", "label": None},
        ], "m")
        pairs = store.state("t")["dpo"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["source"], "manual")
        # Toggle-off removes all manual pairs where x is chosen
        store.rate_dpo_manual("t", PROMPT, {"node_id": "x"}, [], "m",
                              remove=True)
        self.assertEqual(store.state("t")["dpo"], [])

    def test_foreign_lines_preserved(self):
        store, root = make_store()
        root.mkdir(parents=True, exist_ok=True)
        hand_edited = '{"prompt": "external row", "completion": "kept", "label": true}'
        not_json = "# a comment someone left"
        (root / "t.kto.jsonl").write_text(hand_edited + "\n" + not_json + "\n")
        store.rate_kto("t", "n1", PROMPT, "new", True, "m")
        content = (root / "t.kto.jsonl").read_text()
        self.assertIn('"external row"', content)
        self.assertIn(not_json, content)
        self.assertEqual(len(content.strip().splitlines()), 3)

    def test_rows_have_consistent_schema(self):
        # HF datasets loads JSONL via arrow: every row in a file must have
        # the same keys or loading can fail.
        store, root = make_store()
        store.rate_kto("t", "n1", PROMPT, "a", True, "m")
        store.rate_kto("t", "n2", PROMPT, "b", False, "")
        rows = [json.loads(l) for l in
                (root / "t.kto.jsonl").read_text().splitlines()]
        self.assertEqual(set(rows[0]), set(rows[1]))

    def test_invalid_dataset_name_raises(self):
        store, _ = make_store()
        with self.assertRaises(ValueError):
            store.rate_kto("../evil", "n", PROMPT, "c", True, "m")

    def test_list_datasets(self):
        store, root = make_store()
        store.rate_kto("alpha", "n", PROMPT, "c", True, "m")
        store.rate_dpo_manual("beta", PROMPT, {"node_id": "x", "content": "X"},
                              [{"node_id": "y", "content": "Y"}], "m")
        self.assertEqual(store.list_datasets(), ["alpha", "beta"])


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


class TestRatingRoutes(AioHTTPTestCase):

    async def get_application(self):
        self.engine = make_test_engine()
        self.tmpdir = tempfile.mkdtemp()
        self._cfg_patch = patch(
            "ezexl3.chat.server._load_config",
            lambda: {"ratings_dir": self.tmpdir},
        )
        self._cfg_patch.start()
        self.addCleanup(self._cfg_patch.stop)
        return create_app(self.engine)

    async def test_rate_kto_with_auto_pairs(self):
        body = {
            "dataset": "t",
            "prompt": PROMPT,
            "kto": {"node_id": "a", "completion": "good", "label": True},
            "group": [
                {"node_id": "a", "content": "good", "label": True},
                {"node_id": "b", "content": "bad", "label": False},
            ],
        }
        resp = await self.client.request("POST", "/api/rate", json=body)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["kto"], {"a": True})
        self.assertEqual(len(data["dpo"]), 1)
        self.assertEqual(data["dpo"][0]["chosen"], "a")
        # Server stamps the loaded model's name on rows
        row = json.loads(
            (Path(self.tmpdir) / "t.kto.jsonl").read_text())
        self.assertEqual(row["model"], "FakeModel-0.5B")

    async def test_rate_manual_pair(self):
        body = {
            "dataset": "t",
            "prompt": PROMPT,
            "manual": {
                "chosen": {"node_id": "a", "content": "better"},
                "rejected": [{"node_id": "b", "content": "worse"}],
            },
        }
        resp = await self.client.request("POST", "/api/rate", json=body)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["dpo"][0]["source"], "manual")

    async def test_rate_rejects_bad_input(self):
        cases = [
            {"dataset": "../evil", "prompt": PROMPT,
             "kto": {"node_id": "a", "completion": "x", "label": True}},
            {"dataset": "t", "prompt": "not a list",
             "kto": {"node_id": "a", "completion": "x", "label": True}},
            {"dataset": "t",
             "prompt": [{"role": "assistant", "content": "ends wrong"}],
             "kto": {"node_id": "a", "completion": "x", "label": True}},
            {"dataset": "t", "prompt": PROMPT},  # nothing to record
            {"dataset": "t", "prompt": PROMPT,
             "kto": {"node_id": "a", "completion": "x", "label": "yes"}},
        ]
        for body in cases:
            resp = await self.client.request("POST", "/api/rate", json=body)
            self.assertEqual(resp.status, 400, body)

    async def test_ratings_snapshot(self):
        await self.client.request("POST", "/api/rate", json={
            "dataset": "t", "prompt": PROMPT,
            "kto": {"node_id": "a", "completion": "good", "label": True},
        })
        resp = await self.client.request("GET", "/api/ratings?dataset=t")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["kto"], {"a": True})
        self.assertEqual(data["datasets"], ["t"])
        self.assertEqual(data["dir"], self.tmpdir)

    async def test_ratings_rejects_bad_dataset_name(self):
        resp = await self.client.request(
            "GET", "/api/ratings?dataset=../evil")
        self.assertEqual(resp.status, 400)


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------

class TestUiWiring(unittest.TestCase):

    def test_chat_ui_has_rating_controls(self):
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        self.assertIn('id="ratings-dataset"', html)
        self.assertIn('id="ratings-dir"', html)
        self.assertIn('js/ratings.js', html)

    def test_render_wires_rating_buttons(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/render.js").read_text()
        self.assertIn("rateNode", js)
        self.assertIn("preferNode", js)

    def test_ratings_js_defines_api(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        for name in ("rateNode", "preferNode", "refreshRatings",
                     "buildPromptTurns", "siblingGroup"):
            self.assertIn(name, js)


if __name__ == "__main__":
    unittest.main()
