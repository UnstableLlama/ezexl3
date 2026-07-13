"""
Tests for chat preference-rating capture (KTO / DPO data collection).

The chat UI captures in one of two modes: KTO (👍/👎 writes independent
labeled rows) or DPO (each send generates two candidates; picking the
better one writes a single chosen/rejected pair). Rows are JSONL shaped
exactly like what UnstableLlama/exllamav3's training/qlora_train_pref.py
reads with its default keys (--prompt-key prompt, --completion-key
completion, --label-key label, --chosen-key chosen, --rejected-key
rejected).

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

    def test_pair_row_is_trainer_format(self):
        store, root = make_store()
        store.rate_dpo_pair("t", PROMPT, {"node_id": "a", "content": "better"},
                            {"node_id": "b", "content": "worse"}, "FakeModel")
        row = json.loads((root / "t.dpo.jsonl").read_text())
        # Exact default keys qlora_train_pref.py reads
        self.assertEqual(row["prompt"], PROMPT)
        self.assertEqual(row["chosen"], "better")
        self.assertEqual(row["rejected"], "worse")
        # Provenance rides along
        self.assertEqual(row["chosen_node_id"], "a")
        self.assertEqual(row["rejected_node_id"], "b")
        self.assertEqual(row["source"], "duel")
        self.assertEqual(row["model"], "FakeModel")
        self.assertIn("ts", row)

    def test_pair_upsert_is_keyed_by_unordered_duo(self):
        store, root = make_store()
        store.rate_dpo_pair("t", PROMPT, {"node_id": "a", "content": "A"},
                            {"node_id": "b", "content": "B"}, "m")
        # Changing your mind swaps the pair in place, no contradictory rows
        store.rate_dpo_pair("t", PROMPT, {"node_id": "b", "content": "B"},
                            {"node_id": "a", "content": "A"}, "m")
        pairs = store.state("t")["dpo"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"], "b")
        rows = (root / "t.dpo.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        # A different duo is a separate row
        store.rate_dpo_pair("t", PROMPT, {"node_id": "c", "content": "C"},
                            {"node_id": "d", "content": "D"}, "m")
        self.assertEqual(len(store.state("t")["dpo"]), 2)

    def test_pair_remove(self):
        store, _ = make_store()
        store.rate_dpo_pair("t", PROMPT, {"node_id": "a", "content": "A"},
                            {"node_id": "b", "content": "B"}, "m")
        store.rate_dpo_pair("t", PROMPT, {"node_id": "a", "content": ""},
                            {"node_id": "b", "content": ""}, "m", remove=True)
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
        store.rate_dpo_pair("beta", PROMPT, {"node_id": "x", "content": "X"},
                            {"node_id": "y", "content": "Y"}, "m")
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

    async def test_rate_kto_writes_no_pairs(self):
        # KTO mode: 👍 and 👎 on sibling branches stay independent rows —
        # nothing is auto-paired.
        for node, label in (("a", True), ("b", False)):
            resp = await self.client.request("POST", "/api/rate", json={
                "dataset": "t",
                "prompt": PROMPT,
                "kto": {"node_id": node, "completion": node * 2, "label": label},
            })
            self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["kto"], {"a": True, "b": False})
        self.assertEqual(data["dpo"], [])
        # Server stamps the loaded model's directory on rows (full path —
        # the basename alone is ambiguous for .../Model-Name/4 layouts).
        row = json.loads(
            (Path(self.tmpdir) / "t.kto.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["model"], "/fake/FakeModel-0.5B")

    async def test_rate_duel_pair(self):
        body = {
            "dataset": "t",
            "prompt": PROMPT,
            "pair": {
                "chosen": {"node_id": "a", "content": "better"},
                "rejected": {"node_id": "b", "content": "worse"},
            },
        }
        resp = await self.client.request("POST", "/api/rate", json=body)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["dpo"], [
            {"chosen": "a", "rejected": "b", "source": "duel"},
        ])
        # remove withdraws the pair (content not needed)
        body["pair"] = {"chosen": {"node_id": "a"},
                        "rejected": {"node_id": "b"}, "remove": True}
        resp = await self.client.request("POST", "/api/rate", json=body)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["dpo"], [])

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
            {"dataset": "t", "prompt": PROMPT,   # pair without rejected id
             "pair": {"chosen": {"node_id": "a", "content": "x"}}},
            {"dataset": "t", "prompt": PROMPT,   # pair without content
             "pair": {"chosen": {"node_id": "a"},
                      "rejected": {"node_id": "b"}}},
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
        self.assertIn('id="rating-mode-toggle"', html)
        self.assertIn('data-mode="kto"', html)
        self.assertIn('data-mode="dpo"', html)
        self.assertIn('js/ratings.js', html)

    def test_render_wires_rating_controls(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/render.js").read_text()
        self.assertIn("rateNode", js)
        self.assertIn("renderDuelChoice", js)
        self.assertIn("resolveDuel", js)

    def test_ratings_js_defines_api(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        for name in ("rateNode", "resolveDuel", "setRatingsMode",
                     "refreshRatings", "buildPromptTurns", "removePairFor"):
            self.assertIn(name, js)

    def test_chat_js_runs_duels_in_dpo_mode(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/chat.js").read_text()
        self.assertIn("runDuel", js)
        self.assertIn("ratingsMode === 'dpo'", js)


if __name__ == "__main__":
    unittest.main()
