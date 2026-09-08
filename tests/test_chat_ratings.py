"""
Tests for chat preference-rating capture (KTO / DPO data collection).

The chat UI captures in one of two modes — KTO (👍/👎 writes independent
labeled rows) or DPO (each send generates two candidates; mark ▲/▼ and
commit to write a single chosen/rejected pair) — plus an Off position
(the default) that disables capture. Rows are JSONL shaped
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
    RatingsStore, strip_think_text, valid_dataset_name, validate_prompt,
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

    def test_pair_records_generation_prompts(self):
        # Per-candidate generation system prompts land in their own
        # metadata columns; the trainer selects columns by name and
        # never sees them.
        store, root = make_store()
        store.rate_dpo_pair(
            "t", PROMPT,
            {"node_id": "a", "content": "A", "gen_system": "be helpful"},
            {"node_id": "b", "content": "B", "gen_system": "be lazy"}, "m")
        row = json.loads((root / "t.dpo.jsonl").read_text())
        self.assertEqual(row["chosen_system"], "be helpful")
        self.assertEqual(row["rejected_system"], "be lazy")
        # Trainer columns are untouched by spoofing
        self.assertEqual(row["chosen"], "A")
        self.assertEqual(row["rejected"], "B")
        # Unspoofed pairs keep the columns (null) so every row in a file
        # shares one schema.
        store.rate_dpo_pair("t", PROMPT, {"node_id": "c", "content": "C"},
                            {"node_id": "d", "content": "D"}, "m")
        rows = [json.loads(l) for l in
                (root / "t.dpo.jsonl").read_text().splitlines()]
        self.assertEqual(set(rows[0]), set(rows[1]))
        self.assertIsNone(rows[1]["chosen_system"])
        self.assertIsNone(rows[1]["rejected_system"])

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
# Bulk generation rows (queue bulk mode / unattended runs)
# ---------------------------------------------------------------------------

BULK_PROMPT = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Only question"},
]


class TestBulkStore(unittest.TestCase):

    def test_one_sided_rows(self):
        store, root = make_store()
        n = store.add_bulk_rows("t", BULK_PROMPT, [
            {"content": "bad reply 1", "node_id": None},
            {"content": "bad reply 2", "node_id": None},
        ], "rejected", "FakeModel", gen_system="elicit bad")
        self.assertEqual(n, 2)
        rows = [json.loads(l) for l in
                (root / "t.dpo.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for i, row in enumerate(rows):
            self.assertEqual(row["prompt"], BULK_PROMPT)
            self.assertEqual(row["chosen"], "")          # to be joined later
            self.assertEqual(row["rejected"], f"bad reply {i + 1}")
            self.assertEqual(row["rejected_system"], "elicit bad")
            self.assertIsNone(row["chosen_system"])
            self.assertEqual(row["source"], "bulk")
            self.assertIsNone(row["source_row_id"])
            self.assertEqual(row["model"], "FakeModel")

    def test_carried_source_column_makes_full_pairs(self):
        store, root = make_store()
        store.add_bulk_rows("t", BULK_PROMPT,
                            [{"content": "generated bad", "node_id": None}],
                            "rejected", "m",
                            source_row={"chosen": "the gold reply",
                                        "id": "mc-001"})
        row = json.loads((root / "t.dpo.jsonl").read_text())
        self.assertEqual(row["chosen"], "the gold reply")
        self.assertEqual(row["rejected"], "generated bad")
        self.assertEqual(row["source_row_id"], "mc-001")

    def test_target_chosen_fills_other_side_from_source_rejected(self):
        store, root = make_store()
        store.add_bulk_rows("t", BULK_PROMPT,
                            [{"content": "generated good", "node_id": None}],
                            "chosen", "m",
                            source_row={"rejected": "known bad", "id": "r9"})
        row = json.loads((root / "t.dpo.jsonl").read_text())
        self.assertEqual(row["chosen"], "generated good")
        self.assertEqual(row["rejected"], "known bad")
        self.assertIsNone(row["rejected_system"])

    def test_blank_completions_skipped(self):
        store, root = make_store()
        n = store.add_bulk_rows("t", BULK_PROMPT, [
            {"content": "keep me", "node_id": None},
            {"content": "   ", "node_id": None},
        ], "rejected", "m")
        self.assertEqual(n, 1)

    def test_node_keyed_rows_upsert(self):
        # Review-flow saves carry node ids; re-saving the same candidate
        # (e.g. after withdrawing and re-judging) must not duplicate rows.
        store, root = make_store()
        store.add_bulk_rows("t", BULK_PROMPT,
                            [{"content": "v1", "node_id": "n1"}],
                            "rejected", "m")
        store.add_bulk_rows("t", BULK_PROMPT,
                            [{"content": "v2", "node_id": "n1"}],
                            "rejected", "m")
        rows = [json.loads(l) for l in
                (root / "t.dpo.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rejected"], "v2")

    def test_invalid_target_raises(self):
        store, _ = make_store()
        with self.assertRaises(ValueError):
            store.add_bulk_rows("t", BULK_PROMPT,
                                [{"content": "x", "node_id": None}],
                                "best", "m")

    def test_bulk_rows_count_in_state(self):
        # Unattended bulk rows have no node ids but must show in the
        # sidebar pair count (state's dpo list length).
        store, _ = make_store()
        store.add_bulk_rows("t", BULK_PROMPT, [
            {"content": "a", "node_id": None},
            {"content": "b", "node_id": None},
        ], "rejected", "m")
        state = store.state("t")
        self.assertEqual(len(state["dpo"]), 2)

    def test_bulk_and_duel_rows_share_schema(self):
        # Both row shapes land in the same .dpo.jsonl — HF datasets loads
        # it via arrow, so bulk rows may only ADD keys, never diverge on
        # the shared ones.
        store, root = make_store()
        store.rate_dpo_pair("t", PROMPT, {"node_id": "a", "content": "A"},
                            {"node_id": "b", "content": "B"}, "m")
        store.add_bulk_rows("t", BULK_PROMPT,
                            [{"content": "c", "node_id": None}],
                            "rejected", "m")
        rows = [json.loads(l) for l in
                (root / "t.dpo.jsonl").read_text().splitlines()]
        self.assertLessEqual(set(rows[0]), set(rows[1]))


class TestStripThink(unittest.TestCase):

    def test_strips_think_blocks(self):
        self.assertEqual(
            strip_think_text("<think>secret plan</think>  the answer"),
            "the answer")
        self.assertEqual(
            strip_think_text("<|channel>gemma thoughts<channel|>\nreply"),
            "reply")

    def test_plain_text_untouched(self):
        self.assertEqual(strip_think_text("no tags here"), "no tags here")
        self.assertEqual(strip_think_text(""), "")


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

    async def test_rate_pair_with_generation_prompts(self):
        body = {
            "dataset": "t", "prompt": PROMPT,
            "pair": {
                "chosen": {"node_id": "a", "content": "better",
                           "gen_system": "be great"},
                "rejected": {"node_id": "b", "content": "worse",
                             "gen_system": None},
            },
        }
        resp = await self.client.request("POST", "/api/rate", json=body)
        self.assertEqual(resp.status, 200)
        row = json.loads(
            (Path(self.tmpdir) / "t.dpo.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["chosen_system"], "be great")
        self.assertIsNone(row["rejected_system"])
        # Non-string gen_system is rejected
        body["pair"]["chosen"]["gen_system"] = 42
        resp = await self.client.request("POST", "/api/rate", json=body)
        self.assertEqual(resp.status, 400)

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

    async def test_rate_bulk_saves_all_completions(self):
        # Review-flow "Save all": several candidates, one prompt, all on
        # the rejected side, carrying the source row's chosen column.
        resp = await self.client.request("POST", "/api/rate", json={
            "dataset": "t", "prompt": PROMPT,
            "bulk": {
                "completions": [{"node_id": "n1", "content": "bad 1"},
                                {"node_id": "n2", "content": "bad 2"}],
                "target": "rejected",
                "gen_system": "elicit bad",
                "source_row": {"chosen": "gold", "id": "src-1"},
            },
        })
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(len(data["dpo"]), 2)
        rows = [json.loads(l) for l in
                (Path(self.tmpdir) / "t.dpo.jsonl").read_text().splitlines()]
        self.assertEqual([r["rejected"] for r in rows], ["bad 1", "bad 2"])
        self.assertTrue(all(r["chosen"] == "gold" for r in rows))
        self.assertTrue(all(r["source_row_id"] == "src-1" for r in rows))
        self.assertTrue(all(r["rejected_system"] == "elicit bad" for r in rows))

    async def test_rate_bulk_rejects_bad_input(self):
        cases = [
            {"dataset": "t", "prompt": PROMPT,      # empty completions
             "bulk": {"completions": [], "target": "rejected"}},
            {"dataset": "t", "prompt": PROMPT,      # bad target
             "bulk": {"completions": [{"node_id": None, "content": "x"}],
                      "target": "best"}},
            {"dataset": "t", "prompt": PROMPT,      # non-string content
             "bulk": {"completions": [{"node_id": None, "content": 7}],
                      "target": "chosen"}},
        ]
        for body in cases:
            resp = await self.client.request("POST", "/api/rate", json=body)
            self.assertEqual(resp.status, 400, body)


class TestBulkRunRoute(AioHTTPTestCase):
    """The unattended bulk runner: /api/ratings/bulk streams SSE progress
    while writing each finished completion to the dataset."""

    async def get_application(self):
        self.engine = make_test_engine()
        self.engine.settings = MagicMock()
        self.engine.settings.system_prompt = "Be terse."
        self.tmpdir = tempfile.mkdtemp()
        self._cfg_patch = patch(
            "ezexl3.chat.server._load_config",
            lambda: {"ratings_dir": self.tmpdir},
        )
        self._cfg_patch.start()
        self.addCleanup(self._cfg_patch.stop)
        return create_app(self.engine)

    def _arm_engine(self, n):
        # is_loaded reads self.generator; generate_bulk is stubbed with a
        # deterministic async generator on the instance.
        self.engine.generator = MagicMock()

        async def fake_bulk(prompts, n=n, system_prompt=None):
            for i in range(len(prompts)):
                for j in range(n):
                    yield {"type": "row_done", "item": i, "cand": j,
                           "text": f"<think>hmm</think>reply {i}-{j}",
                           "eos_reason": "stop_string"}
            yield {"type": "progress", "done": len(prompts) * n,
                   "total": len(prompts) * n, "new_tokens": 8,
                   "elapsed": 1.0, "tps": 8.0}

        self.engine.generate_bulk = fake_bulk

    @staticmethod
    def _events(sse_text):
        out = []
        for line in sse_text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
        return out

    async def test_bulk_run_writes_rows_and_streams_progress(self):
        self._arm_engine(n=2)
        resp = await self.client.request("POST", "/api/ratings/bulk", json={
            "dataset": "t",
            "rows": [{"prompt": "Q1", "chosen": "gold1", "id": "r1"},
                     {"prompt": "Q2"}],
            "n": 2,
            "system_prompt": "elicit bad",
            "target": "rejected",
            "carry": True,
            "strip_think": True,
        })
        self.assertEqual(resp.status, 200)
        events = self._events(await resp.text())

        done = [e for e in events if e["type"] == "bulk_done"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["rows_written"], 4)
        self.assertEqual(done[0]["items_done"], 2)
        self.assertTrue(any(e["type"] == "saved" for e in events))

        rows = [json.loads(l) for l in
                (Path(self.tmpdir) / "t.dpo.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        # Prompt column: main system prompt + the row's user turn — never
        # the generation prompt.
        self.assertEqual(rows[0]["prompt"], [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Q1"},
        ])
        # strip_think removed the thought block server-side
        self.assertEqual(rows[0]["rejected"], "reply 0-0")
        # carry=True: Q1's rows hold the source chosen; Q2's are half-pairs
        q1 = [r for r in rows if r["source_row_id"] == "r1"]
        q2 = [r for r in rows if r["source_row_id"] is None]
        self.assertEqual(len(q1), 2)
        self.assertTrue(all(r["chosen"] == "gold1" for r in q1))
        self.assertTrue(all(r["chosen"] == "" for r in q2))
        self.assertTrue(all(r["rejected_system"] == "elicit bad" for r in rows))
        self.assertTrue(all(r["source"] == "bulk" for r in rows))

    async def test_bulk_run_without_carry_keeps_id_only(self):
        self._arm_engine(n=1)
        resp = await self.client.request("POST", "/api/ratings/bulk", json={
            "dataset": "t",
            "rows": [{"prompt": "Q1", "chosen": "gold1", "id": "r1"}],
            "n": 1, "target": "rejected", "carry": False,
        })
        self.assertEqual(resp.status, 200)
        await resp.text()
        row = json.loads((Path(self.tmpdir) / "t.dpo.jsonl").read_text())
        self.assertEqual(row["chosen"], "")            # column not carried
        self.assertEqual(row["source_row_id"], "r1")   # id still rides along

    async def test_bulk_run_validation(self):
        self._arm_engine(n=1)
        cases = [
            {"dataset": "../evil", "rows": [{"prompt": "q"}],
             "n": 1, "target": "rejected"},
            {"dataset": "t", "rows": [], "n": 1, "target": "rejected"},
            {"dataset": "t", "rows": [{"prompt": "  "}],
             "n": 1, "target": "rejected"},
            {"dataset": "t", "rows": [{"prompt": "q"}],
             "n": 0, "target": "rejected"},
            {"dataset": "t", "rows": [{"prompt": "q"}],
             "n": 9, "target": "rejected"},
            {"dataset": "t", "rows": [{"prompt": "q"}],
             "n": 1, "target": "best"},
        ]
        for body in cases:
            resp = await self.client.request(
                "POST", "/api/ratings/bulk", json=body)
            self.assertEqual(resp.status, 400, body)

    async def test_bulk_run_requires_loaded_model(self):
        # No _arm_engine: generator stays None
        resp = await self.client.request("POST", "/api/ratings/bulk", json={
            "dataset": "t", "rows": [{"prompt": "q"}],
            "n": 1, "target": "rejected",
        })
        self.assertEqual(resp.status, 400)


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------

class TestUiWiring(unittest.TestCase):

    def test_chat_ui_has_rating_controls(self):
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        self.assertIn('id="ratings-dataset"', html)
        self.assertIn('id="ratings-dir"', html)
        # Two-level top-bar toggle: Chat/Preference picks the scope, then
        # DPO/KTO picks the format once Preference is on.
        self.assertIn('id="chat-mode-toggle"', html)
        self.assertIn('id="pref-kind-toggle"', html)
        self.assertIn('data-scope="chat"', html)
        self.assertIn('data-scope="pref"', html)
        self.assertIn('data-mode="kto"', html)
        self.assertIn('data-mode="dpo"', html)
        self.assertIn('id="ratings-sys-a"', html)
        self.assertIn('id="ratings-sys-b"', html)
        self.assertIn('js/ratings.js', html)

    def test_capture_defaults_to_off(self):
        # Chat (no capture) is the default: the scope toggle pre-selects
        # it and the JS falls back to 'off' for unknown/unset persisted
        # modes. DPO is the format Preference opens on first use.
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        self.assertIn('data-scope="chat" class="active"', html)
        self.assertIn('data-mode="dpo" class="active"', html)
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        self.assertIn("let ratingsMode = 'off'", js)
        self.assertIn("let lastPrefKind = 'dpo'", js)

    def test_preference_controls_hide_until_enabled(self):
        # The decluttering contract: the whole Preference Data block is
        # hidden in Chat mode, and the DPO-only group inside it hides
        # under KTO, where none of those controls do anything.
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        self.assertIn('id="pref-panel" hidden', html)
        self.assertIn('id="pref-dpo-only"', html)
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        self.assertIn("function syncModeToggles()", js)
        # syncModeToggles owns all three visibility decisions.
        for target in ("pref-panel", "pref-dpo-only", "pref-kind-toggle"):
            self.assertIn(target, js)

    def test_render_wires_rating_controls(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/render.js").read_text()
        self.assertIn("rateNode", js)
        self.assertIn("renderDuelChoice", js)
        # Duel judgment controls: per-candidate marks + commit/regen/skip
        for name in ("setDuelMark", "commitDuel", "skipDuel",
                     "regenerateDuelCandidates"):
            self.assertIn(name, js)
        # Off mode renders no rating controls
        self.assertIn("ratingsMode !== 'off'", js)

    def test_ratings_js_defines_api(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        for name in ("rateNode", "setDuelMark", "commitDuel", "skipDuel",
                     "setRatingsMode", "refreshRatings", "buildPromptTurns",
                     "removePairFor", "duelSystemPrompts"):
            self.assertIn(name, js)
        # Committed pairs carry the generation-prompt provenance
        self.assertIn("gen_system", js)

    def test_chat_js_runs_duels_in_dpo_mode(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/chat.js").read_text()
        self.assertIn("runDuel", js)
        self.assertIn("regenerateDuelCandidates", js)
        self.assertIn("ratingsMode === 'dpo'", js)
        # Duels pass the per-candidate generation prompts to /api/chat
        self.assertIn("duelSystemPrompts", js)
        self.assertIn("system_prompts", js)

    def test_chat_ui_has_bulk_controls(self):
        html = (REPO_ROOT / "ezexl3/chat/static/index.html").read_text()
        for el_id in ("ratings-queue-mode", "ratings-bulk-target",
                      "ratings-bulk-carry", "ratings-queue-review"):
            self.assertIn(f'id="{el_id}"', html)
        # Bulk allows a single candidate per prompt
        self.assertIn('id="ratings-batch" min="1"', html)

    def test_ratings_js_defines_bulk_api(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/ratings.js").read_text()
        for name in ("parseQueueRows", "queueRowFromEntry", "saveAllBulk",
                     "startBulkRun", "bulkQueueActive", "bulkCount",
                     "bulkGenSystem"):
            self.assertIn(name, js)
        # The unattended runner hits the server-side endpoint
        self.assertIn("/api/ratings/bulk", js)

    def test_render_js_wires_bulk_save_all(self):
        js = (REPO_ROOT / "ezexl3/chat/static/js/render.js").read_text()
        self.assertIn("saveAllBulk", js)
        self.assertIn("pendingDuel.bulk", js)


if __name__ == "__main__":
    unittest.main()
