# Handoff — 2026-07-23: bulk preference-data generation (DPO)

On branch `dev`, extending the DPO prompt-queue (commits 87c2afb…3f08a8f).
Adds a **bulk-generate** queue mode alongside the existing pair-judging
duel mode, for building one side of a DPO dataset from a scaffold whose
other side is already filled.

## Intent

UnstableLlama has a DPO dataset
(`/home/unstable/datasets/mincraft_dpo/minecraft_220_selected_only.jsonl`,
220 rows) with a full `chosen` column (safe Minecraft-analogy refusals)
and empty `rejected`. Goal: generate the missing side in bulk — multiple
replies per prompt, many prompts at once — and save them to a *new*
dataset to combine later. No ▲/▼ judging; every generated reply is saved
on a chosen-or-rejected side you pick.

## What landed

### Store (`ezexl3/chat/ratings.py`)
- `RatingsStore.add_bulk_rows(dataset, prompt, completions, target,
  model, source_row, gen_system)` — appends one `.dpo.jsonl` row per
  completion, generated text on the `target` side (`chosen`|`rejected`).
  The opposite column comes from `source_row` (the loaded scaffold row)
  when carrying, else `""`. Rows carry `source:"bulk"`, `source_row_id`,
  and `chosen_system`/`rejected_system` (generation prompt, target side
  only). Completions with a `node_id` upsert (review-flow re-saves don't
  duplicate); unattended rows have no node id.
- `state()` now counts bulk rows (node-id-less) in the sidebar pair total.
- `strip_think_text()` — server-side mirror of the JS `stripThink`.

### Engine (`ezexl3/chat/inference.py`)
- `ChatEngine.generate_bulk(prompts, n, system_prompt)` — enqueues every
  prompt × n as one job pool so the generator **batches across prompts**
  (recurrent models capped by the `-ambs` cache slots = `ratings_batch`
  at load; non-recurrent batch freely). Fresh single-turn context per
  prompt; the persistent chat context is untouched. Yields `row_done` /
  `progress` events (no per-token streaming). Honors `cancel()`.
- `_build_input_ids` gained a `context=` override (bulk builds throwaway
  single-turn contexts without mutating `self.context`).

### Server (`ezexl3/chat/server.py`)
- `POST /api/ratings/bulk` (SSE) — the unattended runner. Body
  `{dataset, rows:[{prompt,chosen?,rejected?,id?}], n, system_prompt,
  target, carry, strip_think}`. Runs `generate_bulk`, writes each item's
  completions to disk as they finish, streams `saved`/`progress`/
  `bulk_done`. `POST /api/stop` cancels; rows already written stay. The
  trained `prompt` column = main system prompt + the row's user turn;
  `system_prompt` is generation-only metadata.
- `POST /api/rate` gained a `bulk` section for the review flow's "Save
  all" (many completions, one prompt).

### UI (`static/js/ratings.js`, `chat.js`, `render.js`, `index.html`)
- Sidebar: **Queue mode** select (duel|bulk); bulk sub-options — **Save
  replies as** chosen|rejected, **Carry source columns** (full pairs vs
  half-pairs to join later), **Review each batch** (in-browser vs
  unattended). All persisted in `ui.json` (`ratings_queue_mode`,
  `ratings_bulk_target`, `ratings_bulk_carry`, `ratings_queue_review`).
  Candidate count min lowered 2→1 (bulk allows 1; duels still floor at 2).
- Queue parser rewritten: `parseQueueRows` keeps whole rows
  (`{prompt, chosen?, rejected?, id?}`) instead of reducing to prompt
  strings. Prompt text still extracted so the model never sees raw JSON.
- Bulk review flow reuses the duel UI with ✗-only marks + "Save all &
  next". Bulk generation uses **System Prompt A for every candidate**
  (B ignored).
- Unattended flow POSTs `/api/ratings/bulk` and renders live SSE progress
  (`k/N prompts · R rows · tok/s`, newest-completion preview) + Stop.

### Tests (`tests/test_chat_ratings.py` — 42→ new bulk coverage, file green)
`TestBulkStore`, `TestStripThink`, `/api/rate` bulk validation,
`TestBulkRunRoute` (stubbed-engine SSE end-to-end), UI-wiring asserts.

## Verified
- Full offline suite: `pytest -m "not online"` — the 4 pre-existing
  failures below are the ONLY reds; everything I added is green.
- Live (chat server :8898, **no model loaded**): config round-trips
  bulk mode; queue parser reduces the real 220-row minecraft dataset to
  220 prompts, all 220 carrying `chosen`+`id`; live preview counts
  mixed JSONL+plaintext correctly; no console errors.
- NOT verified: an actual GPU bulk run (no model was loaded this
  session). The generation path reuses the duel batching that already
  works live; the runner logic is covered by the stubbed-engine test.

## Caveats / TODO
1. **4 pre-existing test failures** unrelated to this work —
   `test_chat_ngram.py::TestLoadPassesMinDraftLen` (×3) and
   `test_chat_streaming.py::…test_draft_load_reloads_recurrent_model`
   fail on a clean tree: their fixtures build `ChatEngine` without the
   `batch_slots` attr added in d3ced39. Spawned as a separate task.
2. **Recurrent-model batch ceiling**: "220 at once" is only true on
   non-recurrent models; recurrent ones run `ratings_batch` jobs in
   flight (still far faster than serial, needs a reload to raise).
3. **Live GPU smoke test** of a bulk run into a fresh dataset, then
   combine with the scaffold and confirm it trains.
