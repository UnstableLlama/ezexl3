# Handoff — 2026-09-12: "Datasets" tab + Single (SFT) dataset kind

On branch `dev` (uncommitted at the time of writing). Extends the
KTO/DPO capture from `HANDOFF-2026-07-12-preference-data.md` and the bulk
queue from `HANDOFF-2026-07-23-bulk-preference-gen.md`.

## Intent

UnstableLlama wanted the chat UI's **Preference** tab renamed to
**Datasets**, with a new first kind that runs a stack of prompts through
a model and saves every reply as plain conversation data — automatically
or with conditional approval / rerun — the same way the KTO and DPO
kinds work. Decisions from the design chat:

- Kind label **Single** (not "Chat": two adjacent Chat buttons read badly).
  File suffix `.single.jsonl`.
- Single-turn queue rows for now (the parser still reduces a `messages`
  row to its last user turn). Prior-turn context is a follow-up.
- Per-message 💾 save is included, mirroring KTO's thumbs.
- Candidates per prompt may be 1..8; every kept reply is its own row.

## What landed

### Store (`ezexl3/chat/ratings.py`)
- `add_single_rows(dataset, prompt, completions, model, source, source_row_id,
  gen_system)` — one `{"messages": prompt + [assistant]}` row per
  completion in `<dataset>.single.jsonl`, upsert by `node_id`; `source`
  is `manual` (💾) or `bulk` (queue). `remove_single_rows(dataset,
  node_ids)` withdraws.
- `state()` gains `single: {nodes: [...], count: N}`; `list_datasets()`
  sees `.single.jsonl`.
- Row layout is what the fork's SFT trainers read with `--messages-key
  messages` (verified by reading `training/qlora_train_native.py`'s
  `build_sft_examples`; multi-turn rows go through its segment renderer).

### Server (`ezexl3/chat/server.py`)
- `POST /api/rate` gains a `single` section: `{completions: [{node_id,
  content}], source?, gen_system?, source_row_id?}` or `{remove:
  [node_id, …]}`. Full validation, same snapshot response.
- `POST /api/ratings/bulk` gains `kind: "dpo"|"single"` (default dpo).
  Under single, `target`/`carry` are not required; each finished reply
  is written as a conversation row with `source: "bulk"` and the source
  row's `id`. System turn = the MAIN system prompt; the generation prompt
  (System Prompt A) is `gen_system` metadata — same convention as DPO.

### UI (`static/index.html`, `js/ratings.js`, `js/chat.js`, `js/render.js`, `style.css`)
- Top bar: **Chat | Datasets**, then **Single | DPO | KTO** (Single first
  and the cold-start default; persisted `ratings_mode` still wins).
- Sidebar header **Datasets**. Visibility rules live in
  `syncModeToggles()`: `#pref-gen-group` (candidates, queue, system
  prompts) shows under DPO and Single; under Single the DPO-only rows
  (`#ratings-queue-mode-row`, `#ratings-bulk-dpo-opts`, `#ratings-sys-b-row`)
  hide and the bulk options (review checkbox) show. New CSS rule
  `#pref-panel [hidden] { display: none !important }` because `.sb-row`
  sets `display: flex`.
- `candidateSendActive()` decides whether send/regen streams side-by-side
  candidates: DPO always, Single only while its queue drives
  (`bulkQueueActive()` is now true for any Single queue). Ordinary Single
  chatting streams one reply and shows 💾 save / ✓ saved on it
  (`saveSingleNode`, upsert/withdraw).
- Queue under Single: review on → the existing ✗ / Regenerate / "Save all
  & next" flow, `saveAllBulk()` posts the `single` section; review off →
  `startBulkRun()` with `kind: "single"`.
- Sidebar counter: `N single · N rated · N pairs → dir`.

### Tests (`tests/test_chat_ratings.py` — 54 green)
`TestSingleStore`, `/api/rate` single save/withdraw/review/validation,
`/api/ratings/bulk` kind=single end-to-end (stubbed engine) + validation,
UI wiring updated for the new labels/ids.

### Docs
README (Build training datasets), CHANGELOG (Unreleased), and
`DESIGN-preference-capture.md` (new "Single mode" section).

## Verified
- `pytest tests/test_chat_ratings.py` — 54 passed (tabbyAPI venv).
- Live chat server on :8898, **no model loaded**: toggles render, KTO mode
  restored from `ui.json`, per-kind sidebar visibility checked via computed
  styles for KTO/Single/DPO, no console errors. With an injected assistant
  node in Single mode: 💾 wrote a correct `messages` row to a scratch
  datasets dir, badge flipped to ✓ saved, click again withdrew it (file
  and counter back to 0). Review-flow "Save all" exercised with an
  injected bulk duel (see below).
- NOT verified: a real GPU run of a Single queue (unattended or review).
  The generation path is the unchanged duel/bulk batching; only the save
  target differs, which the stubbed-engine route test covers.

## Caveats / TODO
1. **Live GPU smoke test** of a Single queue into a fresh dataset, then a
   `qlora_train*.py --messages-key messages` run on it.
2. **Multi-turn queue rows** — feed a row's earlier turns as context so
   saved rows are multi-turn conversations.
3. `model` provenance is `""` when nothing is loaded (pre-existing).
