# Handoff — 2026-07-12 (evening): chat preference-data capture (KTO / DPO)

Second session of the day (the morning parity-pass handoff is
`HANDOFF-2026-07-12.md`). Everything below is on the local branch
`chat-preference-data`, branched from `dev` @ 8891acb.

## Intent

Start of the "preference training" arc, from pineapple's Discord proposals
(2026-07-07) plus UnstableLlama's additions. The agreed shape:

- **Phase 1 (this session + next): data collection + inference.** Rate
  completions in the chat UI, store them in structured trainer-ready
  datasets, later add a dataset browser/editor and a training-run
  launcher. Training itself always runs via the exllamav3 fork's scripts
  (`training/qlora_train_pref.py`) — WebUI = frontend, CLI/fork = backend.
- **Phase 2 (deferred): online/interactive training** — pineapple's
  internal queue that takes a gradient step when a batch fills. Needs the
  fork's training path and the chat inference path to coexist; deliberately
  not designed yet.
- **Checkpoint policy for Phase 2 (deferred, shaped)**: rolling
  multi-timescale EMA taps blended at inference time — see
  `DESIGN-ema-checkpoint-blending.md`. Key enabler verified by reading
  code (NOT yet live-tested): the fork's inference `LoRA` supports multiple
  simultaneous adapters with per-adapter `lora_scaling`.

## What landed

### Ratings store + API (`ezexl3/chat/ratings.py`, `server.py`)
- JSONL files `<datasets_dir>/<dataset>.{kto,dpo}.jsonl`, one row per
  rating, in **exactly** the column format `qlora_train_pref.py` reads
  with default keys:
  - KTO: `{"prompt": [turns], "completion": str, "label": bool}`
  - DPO: `{"prompt": [turns], "chosen": str, "rejected": str}`
  - plus provenance keys the trainer ignores: `node_id` /
    `chosen_node_id` / `rejected_node_id`, `source` ("auto"|"manual"),
    `model` (stamped server-side from the loaded model), `ts`.
- `prompt` stores the **full conversation history** as {role, content}
  turns, system prompt included, final turn always `user` (enforced).
- Upsert semantics keyed by conversation-tree node id: re-rating replaces
  the row, clearing removes it. Auto DPO pairs are the 👍×👎 cross product
  within one sibling group, rebuilt on every rating change in that group
  (`sync_dpo_auto`); manual ⚖ pairs are never touched by the resync.
- Atomic rewrites (tmp + `os.replace`), `threading.Lock` around
  read-modify-write, hand-edited / foreign JSONL lines preserved verbatim.
- Routes: `GET /api/ratings?dataset=`, `POST /api/rate` (kto / group /
  manual sections, full validation). Dataset name regex-guarded (no path
  tricks). Config keys `ratings_dir` (empty = XDG default
  `~/.local/share/ezexl3/preference_data`) and `ratings_dataset` persist
  in the existing `~/.config/ezexl3/ui.json`.

### Chat UI (`ratings.js` new; `render.js`, `index.html`, `style.css`, `session.js`)
- 👍/👎 on every assistant message; ⚖ shown when the message has sibling
  regenerations ("prefer this version": manual pairs vs every sibling not
  rated 👍; click again withdraws).
- Buttons appear on hover; rated messages stay visibly marked (green /
  red / gold). Clicking the active thumb clears the rating.
- Sidebar "Preference Data" section: dataset name (datalist of existing),
  datasets dir, live `N rated · M pairs → dir` counter.
- Ratings restore across page reloads and session load (keyed by node id,
  which session JSON preserves).
- Rating an **edited** assistant message works — hand-written gold
  completions are rateable data.

### Tests (`tests/test_chat_ratings.py` — 19 tests, suite 358 offline green)
Store semantics, validation, trainer-format keys, arrow-schema
consistency, foreign-line preservation, route integration
(AioHTTPTestCase, config patched), UI wiring.

## Verified live (chat server on :8898, NO model loaded — GPU-free)
- Full UI flow with an injected conversation tree: thumbs → rows on disk,
  👍×👎 → auto pair, un-rating cascades the pair away, ⚖ → manual pair,
  sidebar counts, reload → marks restore, config round-trip.
- Produced JSONL loads via HF `datasets` with the trainer's default
  column names — the zero-conversion claim is proven at the loader level.
- NOT yet verified: an actual `qlora_train_pref.py` run on a collected
  dataset (loader-level only), and rating real generations with a loaded
  model (test used injected tree nodes; `model` field stamps correctly
  by code path but was empty in the test because nothing was loaded).

## Caveats / TODO

1. **Trainer is single-turn today**: the fork's `_prompt_text` collapses a
   conversational prompt to system + LAST user turn (`extract_single_turn`).
   Full history is stored but not yet trained on. Fork-side upgrade when
   multi-turn matters.
2. **⚖ after thumbs can duplicate a pair** (same chosen/rejected as an
   auto pair, differing only in `source`) — mildly double-weights that
   pair in training. Options: dedupe at train time, or make ⚖ skip
   already-auto-paired siblings. Decision pending.
3. **Dataset browser/editor** (view/edit/delete rows, per-dataset stats)
   — the "edit the database" half of Phase 1. Not started; likely its own
   view or a Training window alongside a run launcher for
   `qlora_train_pref.py` (dashboard already knows the spawn-CLI-and-SSE
   pattern).
4. **Live train smoke test**: run `qlora_train_pref.py --method kto` (and
   dpo) on a small collected dataset to close the loop end-to-end.
5. Older open items from the morning handoff still stand (chat SSE
   disconnect hardening, MTP-in-chat live test, nightly online tests).

## Environment notes
- Venv for all of this: `/home/unstable/exl3/tabbyAPI/venv/` (ezexl3 and
  the exllamav3 fork both editable-installed; verified resolving to the
  local repos).
- `ezexl3/.claude/launch.json` (untracked): chat on :8898, dashboard on
  :8899 for browser-preview dev.
- A `convert.py` quantization of Qwen3.6-27B-semancer-bf16 was running
  during this session; all verification was kept GPU-free around it.
