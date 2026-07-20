# Preference-data capture: v1 semantics and future work

Status as of 2026-07-20. The chat UI captures KTO/DPO training data for
the exllamav3 fork's `training/qlora_train_pref.py`; this doc records the
v1 design decisions and the ideas deliberately deferred.

## v1: explicit modes (header toggle)

An Off / KTO / DPO segmented toggle sits in the chat top bar (persisted
as `ratings_mode` in `~/.config/ezexl3/ui.json`). The rule: **one mode,
one gesture, one kind of row.**

- **Off** (the default) — normal chat. No rating controls render and no
  preference data is written; unknown or unset persisted modes fall back
  here.
- **KTO mode** — 👍/👎 on any assistant reply upserts one labeled row in
  `<dataset>.kto.jsonl`. Ratings are always independent: thumbs on
  sibling regenerations are separate rows, never combined into pairs.
- **DPO mode** — every send/regen generates **two candidates** (n=2) side
  by side, batched concurrently in one generator pass. Each candidate
  carries three judgment marks: **▲ chosen**, **▼ rejected**, **✗
  failed**. ▲ and ▼ are exclusive across the duo (a pair has one of
  each); ✗ can sit on one or both. **Regenerate** replaces just the ✗
  candidates with fresh generations (the kept candidate's text and mark
  survive). **Commit** — enabled once one candidate is ▲ and the other ▼
  — writes the chosen/rejected pair to `<dataset>.dpo.jsonl` and
  continues the conversation from the chosen reply. **Skip** continues
  without recording. Pairs are upserted keyed by the unordered candidate
  duo, so changing your mind replaces the row. A "✓ preferred" badge
  marks recorded picks (click to withdraw).

### DPO generation prompts (contrastive spoofing)

Two sidebar fields, **System Prompt A** and **System Prompt B**, override
the system prompt *at generation time only* for duel candidates A and B
(blank = the main system prompt; the fields persist in `ui.json` as
`ratings_system_a/b`). The intended use is contrastive: bias one
candidate toward the behavior being trained and the other away from it,
guaranteeing a behavioral delta in every pair — the same idea as RLCD
(Yang et al. 2023, arXiv:2307.12950), with a human judging each pair
instead of an automatic label. Assignment is fixed (A→A, B→B, no
shuffling), and the human still picks ▲/▼ — a "negative" prompt
sometimes produces the better reply, and blind label-by-prompt would
poison the data.

The dataset's `prompt` column always keeps the **main** system prompt —
that's what the model trains against. The generation prompts are
recorded per side as `chosen_system` / `rejected_system` metadata
columns (null when the main prompt was used); like `node_id`/`ts`, the
trainer selects columns by name and never sees them. Regenerating a ✗
candidate re-uses that slot's own prompt. A small "sys" tag on a duel
candidate shows the prompt it was generated under (hover to read it).

Every row carries provenance the trainer ignores: node ids, `source`
("duel" for v1 DPO picks), `model` (full model directory path), `ts`.

### Candidates per duel (batch size)

"Candidates per Duel" in the sidebar (2–4, persisted as
`ratings_duel_n`; the server caps `/api/chat`'s `n` at 4) sets how many
candidates each DPO send generates, batched concurrently in one
generator pass. Judging is **best-vs-worst regardless of n**: exactly
one ▲ and one ▼ make the pair, other candidates stay unrecorded
siblings, ✗ + Regenerate replaces any subset. This deliberately sidesteps
the best-vs-each cross-product subtlety that kept n=2 fixed at first —
one pair per duel, always. With n>2 the contrastive generation prompts
split by halves: System Prompt A covers the first ⌈n/2⌉ candidates, B
the rest.

### Prompt queue (batch capture from a JSONL)

A sidebar "Prompt Queue" panel opens a JSONL file of prompts and runs
through it as DPO duels: each prompt starts a **fresh single-turn
conversation** (a new root branch, no carried context), and every
Commit/Skip advances the queue and auto-starts the next prompt. Line
shapes accepted: JSON strings, `{"prompt": ...}`-style objects (also
`text`/`instruction`/`question`/`message`), turn lists (last user turn
wins), or plain non-JSON text lines used verbatim.

Progress is checkpointed per queue file — the next unserved 1-based
*file* line number, written to `<ratings_dir>/queue_checkpoints.json`
on every advance — so browser or server restarts resume where judging
left off. An explicit "Start at Line" overrides the checkpoint.
Advancing is idempotent (guarded by the served entry's index), a
mode-switch away from DPO abandons the pending duel without advancing,
and "Skip prompt" passes over a prompt without generating. Server state
is one open queue per server (`/api/queue`, `/api/queue/open`,
`/api/queue/advance`, `/api/queue/close`).

## Deferred / future work

1. **BOTH (hybrid) mode** — a third toggle position where thumbs and
   duels coexist, plus the ⚖ "prefer this version over its siblings"
   button from the first iteration (one click on the best sibling writes
   pairs against every non-👍 sibling). Shelved for v1 because the
   auto-pairing + scales semantics were too subtle to grasp without
   living with the simple modes first. The removed implementation is in
   git history (branch `chat-preference-data`, pre-2026-07-13) if wanted.
2. **Best-of-n ranking beyond one pair** — candidates-per-duel shipped
   with best-vs-worst semantics (one ▲/▼ pair per duel). Full ranking UX
   (best-vs-each, ordered ranks) reintroduces the cross-product
   subtlety — design before building.
3. **Dataset browser/editor** — view/edit/delete rows, per-dataset stats;
   the "edit the database" half of Phase 1.
4. **Training-run launcher** — a WebUI form that spawns
   `qlora_train_pref.py` (dashboard already has the spawn-CLI-and-SSE
   pattern).
5. **Train-time dedupe** — not needed for v1 (upsert-by-duo prevents
   duplicate pairs at capture time), but a trainer-side guard would also
   cover hand-merged datasets.
6. **Queue niceties** — multi-turn queue rows (seed full conversations,
   not just the last user turn), per-queue sampling overrides, and a
   progress bar over the dataset.
