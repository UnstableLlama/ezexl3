# Preference-data capture: v1 semantics and future work

Status as of 2026-07-13. The chat UI captures KTO/DPO training data for
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

## Single mode (added 2026-09-12)

The top-bar scope toggle became **Chat | Datasets** and the kind toggle
gained a first position, **Single**, for plain SFT data. One gesture, one
kind of row, as before:

- **💾 on any reply** saves the conversation up to and including it as
  one `{"messages": [...]}` row in `<dataset>.single.jsonl` (upsert by
  node id; click again withdraws). This is the fork's SFT layout
  (`--messages-key messages`), single- or multi-turn.
- **The prompt queue always runs bulk-style** under Single: n candidates
  per prompt (1 by default), ✗ / Regenerate / "Save all & next" when
  reviewing, or the unattended cross-prompt runner with `kind: "single"`.
  Every kept reply becomes its own row. Queue mode, save-side and carry
  are DPO-only and hide; the review checkbox and System Prompt A stay.
- **System prompts** follow the DPO convention: the saved system turn is
  always the main system prompt; the generation prompt (System Prompt A)
  is recorded as `gen_system` metadata. Generating under a spoofed
  prompt and training against the plain one is thus available for SFT.
- Queue rows are still reduced to their last user turn (single-turn
  prompts). Feeding a row's earlier turns as context is a follow-up.

## Deferred / future work

1. **BOTH (hybrid) mode** — a third toggle position where thumbs and
   duels coexist, plus the ⚖ "prefer this version over its siblings"
   button from the first iteration (one click on the best sibling writes
   pairs against every non-👍 sibling). Shelved for v1 because the
   auto-pairing + scales semantics were too subtle to grasp without
   living with the simple modes first. The removed implementation is in
   git history (branch `chat-preference-data`, pre-2026-07-13) if wanted.
2. **Candidates-per-generation `n` as a parameter** — KTO stays 1, DPO
   is fixed at 2 for v1. Raising DPO's n means best-of-n ranking UX and a
   decision about which pairs a pick implies (best-vs-each reintroduces
   the cross-product subtlety) — design before building.
3. **Dataset browser/editor** — view/edit/delete rows, per-dataset stats;
   the "edit the database" half of Phase 1.
4. **Training-run launcher** — a WebUI form that spawns
   `qlora_train_pref.py` (dashboard already has the spawn-CLI-and-SSE
   pattern).
5. **Train-time dedupe** — not needed for v1 (upsert-by-duo prevents
   duplicate pairs at capture time), but a trainer-side guard would also
   cover hand-merged datasets.
