# Preference-data capture: v1 semantics and future work

Status as of 2026-07-13. The chat UI captures KTO/DPO training data for
the exllamav3 fork's `training/qlora_train_pref.py`; this doc records the
v1 design decisions and the ideas deliberately deferred.

## v1: two explicit modes (header toggle)

A KTO / DPO segmented toggle sits in the chat top bar (persisted as
`ratings_mode` in `~/.config/ezexl3/ui.json`). The rule: **one mode, one
gesture, one kind of row.**

- **KTO mode** — 👍/👎 on any assistant reply upserts one labeled row in
  `<dataset>.kto.jsonl`. Ratings are always independent: thumbs on
  sibling regenerations are separate rows, never combined into pairs.
- **DPO mode** — every send/regen generates **two candidates** (n=2) side
  by side. Picking the better one writes a single chosen/rejected pair to
  `<dataset>.dpo.jsonl` and continues the conversation from the winner;
  Skip continues without recording. Pairs are upserted keyed by the
  unordered candidate duo, so changing your mind replaces the row.
  A "✓ preferred" badge marks recorded picks (click to withdraw).

Every row carries provenance the trainer ignores: node ids, `source`
("duel" for v1 DPO picks), `model` (full model directory path), `ts`.

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
