// ── Ratings: KTO/DPO preference-data capture ────────────────────
// A header toggle picks the capture mode:
//   Off — normal chat, no capture UI (the default).
//   KTO — 👍/👎 on any assistant reply writes one independent labeled row.
//   DPO — each send/regen generates N candidates side by side; mark one
//         ▲ chosen and one ▼ rejected, then Commit writes the pair. ✗
//         marks a candidate failed so Regenerate replaces just that one.
// The prompt queue additionally supports a bulk-generate mode: every
// non-✗ reply is saved on one side (chosen|rejected) — reviewed batch by
// batch in the browser, or fully unattended via the server-side
// cross-prompt runner (/api/ratings/bulk).
// Rows land in <ratings_dir>/<dataset>.{kto,dpo}.jsonl, trainer-ready.

let ratingsDataset = 'chat';
let ratingsMode = 'off';    // 'off' | 'kto' | 'dpo'
let ratingsStripThink = false;  // strip thought blocks from saved rows
let ratingsBatch = 2;       // candidates per prompt (duel 2..8, bulk 1..8)
// Queue mode: 'duel' judges ▲/▼ pairs; 'bulk' saves every non-✗ reply on
// one side (chosen|rejected) — single system prompt (A), optionally
// carrying the source row's other column for trainer-ready pairs.
let ratingsQueueMode = 'duel';
let ratingsBulkTarget = 'rejected';  // side the generated replies fill
let ratingsBulkCarry = true;         // copy source row's other column
let ratingsQueueReview = false;      // bulk: review in-browser vs unattended
// Prompt queue: run a dataset of prompts in series, one fresh conversation
// each, auto-advancing after every commit/save/skip. Rows keep the source
// dataset's chosen/rejected/id columns for bulk mode.
//   {rows: [{prompt, chosen?, rejected?, id?}], index: <next to dispatch>,
//    active: bool, finished: bool, current: <row being judged>}
let promptQueue = {rows: [], index: 0, active: false, finished: false,
                   current: null};
// Unattended bulk run (server-side generation, SSE progress).
let bulkRun = {active: false, itemsDone: 0, totalItems: 0, rowsWritten: 0,
               tps: 0, preview: '', error: ''};
// Pending DPO duel awaiting judgment:
//   {userNodeId, ids: [aId, bId, …], marks: {nodeId: 'up'|'down'|'fail'},
//    bulk: bool, sourceRow: <queue row>|null}
let pendingDuel = null;
const ratingsState = {
  kto: new Map(),           // node_id -> bool
  pairs: [],                // [{chosen, rejected}] node-id pairs on disk
  counts: {kto: 0, dpo: 0},
  dir: '',
};

function getRating(nodeId) { return ratingsState.kto.get(nodeId); }

function duelCount() {
  // Candidates per DPO duel, clamped to the batch ceiling.
  const n = parseInt(ratingsBatch, 10);
  return Number.isFinite(n) ? Math.max(2, Math.min(8, n)) : 2;
}

function bulkCount() {
  // Candidates per bulk-generate prompt — unlike duels, 1 is allowed.
  const n = parseInt(ratingsBatch, 10);
  return Number.isFinite(n) ? Math.max(1, Math.min(8, n)) : 1;
}

function bulkQueueActive() {
  // Queue-driven sends generate in bulk mode (single prompt, save-all UI).
  return promptQueue.active && ratingsQueueMode === 'bulk';
}

function bulkGenSystem() {
  // Bulk generation system prompt: System Prompt A for every candidate.
  const el = document.getElementById('ratings-sys-a');
  const v = el ? el.value.trim() : '';
  return v || null;
}

function duelSystemPromptsFor(n) {
  // Per-candidate generation system prompts for DPO duels, length n:
  // candidates A and B take the two sidebar fields; any further candidates
  // use the main (trained) prompt. null = use the main prompt.
  const read = id => {
    const el = document.getElementById(id);
    const v = el ? el.value.trim() : '';
    return v || null;
  };
  const a = read('ratings-sys-a');
  const b = read('ratings-sys-b');
  return Array.from({length: n}, (_, i) => (i === 0 ? a : i === 1 ? b : null));
}
function pairForNode(nodeId) {
  return ratingsState.pairs.find(p => p.chosen === nodeId) || null;
}

function applyRatingsPayload(data) {
  ratingsState.kto = new Map(Object.entries(data.kto || {}));
  ratingsState.pairs = (data.dpo || [])
    .filter(p => p.chosen && p.rejected)
    .map(p => ({chosen: p.chosen, rejected: p.rejected}));
  ratingsState.counts = {
    kto: Object.keys(data.kto || {}).length,
    dpo: (data.dpo || []).length,
  };
  ratingsState.dir = data.dir || '';
  updateRatingsSidebar(data.datasets || []);
}

async function refreshRatings() {
  try {
    const res = await fetch('/api/ratings?dataset=' + encodeURIComponent(ratingsDataset));
    if (!res.ok) return;
    applyRatingsPayload(await res.json());
    renderActiveTree();
  } catch {}
}

// ── Capture mode ────────────────────────────────────────────────

function setRatingsMode(mode, persist = true) {
  ratingsMode = (mode === 'dpo' || mode === 'kto') ? mode : 'off';
  document.querySelectorAll('#rating-mode-toggle button').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === ratingsMode));
  // A queue/duel can't outlive a switch away from DPO. Stop the queue
  // first so the skip below doesn't auto-advance it.
  if (ratingsMode !== 'dpo') {
    if (promptQueue.active) stopQueue();
    if (pendingDuel) skipDuel();
  }
  if (persist) {
    fetch('/api/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ratings_mode: ratingsMode}),
    }).catch(() => {});
  }
  renderActiveTree();
}

// ── Row construction ────────────────────────────────────────────

function stripThink(text) {
  // Remove thought blocks — tags included — from text bound for a
  // preference dataset: gemma4 <|channel>...<channel|> and generic
  // <think>...</think>. No-op unless the sidebar checkbox is on.
  if (!ratingsStripThink || !text) return text;
  return text
    .replace(/<\|channel>[\s\S]*?<channel\|>/g, '')
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/^\s+/, '');
}

function buildPromptTurns(assistantNodeId) {
  // Full conversation history before this assistant message, as
  // TRL-conversational {role, content} turns (system prompt included).
  const assistNode = tree.nodes.get(assistantNodeId);
  const userNode = assistNode ? tree.nodes.get(assistNode.parentId) : null;
  if (!userNode) return null;
  const turns = [];
  const sys = (settings.system_prompt || '').trim();
  if (sys) turns.push({role: 'system', content: sys});
  for (const [u, a] of getActivePathUpTo(userNode.id)) {
    turns.push({role: 'user', content: u});
    if (a != null) turns.push({role: 'assistant', content: stripThink(a)});
  }
  turns.push({role: 'user', content: userNode.content});
  return turns;
}

// ── Actions ─────────────────────────────────────────────────────

async function rateNode(nodeId, label) {
  const node = tree.nodes.get(nodeId);
  if (!node || node.role !== 'assistant' || !node.content.trim()) return;
  const prompt = buildPromptTurns(nodeId);
  if (!prompt) return;

  // Clicking the active rating again clears it.
  const current = ratingsState.kto.get(nodeId);
  const newLabel = (current === label) ? null : label;

  await postRate({
    dataset: ratingsDataset,
    prompt,
    kto: {node_id: nodeId, completion: stripThink(node.content),
          label: newLabel},
  });
}

function setDuelMark(nodeId, mark) {
  // Toggle a judgment mark on one duel candidate. ▲ (up) and ▼ (down)
  // are exclusive across candidates — a pair has one of each; ✗ (fail)
  // can sit on both.
  if (!pendingDuel || !pendingDuel.ids.includes(nodeId)) return;
  if (pendingDuel.bulk && mark !== 'fail') return;  // bulk judges ✗ only
  const marks = pendingDuel.marks;
  if (marks[nodeId] === mark) {
    delete marks[nodeId];
  } else {
    if (mark !== 'fail') {
      for (const id of pendingDuel.ids) {
        if (marks[id] === mark) delete marks[id];
      }
    }
    marks[nodeId] = mark;
  }
  renderActiveTree();
}

function _activateDuelNode(nodeId) {
  // Point the conversation's active branch at *nodeId*.
  const node = tree.nodes.get(nodeId);
  const parent = node ? tree.nodes.get(node.parentId) : null;
  if (parent) {
    const idx = parent.children.indexOf(nodeId);
    if (idx >= 0) parent.activeChild = idx;
  }
}

async function commitDuel() {
  // Save the judged pair (▲ chosen / ▼ rejected) and continue the
  // conversation from the chosen candidate.
  if (!pendingDuel) return;
  const {ids, marks} = pendingDuel;
  const chosenId = ids.find(id => marks[id] === 'up');
  const rejectedId = ids.find(id => marks[id] === 'down');
  if (!chosenId || !rejectedId) return;
  pendingDuel = null;
  _activateDuelNode(chosenId);
  const chosen = tree.nodes.get(chosenId);
  const rejected = tree.nodes.get(rejectedId);
  const prompt = chosen ? buildPromptTurns(chosenId) : null;
  if (chosen && rejected && prompt) {
    // genSystem records the generation-time system prompt when it
    // differed from the trained one (null otherwise) — metadata only.
    await postRate({
      dataset: ratingsDataset,
      prompt,
      pair: {
        chosen: {node_id: chosenId, content: stripThink(chosen.content),
                 gen_system: chosen.genSystem ?? null},
        rejected: {node_id: rejectedId, content: stripThink(rejected.content),
                   gen_system: rejected.genSystem ?? null},
      },
    });
  }
  renderActiveTree();
  inputBox.focus();
  queueAfterJudgment();
}

async function saveAllBulk() {
  // Bulk review flow: save every candidate not marked ✗ as one dataset
  // row each (generated text on the ratingsBulkTarget side), then
  // continue/advance. With every candidate ✗-marked this is a skip.
  if (!pendingDuel || !pendingDuel.bulk) return;
  const {ids, marks} = pendingDuel;
  const keptIds = ids.filter(id => marks[id] !== 'fail');
  const sourceRow = pendingDuel.sourceRow || null;
  const completions = keptIds
    .map(id => {
      const node = tree.nodes.get(id);
      return node ? {node_id: id, content: stripThink(node.content)} : null;
    })
    .filter(c => c && c.content.trim());
  const continueId = keptIds[0] || ids[0];
  pendingDuel = null;
  _activateDuelNode(continueId);
  const prompt = buildPromptTurns(continueId);
  if (completions.length && prompt) {
    const genSys = tree.nodes.get(completions[0].node_id)?.genSystem ?? null;
    let source_row = null;
    if (sourceRow && ratingsBulkCarry) source_row = sourceRow;
    else if (sourceRow && sourceRow.id) source_row = {id: sourceRow.id};
    await postRate({
      dataset: ratingsDataset,
      prompt,
      bulk: {completions, target: ratingsBulkTarget,
             gen_system: genSys, source_row},
    });
  }
  renderActiveTree();
  inputBox.focus();
  queueAfterJudgment();
}

function skipDuel() {
  // Dismiss the duel without recording: continue from the ▲ candidate
  // if one is marked, otherwise from B.
  if (!pendingDuel) return;
  const {ids, marks} = pendingDuel;
  const keepId = ids.find(id => marks[id] === 'up') || ids[ids.length - 1];
  pendingDuel = null;
  _activateDuelNode(keepId);
  renderActiveTree();
  inputBox.focus();
  queueAfterJudgment();
}

async function removePairFor(nodeId) {
  // Click on the "preferred" badge: withdraw the recorded pair.
  const p = pairForNode(nodeId);
  if (!p) return;
  const prompt = buildPromptTurns(nodeId);
  if (!prompt) return;
  await postRate({
    dataset: ratingsDataset,
    prompt,
    pair: {
      chosen: {node_id: p.chosen},
      rejected: {node_id: p.rejected},
      remove: true,
    },
  });
}

async function postRate(body) {
  try {
    const res = await fetch('/api/rate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      console.error('Rating failed:', data.error || res.status);
      await refreshRatings();  // resync from disk truth
      return;
    }
    applyRatingsPayload(data);
  } catch (e) {
    console.error('Rating failed:', e);
  }
  renderActiveTree();
}

// ── Prompt queue ────────────────────────────────────────────────
// Run a dataset of prompts in series. Each prompt gets a fresh
// conversation; after the duel is committed or skipped the queue
// advances to the next prompt automatically.

function queueAfterJudgment() {
  // Called at the end of commit/skip: advance the queue if it's running.
  if (promptQueue.active && ratingsMode === 'dpo') advanceQueue();
}

function startQueue() {
  if (generating || !modelLoaded || bulkRun.active) return;
  const ta = document.getElementById('ratings-queue-text');
  const rows = parseQueueRows(ta ? ta.value : '');
  if (!rows.length) return;
  if (ratingsQueueMode === 'bulk' && !ratingsQueueReview) {
    startBulkRun(rows);          // unattended: server-side cross-prompt run
    return;
  }
  if (ratingsMode !== 'dpo') setRatingsMode('dpo');
  promptQueue = {rows, index: 0, active: true, finished: false, current: null};
  advanceQueue();
}

function stopQueue() {
  promptQueue.active = false;
  updateQueueUI();
}

function advanceQueue() {
  if (!promptQueue.active) return;
  if (promptQueue.index >= promptQueue.rows.length) {
    promptQueue.active = false;   // reached the end
    promptQueue.finished = true;
    updateQueueUI();
    inputBox.focus();
    return;
  }
  runQueuePrompt();
}

async function runQueuePrompt() {
  if (!modelLoaded) { stopQueue(); return; }
  const row = promptQueue.rows[promptQueue.index];
  promptQueue.index++;
  promptQueue.current = row;     // source columns for bulk save-all
  updateQueueUI();
  await clearContext();          // fresh conversation per prompt
  inputBox.value = row.prompt;
  await sendMessage();           // DPO mode → runDuel arms the judgment UI
}

// ── Unattended bulk run ─────────────────────────────────────────
// POST the parsed rows once; the server enqueues every prompt × N jobs in
// one generator pool (batching across prompts), writes each reply to the
// dataset as it finishes, and streams progress back over SSE. Stop calls
// /api/stop — everything already written stays.

async function startBulkRun(rows) {
  bulkRun = {active: true, itemsDone: 0, totalItems: rows.length,
             rowsWritten: 0, tps: 0, preview: '', error: ''};
  updateQueueUI();
  try {
    const resp = await fetch('/api/ratings/bulk', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dataset: ratingsDataset,
        rows,
        n: bulkCount(),
        system_prompt: bulkGenSystem(),
        target: ratingsBulkTarget,
        carry: ratingsBulkCarry,
        strip_think: ratingsStripThink,
      }),
    });
    if (!resp.ok || !resp.body) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        try { applyBulkEvent(JSON.parse(payload)); } catch {}
      }
    }
  } catch (e) {
    console.error('Bulk run failed:', e);
    bulkRun.error = e.message || String(e);
  }
  bulkRun.active = false;
  promptQueue.finished = false;   // bulk summary owns the status line
  updateQueueUI();
  refreshRatings();
}

function applyBulkEvent(evt) {
  switch (evt.type) {
    case 'saved':
      bulkRun.itemsDone = evt.items_done;
      bulkRun.totalItems = evt.total_items;
      bulkRun.rowsWritten = evt.rows_written;
      if (evt.preview) bulkRun.preview = evt.preview;
      break;
    case 'progress':
      if (typeof evt.items_done === 'number') bulkRun.itemsDone = evt.items_done;
      if (typeof evt.total_items === 'number') bulkRun.totalItems = evt.total_items;
      if (typeof evt.rows_written === 'number') bulkRun.rowsWritten = evt.rows_written;
      if (typeof evt.tps === 'number') bulkRun.tps = evt.tps;
      break;
    case 'bulk_done':
      bulkRun.itemsDone = evt.items_done;
      bulkRun.rowsWritten = evt.rows_written;
      break;
    case 'error':
      bulkRun.error = evt.message || 'unknown error';
      break;
  }
  updateQueueUI();
}

function importPromptFile(file) {
  // Drop the raw file text into the box; parsing happens uniformly at
  // preview/start time, so pasted and loaded data behave identically
  // (and multi-line JSON prompts survive intact).
  if (!file) return;
  file.text().then(text => {
    const ta = document.getElementById('ratings-queue-text');
    if (ta) ta.value = text;
    promptQueue.finished = false;
    if (!bulkRun.active) {
      bulkRun = {active: false, itemsDone: 0, totalItems: 0,
                 rowsWritten: 0, tps: 0, preview: '', error: ''};
    }
    updateQueueUI();
  }).catch(e => console.error('Prompt file load failed:', e));
}

// ── Prompt parsing ──────────────────────────────────────────────
// Reduce whatever is in the box (pasted or loaded) to a list of queue
// rows {prompt, chosen?, rejected?, id?}. The model only ever sees the
// extracted prompt text — never raw JSON; chosen/rejected/id columns are
// kept so bulk mode can carry them into saved rows. Accepts a JSON array
// ([{…}, …]), JSONL (one object per line), or plain text (one prompt per
// line); the three may even be mixed by line.
function parseQueueRows(text) {
  const trimmed = (text || '').trim();
  if (!trimmed) return [];
  // Whole-text JSON first: a .json array/object or a single-line array.
  if (trimmed[0] === '[' || trimmed[0] === '{') {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed.map(queueRowFromEntry).filter(Boolean);
      const r = queueRowFromEntry(parsed);
      if (r) return [r];
    } catch { /* not one JSON value — fall through to line-by-line */ }
  }
  // Line-delimited: JSONL rows and/or plain-text prompts.
  const out = [];
  for (const raw of trimmed.split('\n')) {
    const s = raw.trim();
    if (!s) continue;
    if (s[0] === '{' || s[0] === '[') {
      try { const r = queueRowFromEntry(JSON.parse(s)); if (r) out.push(r); continue; }
      catch { /* not JSON after all — treat as plain text */ }
    }
    out.push({prompt: s});
  }
  return out;
}

function queueRowFromEntry(entry) {
  // One dataset entry → queue row. The prompt text comes from
  // promptFromRow; chosen/rejected (string, or a turn list reduced to
  // its last assistant turn) and id/source_row_id ride along for bulk.
  const prompt = promptFromRow(entry);
  if (!prompt) return null;
  const row = {prompt};
  if (entry && typeof entry === 'object' && !Array.isArray(entry)) {
    for (const key of ['chosen', 'rejected']) {
      const v = entry[key];
      if (typeof v === 'string' && v.trim()) row[key] = v;
      else if (Array.isArray(v)) {
        const t = lastAssistantTurn(v);
        if (t) row[key] = t;
      }
    }
    const id = entry.id ?? entry.source_row_id;
    if (typeof id === 'string' || typeof id === 'number') row.id = String(id);
  }
  return row;
}

function lastAssistantTurn(turns) {
  // Content of the last assistant/gpt turn in a chosen/rejected turn list.
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i];
    const c = t && (t.content ?? t.value ?? t.text);
    const role = t && (t.role || t.from);
    if (typeof c === 'string' && c.trim() &&
        (role === 'assistant' || role === 'gpt' || role === 'model')) {
      return c.trim();
    }
  }
  return null;
}

function promptFromRow(row) {
  // Reduce one dataset row to its user-prompt text across the common
  // instruction/chat formats. Returns null when nothing usable is found.
  if (typeof row === 'string') return row.trim() || null;
  if (!row || typeof row !== 'object') return null;

  // Conversational turn lists — OpenAI/TRL `messages`, ShareGPT
  // `conversations`, TRL `prompt` as a turn list: take the last user turn.
  for (const key of ['messages', 'conversations', 'conversation', 'prompt', 'chat']) {
    if (Array.isArray(row[key])) {
      const t = lastUserTurn(row[key]);
      if (t) return t;
    }
  }

  // Alpaca-style: instruction (+ optional input).
  if (typeof row.instruction === 'string' && row.instruction.trim()) {
    const instr = row.instruction.trim();
    const inp = typeof row.input === 'string' ? row.input.trim() : '';
    return inp ? `${instr}\n\n${inp}` : instr;
  }

  // Single string fields, most-specific first.
  for (const key of ['prompt', 'text', 'content', 'question', 'query',
                     'input', 'user', 'human', 'q', 'source']) {
    if (typeof row[key] === 'string' && row[key].trim()) return row[key].trim();
  }
  return null;
}

function lastUserTurn(turns) {
  // Content of the last user/human turn; else the last turn with content.
  const contentOf = t => (t && (t.content ?? t.value ?? t.text));
  const roleOf = t => (t && (t.role || t.from));
  for (let i = turns.length - 1; i >= 0; i--) {
    const c = contentOf(turns[i]);
    const role = roleOf(turns[i]);
    if (typeof c === 'string' && c.trim() &&
        (role === 'user' || role === 'human' || role === 'prompter')) return c.trim();
  }
  for (let i = turns.length - 1; i >= 0; i--) {
    const c = contentOf(turns[i]);
    if (typeof c === 'string' && c.trim()) return c.trim();
  }
  return null;
}

function updateQueueUI() {
  const info = document.getElementById('queue-status');
  const startBtn = document.getElementById('queue-start-btn');
  const stopBtn = document.getElementById('queue-stop-btn');
  const running = promptQueue.active || bulkRun.active;
  if (info) {
    if (bulkRun.active) {
      let html = `Bulk: ${bulkRun.itemsDone} / ${bulkRun.totalItems} prompts · ` +
                 `${bulkRun.rowsWritten} rows written` +
                 (bulkRun.tps ? ` · ${bulkRun.tps} tok/s` : '');
      if (bulkRun.preview) {
        const clip = bulkRun.preview.replace(/\s+/g, ' ').trim().slice(0, 120);
        html += `<br>latest: <em>${escHtml(clip)}…</em>`;
      }
      info.innerHTML = html;
      info.style.display = '';
    } else if (bulkRun.rowsWritten || bulkRun.error) {
      info.textContent = bulkRun.error
        ? `Bulk run failed: ${bulkRun.error} — ${bulkRun.rowsWritten} rows kept`
        : `Bulk run finished — ${bulkRun.rowsWritten} rows from ` +
          `${bulkRun.itemsDone} prompt${bulkRun.itemsDone === 1 ? '' : 's'}`;
      info.style.display = '';
    } else if (promptQueue.active) {
      info.textContent =
        `Running prompt ${promptQueue.index} / ${promptQueue.rows.length}`;
      info.style.display = '';
    } else if (promptQueue.finished) {
      const n = promptQueue.rows.length;
      info.textContent = `Queue finished — ${n} prompt${n === 1 ? '' : 's'}`;
      info.style.display = '';
    } else {
      // Idle: live-preview what will actually be sent (extracted text,
      // never raw JSON) so mis-parsed data is obvious before you start.
      const ta = document.getElementById('ratings-queue-text');
      const rows = parseQueueRows(ta ? ta.value : '');
      if (rows.length) {
        const first = rows[0].prompt.replace(/\s+/g, ' ').trim();
        const clip = first.length > 60 ? first.slice(0, 60) + '…' : first;
        const withOther = rows.filter(r => r.chosen || r.rejected).length;
        info.innerHTML =
          `${rows.length} prompt${rows.length === 1 ? '' : 's'} ready` +
          (withOther ? ` (${withOther} with chosen/rejected)` : '') +
          ` · first: <em>${escHtml(clip)}</em>`;
        info.style.display = '';
      } else {
        info.style.display = 'none';
      }
    }
  }
  if (startBtn) startBtn.style.display = running ? 'none' : '';
  if (stopBtn) stopBtn.style.display = running ? '' : 'none';
}

// ── Sidebar ─────────────────────────────────────────────────────

function updateRatingsSidebar(datasets) {
  const info = document.getElementById('ratings-info');
  if (info) {
    const c = ratingsState.counts;
    info.textContent =
      `${c.kto} rated · ${c.dpo} pairs → ${ratingsState.dir}`;
  }
  const list = document.getElementById('ratings-datasets');
  if (list) {
    list.innerHTML = '';
    for (const name of datasets) {
      const opt = document.createElement('option');
      opt.value = name;
      list.appendChild(opt);
    }
  }
}

async function initRatings() {
  try {
    const cfgRes = await fetch('/api/config');
    const cfg = await cfgRes.json();
    if (cfg.ratings_dataset) ratingsDataset = cfg.ratings_dataset;
    setRatingsMode(cfg.ratings_mode, false);
    const nameInput = document.getElementById('ratings-dataset');
    const dirInput = document.getElementById('ratings-dir');
    nameInput.value = ratingsDataset;
    if (cfg.ratings_dir) dirInput.value = cfg.ratings_dir;

    // Strip-thinking toggle (persisted)
    const stripEl = document.getElementById('ratings-strip-think');
    if (stripEl) {
      ratingsStripThink = !!cfg.ratings_strip_think;
      stripEl.checked = ratingsStripThink;
      stripEl.addEventListener('change', () => {
        ratingsStripThink = stripEl.checked;
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_strip_think: ratingsStripThink}),
        }).catch(() => {});
      });
    }

    // Duel batch size (persisted). Recurrent models allocate this many
    // cache slots at load, so a change only takes full effect on the next
    // model reload; non-recurrent models pick it up immediately.
    const batchEl = document.getElementById('ratings-batch');
    if (batchEl) {
      // 1 is a valid bulk count; duelCount() still floors duels at 2.
      ratingsBatch = Math.max(1, Math.min(8, parseInt(cfg.ratings_batch, 10) || 2));
      batchEl.value = ratingsBatch;
      batchEl.addEventListener('change', () => {
        ratingsBatch = Math.max(1, Math.min(8, parseInt(batchEl.value, 10) || 2));
        batchEl.value = ratingsBatch;
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_batch: ratingsBatch}),
        }).catch(() => {});
      });
    }

    // Queue mode (duel | bulk) + bulk options, all persisted.
    const queueModeEl = document.getElementById('ratings-queue-mode');
    const bulkOptsEl = document.getElementById('ratings-bulk-opts');
    const showBulkOpts = () => {
      if (bulkOptsEl) {
        bulkOptsEl.style.display = ratingsQueueMode === 'bulk' ? '' : 'none';
      }
    };
    if (queueModeEl) {
      ratingsQueueMode = cfg.ratings_queue_mode === 'bulk' ? 'bulk' : 'duel';
      queueModeEl.value = ratingsQueueMode;
      queueModeEl.addEventListener('change', () => {
        ratingsQueueMode = queueModeEl.value === 'bulk' ? 'bulk' : 'duel';
        showBulkOpts();
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_queue_mode: ratingsQueueMode}),
        }).catch(() => {});
      });
    }
    const bulkTargetEl = document.getElementById('ratings-bulk-target');
    if (bulkTargetEl) {
      ratingsBulkTarget =
        cfg.ratings_bulk_target === 'chosen' ? 'chosen' : 'rejected';
      bulkTargetEl.value = ratingsBulkTarget;
      bulkTargetEl.addEventListener('change', () => {
        ratingsBulkTarget =
          bulkTargetEl.value === 'chosen' ? 'chosen' : 'rejected';
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_bulk_target: ratingsBulkTarget}),
        }).catch(() => {});
      });
    }
    const bulkCarryEl = document.getElementById('ratings-bulk-carry');
    if (bulkCarryEl) {
      ratingsBulkCarry = cfg.ratings_bulk_carry !== false;  // default on
      bulkCarryEl.checked = ratingsBulkCarry;
      bulkCarryEl.addEventListener('change', () => {
        ratingsBulkCarry = bulkCarryEl.checked;
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_bulk_carry: ratingsBulkCarry}),
        }).catch(() => {});
      });
    }
    const reviewEl = document.getElementById('ratings-queue-review');
    if (reviewEl) {
      ratingsQueueReview = !!cfg.ratings_queue_review;
      reviewEl.checked = ratingsQueueReview;
      reviewEl.addEventListener('change', () => {
        ratingsQueueReview = reviewEl.checked;
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_queue_review: ratingsQueueReview}),
        }).catch(() => {});
      });
    }
    showBulkOpts();

    // Prompt queue controls
    const queueStart = document.getElementById('queue-start-btn');
    const queueStop = document.getElementById('queue-stop-btn');
    const queueLoad = document.getElementById('queue-load-btn');
    const queueFile = document.getElementById('queue-file-input');
    if (queueStart) queueStart.addEventListener('click', () => startQueue());
    if (queueStop) queueStop.addEventListener('click', () => {
      // A server-side bulk run is cancelled like any generation; rows
      // already written stay. The review queue just stops advancing.
      if (bulkRun.active) fetch('/api/stop', {method: 'POST'}).catch(() => {});
      else stopQueue();
    });
    if (queueLoad && queueFile) {
      queueLoad.addEventListener('click', () => queueFile.click());
      queueFile.addEventListener('change', e => {
        importPromptFile(e.target.files[0]);
        e.target.value = '';
      });
    }
    // Live-preview the parsed prompt count/first line as you paste or edit.
    const queueText = document.getElementById('ratings-queue-text');
    if (queueText) {
      let previewTimer = null;
      queueText.addEventListener('input', () => {
        promptQueue.finished = false;
        if (!bulkRun.active) {   // editing clears the last run's summary
          bulkRun = {active: false, itemsDone: 0, totalItems: 0,
                     rowsWritten: 0, tps: 0, preview: '', error: ''};
        }
        if (promptQueue.active) return;
        clearTimeout(previewTimer);
        previewTimer = setTimeout(updateQueueUI, 200);
      });
    }
    updateQueueUI();

    // Duel generation prompts (persisted; blank = main system prompt)
    for (const [id, key] of [['ratings-sys-a', 'ratings_system_a'],
                             ['ratings-sys-b', 'ratings_system_b']]) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (cfg[key]) el.value = cfg[key];
      el.addEventListener('change', () => {
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({[key]: el.value.trim()}),
        }).catch(() => {});
      });
    }

    document.querySelectorAll('#rating-mode-toggle button').forEach(btn =>
      btn.addEventListener('click', () => setRatingsMode(btn.dataset.mode)));

    nameInput.addEventListener('change', async () => {
      const name = nameInput.value.trim() || 'chat';
      nameInput.value = name;
      ratingsDataset = name;
      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ratings_dataset: name}),
      });
      await refreshRatings();
    });
    dirInput.addEventListener('change', async () => {
      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ratings_dir: dirInput.value.trim()}),
      });
      await refreshRatings();
    });
  } catch (e) {
    console.error('Ratings init failed:', e);
  }
  await refreshRatings();
}

// Exposed so send/regen can await mode restoration — a message sent
// right after page load must not race the config fetch and go out in
// the wrong capture mode. initRatings never rejects.
const ratingsReady = initRatings();
