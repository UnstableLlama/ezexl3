// ── Ratings: KTO/DPO preference-data capture ────────────────────
// A header toggle picks the capture mode:
//   Off — normal chat, no capture UI (the default).
//   KTO — 👍/👎 on any assistant reply writes one independent labeled row.
//   DPO — each send/regen generates TWO candidates side by side; mark one
//         ▲ chosen and one ▼ rejected, then Commit writes the pair. ✗
//         marks a candidate failed so Regenerate replaces just that one.
// Rows land in <ratings_dir>/<dataset>.{kto,dpo}.jsonl, trainer-ready.

let ratingsDataset = 'chat';
let ratingsMode = 'off';    // 'off' | 'kto' | 'dpo'
let ratingsStripThink = false;  // strip thought blocks from saved rows
let ratingsBatch = 2;       // DPO duel candidates per turn (2..8)
// Prompt queue: run a dataset of prompts in series, one fresh conversation
// each, auto-advancing after every commit/skip.
//   {prompts: [str], index: <next to dispatch>, active: bool}
let promptQueue = {prompts: [], index: 0, active: false};
// Pending DPO duel awaiting judgment:
//   {userNodeId, ids: [aId, bId, …], marks: {nodeId: 'up'|'down'|'fail'}}
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
  if (generating) return;
  const ta = document.getElementById('ratings-queue-text');
  const prompts = (ta ? ta.value : '')
    .split('\n').map(s => s.trim()).filter(Boolean);
  if (!prompts.length) return;
  if (!modelLoaded) return;
  if (ratingsMode !== 'dpo') setRatingsMode('dpo');
  promptQueue = {prompts, index: 0, active: true};
  advanceQueue();
}

function stopQueue() {
  promptQueue.active = false;
  updateQueueUI();
}

function advanceQueue() {
  if (!promptQueue.active) return;
  if (promptQueue.index >= promptQueue.prompts.length) {
    promptQueue.active = false;   // reached the end
    updateQueueUI();
    inputBox.focus();
    return;
  }
  runQueuePrompt();
}

async function runQueuePrompt() {
  if (!modelLoaded) { stopQueue(); return; }
  const prompt = promptQueue.prompts[promptQueue.index];
  promptQueue.index++;
  updateQueueUI();
  await clearContext();          // fresh conversation per prompt
  inputBox.value = prompt;
  await sendMessage();           // DPO mode → runDuel arms the judgment UI
}

function importPromptFile(file) {
  if (!file) return;
  file.text().then(text => {
    const prompts = parsePromptFile(file.name, text);
    const ta = document.getElementById('ratings-queue-text');
    if (ta) ta.value = prompts.join('\n');
    if (!promptQueue.active) { promptQueue.index = 0; }
    updateQueueUI();
  }).catch(e => console.error('Prompt file load failed:', e));
}

function parsePromptFile(name, text) {
  // Returns an array of prompt strings. A .jsonl/.json file (or content
  // whose lines are all JSON) contributes one prompt per row via
  // promptFromRow; anything else is treated as one prompt per non-blank
  // line.
  const lines = text.split('\n').map(s => s.trim()).filter(Boolean);
  if (!lines.length) return [];
  const jsonish = /\.jsonl?$/i.test(name) ||
    lines.every(l => l.startsWith('{') || l.startsWith('['));
  if (!jsonish) return lines;
  const out = [];
  for (const line of lines) {
    let row;
    try { row = JSON.parse(line); } catch { out.push(line); continue; }
    const p = promptFromRow(row);
    if (p) out.push(p);
  }
  return out;
}

function promptFromRow(row) {
  // Pull a user prompt out of one parsed JSON row: a bare string field,
  // or the last user turn of a TRL-style turn list.
  if (typeof row === 'string') return row.trim();
  if (!row || typeof row !== 'object') return null;
  for (const key of ['prompt', 'messages', 'conversations']) {
    const v = row[key];
    if (typeof v === 'string' && v.trim()) return v.trim();
    if (Array.isArray(v)) {
      for (let i = v.length - 1; i >= 0; i--) {
        const t = v[i] || {};
        const role = t.role || t.from;
        const content = t.content ?? t.value;
        if ((role === 'user' || role === 'human') &&
            typeof content === 'string' && content.trim()) return content.trim();
      }
      const last = v[v.length - 1] || {};
      const c = last.content ?? last.value;
      if (typeof c === 'string' && c.trim()) return c.trim();
    }
  }
  for (const key of ['text', 'content', 'instruction', 'question', 'input']) {
    if (typeof row[key] === 'string' && row[key].trim()) return row[key].trim();
  }
  return null;
}

function updateQueueUI() {
  const info = document.getElementById('queue-status');
  const startBtn = document.getElementById('queue-start-btn');
  const stopBtn = document.getElementById('queue-stop-btn');
  const total = promptQueue.prompts.length;
  if (info) {
    if (promptQueue.active) {
      info.textContent = `Running prompt ${promptQueue.index} / ${total}`;
      info.style.display = '';
    } else if (total && promptQueue.index >= total) {
      info.textContent = `Queue finished — ${total} prompt${total === 1 ? '' : 's'}`;
      info.style.display = '';
    } else if (total) {
      info.textContent = `${total} prompt${total === 1 ? '' : 's'} loaded`;
      info.style.display = '';
    } else {
      info.style.display = 'none';
    }
  }
  if (startBtn) startBtn.style.display = promptQueue.active ? 'none' : '';
  if (stopBtn) stopBtn.style.display = promptQueue.active ? '' : 'none';
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
      ratingsBatch = Math.max(2, Math.min(8, parseInt(cfg.ratings_batch, 10) || 2));
      batchEl.value = ratingsBatch;
      batchEl.addEventListener('change', () => {
        ratingsBatch = Math.max(2, Math.min(8, parseInt(batchEl.value, 10) || 2));
        batchEl.value = ratingsBatch;
        fetch('/api/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ratings_batch: ratingsBatch}),
        }).catch(() => {});
      });
    }

    // Prompt queue controls
    const queueStart = document.getElementById('queue-start-btn');
    const queueStop = document.getElementById('queue-stop-btn');
    const queueLoad = document.getElementById('queue-load-btn');
    const queueFile = document.getElementById('queue-file-input');
    if (queueStart) queueStart.addEventListener('click', () => startQueue());
    if (queueStop) queueStop.addEventListener('click', () => stopQueue());
    if (queueLoad && queueFile) {
      queueLoad.addEventListener('click', () => queueFile.click());
      queueFile.addEventListener('change', e => {
        importPromptFile(e.target.files[0]);
        e.target.value = '';
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
