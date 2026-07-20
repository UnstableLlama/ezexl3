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
// Pending DPO duel awaiting judgment:
//   {userNodeId, ids: [aId, bId], marks: {nodeId: 'up'|'down'|'fail'}}
let pendingDuel = null;
const ratingsState = {
  kto: new Map(),           // node_id -> bool
  pairs: [],                // [{chosen, rejected}] node-id pairs on disk
  counts: {kto: 0, dpo: 0},
  dir: '',
};

function getRating(nodeId) { return ratingsState.kto.get(nodeId); }
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
  // A duel can't outlive a mode switch — dismiss it without recording.
  if (ratingsMode !== 'dpo' && pendingDuel) skipDuel();
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
    if (a != null) turns.push({role: 'assistant', content: a});
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
    kto: {node_id: nodeId, completion: node.content, label: newLabel},
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
    await postRate({
      dataset: ratingsDataset,
      prompt,
      pair: {
        chosen: {node_id: chosenId, content: chosen.content},
        rejected: {node_id: rejectedId, content: rejected.content},
      },
    });
  }
  renderActiveTree();
  inputBox.focus();
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
