// ── Ratings: KTO/DPO preference-data capture ────────────────────
// 👍/👎 on assistant messages writes KTO rows; 👍×👎 within a sibling
// group auto-generates DPO pairs; ⚖ records explicit manual pairs.
// Rows land in <ratings_dir>/<dataset>.{kto,dpo}.jsonl, trainer-ready.

let ratingsDataset = 'chat';
const ratingsState = {
  kto: new Map(),           // node_id -> bool
  manualChosen: new Set(),  // node_ids chosen in a manual pair
  counts: {kto: 0, dpo: 0},
  dir: '',
};

function getRating(nodeId) { return ratingsState.kto.get(nodeId); }
function isManualChosen(nodeId) { return ratingsState.manualChosen.has(nodeId); }

function applyRatingsPayload(data) {
  ratingsState.kto = new Map(Object.entries(data.kto || {}));
  ratingsState.manualChosen = new Set(
    (data.dpo || []).filter(p => p.source === 'manual').map(p => p.chosen));
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

function siblingGroup(assistantNodeId) {
  // All assistant siblings (same user parent) with current labels.
  const node = tree.nodes.get(assistantNodeId);
  const parent = node ? tree.nodes.get(node.parentId) : null;
  if (!parent) return [];
  return parent.children
    .map(id => tree.nodes.get(id))
    .filter(n => n && n.role === 'assistant' && n.content.trim())
    .map(n => ({
      node_id: n.id,
      content: n.content,
      label: ratingsState.kto.has(n.id) ? ratingsState.kto.get(n.id) : null,
    }));
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

  // Update local state first so the sibling group carries the new label.
  if (newLabel === null) ratingsState.kto.delete(nodeId);
  else ratingsState.kto.set(nodeId, newLabel);

  await postRate({
    dataset: ratingsDataset,
    prompt,
    kto: {node_id: nodeId, completion: node.content, label: newLabel},
    group: siblingGroup(nodeId),
  });
}

async function preferNode(nodeId) {
  // ⚖ toggle: mark this sibling as preferred over every sibling not
  // rated 👍 (manual DPO pairs); click again to withdraw them.
  const node = tree.nodes.get(nodeId);
  if (!node || node.role !== 'assistant' || !node.content.trim()) return;
  const prompt = buildPromptTurns(nodeId);
  if (!prompt) return;

  const remove = ratingsState.manualChosen.has(nodeId);
  const rejected = siblingGroup(nodeId)
    .filter(g => g.node_id !== nodeId && g.label !== true)
    .map(({node_id, content}) => ({node_id, content}));
  if (!remove && rejected.length === 0) return;

  await postRate({
    dataset: ratingsDataset,
    prompt,
    manual: {
      chosen: {node_id: nodeId, content: node.content},
      rejected,
      remove,
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
    const nameInput = document.getElementById('ratings-dataset');
    const dirInput = document.getElementById('ratings-dir');
    nameInput.value = ratingsDataset;
    if (cfg.ratings_dir) dirInput.value = cfg.ratings_dir;

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

initRatings();
