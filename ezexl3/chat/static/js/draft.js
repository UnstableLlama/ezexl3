// ── Draft Model Panel: DFlash speculative decoding ───────────────

let draftModelDir = '';
let draftModelLoaded = false;
let draftModelName = '';

function initDraftPanel() {
  document.getElementById('draft-panel-toggle').onclick = toggleDraftPanel;
  document.getElementById('draft-load-btn').onclick = loadDraft;
  document.getElementById('draft-unload-btn').onclick = unloadDraft;
  document.getElementById('draft-dir-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); loadDraft(); }
  });
}

function toggleDraftPanel() {
  const body = document.getElementById('draft-panel-body');
  const chevron = document.querySelector('#draft-panel-toggle .chevron');
  const collapsed = body.style.display === 'none';
  body.style.display = collapsed ? '' : 'none';
  chevron.textContent = collapsed ? '▲' : '▼';
}

function showDraftPanel(show) {
  document.getElementById('draft-panel').style.display = show ? '' : 'none';
}

function syncDraftState(status) {
  draftModelLoaded = status.draft_model_loaded || false;
  draftModelDir = status.draft_model_dir || '';
  draftModelName = status.draft_model_name || '';

  if (draftModelDir) {
    document.getElementById('draft-dir-input').value = draftModelDir;
  }

  updateDraftBadge();
  updateDraftControls();
}

function updateDraftBadge() {
  const badge = document.getElementById('draft-panel-badge');
  if (draftModelLoaded) {
    badge.textContent = draftModelName || 'active';
    badge.className = 'panel-badge loaded';
  } else {
    badge.textContent = 'none';
    badge.className = 'panel-badge';
  }
}

function updateDraftControls() {
  const loadBtn = document.getElementById('draft-load-btn');
  const unloadBtn = document.getElementById('draft-unload-btn');
  const dirInput = document.getElementById('draft-dir-input');
  const info = document.getElementById('draft-info');

  if (draftModelLoaded) {
    loadBtn.textContent = 'Replace';
    unloadBtn.disabled = false;
    info.style.display = '';
    info.textContent = draftModelName;
  } else {
    loadBtn.textContent = 'Load';
    unloadBtn.disabled = true;
    info.style.display = 'none';
  }
  loadBtn.disabled = false;
  dirInput.disabled = false;
}

async function loadDraft() {
  const dirInput = document.getElementById('draft-dir-input');
  const dir = dirInput.value.trim();
  if (!dir) return;

  const loadBtn = document.getElementById('draft-load-btn');
  const statusEl = document.getElementById('draft-status');
  loadBtn.disabled = true;
  loadBtn.textContent = 'Loading...';
  statusEl.style.display = '';
  statusEl.textContent = 'Loading draft model...';

  try {
    const res = await fetch('/api/draft/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ draft_model_dir: dir }),
    });
    const data = await res.json();
    if (data.ok) {
      syncDraftState(data.status);
      updateDraftModelInfo(data.status);
      statusEl.textContent = 'Draft model loaded.';
      setTimeout(() => { statusEl.style.display = 'none'; }, 2000);
    } else {
      statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
      updateDraftControls();
    }
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
    updateDraftControls();
  }
}

async function unloadDraft() {
  const unloadBtn = document.getElementById('draft-unload-btn');
  const statusEl = document.getElementById('draft-status');
  unloadBtn.disabled = true;
  statusEl.style.display = '';
  statusEl.textContent = 'Unloading draft model...';

  try {
    const res = await fetch('/api/draft/unload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.ok) {
      syncDraftState(data.status);
      updateDraftModelInfo(data.status);
      statusEl.textContent = 'Draft model unloaded.';
      setTimeout(() => { statusEl.style.display = 'none'; }, 2000);
    } else {
      statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
      updateDraftControls();
    }
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
    updateDraftControls();
  }
}

function updateDraftModelInfo(status) {
  const loraInfo = status.lora_count
    ? `LoRAs: ${status.lora_count}<br>`
    : '';
  const draftInfo = status.draft_model_loaded
    ? `Draft: ${escHtml(status.draft_model_name)}<br>`
    : '';
  document.getElementById('model-info').innerHTML =
    `<strong>${status.model_name}</strong><br>` +
    `Context: ${(status.context_length || 0).toLocaleString()} tokens<br>` +
    loraInfo +
    draftInfo +
    `${status.model_dir || ''}`;
}
