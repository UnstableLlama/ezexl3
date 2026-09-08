// ── Draft Model Panel: speculative decoding (DFlash draft dir, MTP, or n-gram) ──

let draftModelDir = '';
let draftModelLoaded = false;
let draftModelName = '';
let draftMtp = false;
let draftNgramMin = 0;

function initDraftPanel() {
  document.getElementById('draft-panel-toggle').onclick = toggleDraftPanel;
  document.getElementById('draft-load-btn').onclick = loadDraft;
  document.getElementById('draft-unload-btn').onclick = unloadDraft;
  document.getElementById('draft-dir-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); loadDraft(); }
  });
  document.getElementById('draft-mtp-checkbox').addEventListener('change', e => {
    if (e.target.checked) document.getElementById('draft-ngram-checkbox').checked = false;
    updateDraftInputs();
  });
  document.getElementById('draft-ngram-checkbox').addEventListener('change', e => {
    if (e.target.checked) document.getElementById('draft-mtp-checkbox').checked = false;
    updateDraftInputs();
  });
}

// Draft sources are mutually exclusive: a draft model dir, MTP, or n-gram
// drafting (checkboxes behave like radio buttons).
function updateDraftInputs() {
  const mtp = document.getElementById('draft-mtp-checkbox').checked;
  const ngram = document.getElementById('draft-ngram-checkbox').checked;
  document.getElementById('draft-dir-input').disabled = mtp || ngram;
  document.getElementById('draft-ngram-min').disabled = !ngram;
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

function draftActive() {
  return draftModelLoaded || draftNgramMin > 0;
}

function draftLabel() {
  if (draftNgramMin > 0) return `n-gram (min ${draftNgramMin})`;
  return draftModelName || 'active';
}

function syncDraftState(status) {
  draftModelLoaded = status.draft_model_loaded || false;
  draftModelDir = status.draft_model_dir || '';
  draftModelName = status.draft_model_name || '';
  draftMtp = status.draft_mtp || false;
  draftNgramMin = status.ngram_min || 0;

  if (draftModelDir && !draftMtp) {
    document.getElementById('draft-dir-input').value = draftModelDir;
  }
  document.getElementById('draft-mtp-checkbox').checked = draftMtp;
  document.getElementById('draft-ngram-checkbox').checked = draftNgramMin > 0;
  if (draftNgramMin > 0) {
    document.getElementById('draft-ngram-min').value = draftNgramMin;
  }
  updateDraftInputs();

  updateDraftBadge();
  updateDraftControls();
}

function updateDraftBadge() {
  const badge = document.getElementById('draft-panel-badge');
  if (draftActive()) {
    badge.textContent = draftLabel();
    badge.className = 'panel-badge loaded';
  } else {
    badge.textContent = 'none';
    badge.className = 'panel-badge';
  }
}

function updateDraftControls() {
  const loadBtn = document.getElementById('draft-load-btn');
  const unloadBtn = document.getElementById('draft-unload-btn');
  const info = document.getElementById('draft-info');

  if (draftActive()) {
    loadBtn.textContent = 'Replace';
    unloadBtn.disabled = false;
    info.style.display = '';
    info.textContent = draftLabel();
  } else {
    loadBtn.textContent = 'Load';
    unloadBtn.disabled = true;
    info.style.display = 'none';
  }
  loadBtn.disabled = false;
  updateDraftInputs();
}

async function loadDraft() {
  const dirInput = document.getElementById('draft-dir-input');
  const dir = dirInput.value.trim();
  const mtp = document.getElementById('draft-mtp-checkbox').checked;
  const ngram = document.getElementById('draft-ngram-checkbox').checked;
  const ngramMin = parseInt(document.getElementById('draft-ngram-min').value) || 3;
  if (!dir && !mtp && !ngram) return;

  const loadBtn = document.getElementById('draft-load-btn');
  const statusEl = document.getElementById('draft-status');
  loadBtn.disabled = true;
  loadBtn.textContent = 'Loading...';
  statusEl.style.display = '';
  statusEl.textContent =
    'Loading draft (recurrent models reload with the draft — may take a while)...';

  try {
    const res = await fetch('/api/draft/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        ngram ? { ngram_min: ngramMin }
          : mtp ? { mtp: true }
          : { draft_model_dir: dir }),
    });
    const data = await res.json();
    if (data.ok) {
      syncDraftState(data.status);
      updateDraftModelInfo(data.status);
      statusEl.textContent = data.reloaded
        ? 'Model reloaded with draft.' : 'Draft loaded.';
      setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
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
      statusEl.textContent = 'Draft unloaded.';
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
    : status.ngram_min
      ? `Draft: n-gram (min ${status.ngram_min})<br>`
      : '';
  document.getElementById('model-info').innerHTML =
    `<strong>${status.model_name}</strong><br>` +
    `Context: ${(status.context_length || 0).toLocaleString()} tokens<br>` +
    loraInfo +
    draftInfo +
    `${status.model_dir || ''}`;
}
