// ── Model Panel: file browser, GPU config, load/unload ──────────

let currentBrowsePath = '';
let browseIsModel = false;
let gpuData = [];

// ── Init ────────────────────────────────────────────────────────

async function initModelPanel(status) {
  gpuData = status.gpus || [];
  renderGpuList();

  // Wire up events
  document.getElementById('model-panel-toggle').onclick = toggleModelPanel;
  document.getElementById('auto-gpu').onchange = onAutoGpuToggle;
  document.getElementById('load-btn').onclick = loadModel;
  document.getElementById('unload-btn').onclick = unloadModel;
  document.getElementById('browser-path-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      browseTo(e.target.value.trim());
    }
  });

  if (status.loaded) {
    setModelPanelLoaded(status.model_name);
  } else {
    setModelPanelUnloaded();
    // Start browsing at home directory
    await browseTo('');
  }
}

// ── Panel toggle ────────────────────────────────────────────────

function toggleModelPanel() {
  const body = document.getElementById('model-panel-body');
  const chevron = document.querySelector('#model-panel-toggle .chevron');
  const collapsed = body.style.display === 'none';
  body.style.display = collapsed ? '' : 'none';
  chevron.textContent = collapsed ? '\u25B2' : '\u25BC';
}

function expandModelPanel() {
  document.getElementById('model-panel-body').style.display = '';
  document.querySelector('#model-panel-toggle .chevron').textContent = '\u25B2';
}

function collapseModelPanel() {
  document.getElementById('model-panel-body').style.display = 'none';
  document.querySelector('#model-panel-toggle .chevron').textContent = '\u25BC';
}

// ── Panel state ─────────────────────────────────────────────────

function setModelPanelLoaded(name) {
  const badge = document.getElementById('model-panel-status');
  badge.textContent = name;
  badge.className = 'panel-badge loaded';
  document.getElementById('load-btn').disabled = true;
  document.getElementById('unload-btn').disabled = false;
  document.getElementById('model-loading').style.display = 'none';
  collapseModelPanel();
}

function setModelPanelUnloaded() {
  const badge = document.getElementById('model-panel-status');
  badge.textContent = 'No model';
  badge.className = 'panel-badge';
  document.getElementById('load-btn').disabled = true;
  document.getElementById('unload-btn').disabled = true;
  document.getElementById('model-loading').style.display = 'none';
  expandModelPanel();
}

// ── File browser ────────────────────────────────────────────────

async function browseTo(path) {
  const list = document.getElementById('browser-list');
  list.innerHTML = '<div class="browser-loading">Loading...</div>';
  try {
    const url = '/api/browse' + (path ? '?path=' + encodeURIComponent(path) : '');
    const res = await fetch(url);
    const data = await res.json();
    if (data.error) {
      list.innerHTML = `<div class="browser-error">${escHtml(data.error)}</div>`;
      return;
    }
    currentBrowsePath = data.current;
    browseIsModel = data.is_model;
    renderBreadcrumb(data.current);
    renderBrowserEntries(data);
    updateLoadButton();
  } catch (e) {
    list.innerHTML = `<div class="browser-error">Failed to browse</div>`;
  }
}

function renderBreadcrumb(fullPath) {
  const input = document.getElementById('browser-path-input');
  input.value = fullPath;
}

function renderBrowserEntries(data) {
  const list = document.getElementById('browser-list');
  list.innerHTML = '';

  // Parent directory entry
  if (data.parent) {
    const el = document.createElement('div');
    el.className = 'browser-entry browser-dir';
    el.innerHTML = '<span class="browser-icon">\u{1F4C1}</span> <span class="browser-name">..</span>';
    el.onclick = () => browseTo(data.parent);
    list.appendChild(el);
  }

  for (const entry of data.entries) {
    const el = document.createElement('div');
    if (entry.type === 'dir') {
      el.className = 'browser-entry browser-dir';
      el.innerHTML = `<span class="browser-icon">\u{1F4C1}</span> <span class="browser-name">${escHtml(entry.name)}</span>`;
      el.onclick = () => browseTo(currentBrowsePath + '/' + entry.name);
    } else {
      el.className = 'browser-entry browser-file';
      el.innerHTML = `<span class="browser-icon">\u{1F4C4}</span> <span class="browser-name">${escHtml(entry.name)}</span>`;
    }
    list.appendChild(el);
  }

  // Model indicator
  const indicator = document.getElementById('browser-model-indicator');
  if (data.is_model) {
    indicator.style.display = '';
    indicator.textContent = '\u2713 Valid model directory';
  } else {
    indicator.style.display = 'none';
  }
}

function updateLoadButton() {
  const btn = document.getElementById('load-btn');
  btn.disabled = !browseIsModel;
}

// ── GPU configuration ───────────────────────────────────────────

function renderGpuList() {
  const list = document.getElementById('gpu-list');
  list.innerHTML = '';
  if (gpuData.length === 0) {
    list.innerHTML = '<div class="gpu-none">No GPUs detected</div>';
    return;
  }
  for (const gpu of gpuData) {
    const row = document.createElement('div');
    row.className = 'gpu-row';
    row.innerHTML =
      `<label class="gpu-label">` +
        `<input type="checkbox" class="gpu-checkbox" data-index="${gpu.index}" checked>` +
        `<span class="gpu-name">${escHtml(gpu.name)}</span>` +
        `<span class="gpu-vram">${gpu.vram_gb} GB</span>` +
      `</label>` +
      `<input type="text" class="gpu-ratio" data-index="${gpu.index}" placeholder="auto" title="VRAM ratio (GB)">`;
    list.appendChild(row);
  }
}

function onAutoGpuToggle() {
  const auto = document.getElementById('auto-gpu').checked;
  document.getElementById('gpu-manual').style.display = auto ? 'none' : '';
}

function getGpuConfig() {
  const auto = document.getElementById('auto-gpu').checked;
  if (auto) {
    return { devices: null, device_ratios: null };
  }
  const devices = [];
  const ratios = [];
  let hasRatios = false;
  for (const cb of document.querySelectorAll('.gpu-checkbox:checked')) {
    const idx = parseInt(cb.dataset.index);
    devices.push(idx);
    const ratioInput = document.querySelector(`.gpu-ratio[data-index="${idx}"]`);
    const r = ratioInput ? ratioInput.value.trim() : '';
    ratios.push(r);
    if (r) hasRatios = true;
  }
  if (devices.length === 0) return { devices: null, device_ratios: null };
  return {
    devices: devices,
    device_ratios: hasRatios ? ratios.join(',') : null,
  };
}

// ── Load / Unload ───────────────────────────────────────────────

async function loadModel() {
  if (!browseIsModel || !currentBrowsePath) return;

  const loadingEl = document.getElementById('model-loading');
  const loadBtn = document.getElementById('load-btn');
  loadBtn.disabled = true;
  loadingEl.style.display = '';
  loadingEl.textContent = 'Loading model...';

  const gpuConfig = getGpuConfig();
  const cacheSize = parseInt(document.getElementById('model-cache-size').value) || null;
  const cacheQuant = document.getElementById('model-cache-quant').value.trim() || null;

  try {
    const res = await fetch('/api/model/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_dir: currentBrowsePath,
        devices: gpuConfig.devices,
        device_ratios: gpuConfig.device_ratios,
        cache_size: cacheSize,
        cache_quant: cacheQuant,
      }),
    });
    const data = await res.json();
    if (data.ok) {
      modelLoaded = true;
      settings = data.settings;
      modes = data.status.available_modes || {};
      populateUI(data.status);
      setModelPanelLoaded(data.status.model_name);
      updateChatEnabled();
    } else {
      loadingEl.textContent = 'Error: ' + (data.error || 'Unknown error');
      loadBtn.disabled = !browseIsModel;
    }
  } catch (e) {
    loadingEl.textContent = 'Error: ' + e.message;
    loadBtn.disabled = !browseIsModel;
  }
}

async function unloadModel() {
  try {
    await fetch('/api/model/unload', { method: 'POST' });
  } catch (e) {
    // ignore
  }
  modelLoaded = false;
  setModelPanelUnloaded();
  updateChatEnabled();
  document.getElementById('header-model').textContent = '';
  document.getElementById('model-info').innerHTML = '<em>No model loaded</em>';
  await browseTo(currentBrowsePath || '');
}
