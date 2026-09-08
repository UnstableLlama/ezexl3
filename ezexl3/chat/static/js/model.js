// ── Model Panel: file browser, GPU config, load/unload ──────────
// ── Dashboard switch ────────────────────────────────────────────

let currentBrowsePath = '';
let browseIsModel = false;
let gpuData = [];

function parseLoraDirs() {
  const raw = (document.getElementById('lora-dirs-input')?.value || '');
  return raw
    .split('\n')
    .map(s => s.trim())
    .filter(Boolean);
}

function updateDashboardButton() {
  const btn = document.getElementById('dashboard-btn');
  if (!btn) return;
  btn.classList.toggle('disabled', modelLoaded);
}

function flashDashboardWarning() {
  const el = document.getElementById('dashboard-flash');
  if (!el) return;
  el.classList.remove('show');
  void el.offsetWidth;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 1500);
}

async function launchDashboard() {
  if (modelLoaded) {
    flashDashboardWarning();
    return;
  }
  const btn = document.getElementById('dashboard-btn');
  btn.disabled = true;
  btn.classList.add('disabled');

  try {
    const res = await fetch('/api/ui/launch', { method: 'POST' });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      btn.disabled = false;
      btn.classList.remove('disabled');
      return;
    }
    const url = data.url || window.location.origin;
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#a0a0a0;font-size:14px">Launching dashboard...</div>';
    const tryRedirect = () => {
      fetch(url + '/api/gpus', { cache: 'no-store' }).then(r => {
        if (r.ok) window.location.replace(url + '/?t=' + Date.now());
        else setTimeout(tryRedirect, 500);
      }).catch(() => setTimeout(tryRedirect, 500));
    };
    setTimeout(tryRedirect, 1500);
  } catch (e) {
    btn.disabled = false;
    btn.classList.remove('disabled');
  }
}

// ── Init ────────────────────────────────────────────────────────

async function initModelPanel(status) {
  gpuData = status.gpus || [];
  renderGpuList();

  // Wire up events
  document.getElementById('model-panel-toggle').onclick = toggleModelPanel;
  document.getElementById('auto-gpu').onchange = onAutoGpuToggle;
  document.getElementById('load-btn').onclick = loadModel;
  document.getElementById('unload-btn').onclick = unloadModel;
  document.getElementById('dashboard-btn').onclick = launchDashboard;
  document.getElementById('browse-native-btn').onclick = pickDirectoryNative;
  updateDashboardButton();
  document.getElementById('browser-path-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      browseTo(e.target.value.trim());
    }
  });
  document.getElementById('use-mtp-checkbox').addEventListener('change', e => {
    if (e.target.checked) document.getElementById('use-ngram-checkbox').checked = false;
    updateLoadDraftInputs();
  });
  document.getElementById('use-ngram-checkbox').addEventListener('change', e => {
    if (e.target.checked) document.getElementById('use-mtp-checkbox').checked = false;
    updateLoadDraftInputs();
  });
  updateLoadDraftInputs();
  document.getElementById('cpu-offload-toggle').onclick = toggleCpuOffloadPanel;
  await initCpuOffload();

  if (status.loaded) {
    setModelPanelLoaded(status.model_name);
  } else {
    setModelPanelUnloaded();
    // Browse to last-used model directory, or fall back to home
    let startPath = '';
    try {
      const cfgRes = await fetch('/api/config');
      const cfg = await cfgRes.json();
      if (cfg.last_model_dir) startPath = cfg.last_model_dir;
    } catch (_) {}
    await browseTo(startPath);
  }
}

// Draft options in the load panel are mutually exclusive: a draft model
// dir, MTP, or n-gram drafting (checkboxes behave like radio buttons).
function updateLoadDraftInputs() {
  const mtp = document.getElementById('use-mtp-checkbox').checked;
  const ngram = document.getElementById('use-ngram-checkbox').checked;
  document.getElementById('draft-model-dir-input').disabled = mtp || ngram;
  document.getElementById('use-ngram-min').disabled = !ngram;
}

// ── CPU offload ─────────────────────────────────────────────────

// id -> key in the cpu_offload block sent to /api/model/load.
const CPU_OFFLOAD_FIELDS = {
  'cpu-moe-layers': 'moe_layers',
  'cpu-moe-threads': 'moe_threads',
  'cpu-cache-gb': 'cache_gb',
  'cpu-draft-moe-layers': 'draft_moe_layers',
  'cpu-draft-moe-threads': 'draft_moe_threads',
};

function toggleCpuOffloadPanel() {
  const body = document.getElementById('cpu-offload-body');
  const chevron = document.querySelector('#cpu-offload-toggle .chevron');
  const collapsed = body.style.display === 'none';
  body.style.display = collapsed ? '' : 'none';
  chevron.textContent = collapsed ? '▲' : '▼';
}

// Restore saved values and grey the panel out if the running exllamav3
// predates CPU offload — better than accepting numbers we'd silently drop.
async function initCpuOffload() {
  let support = {};
  try {
    const res = await fetch('/api/gpus', { cache: 'no-store' });
    const data = await res.json();
    support = data.cpu_offload || {};
    // -ngr (n-gram table in RAM) shares the same support probe round-trip
    const ngramRamEl = document.getElementById('ngram-ram-checkbox');
    if (ngramRamEl && !data.ngram_ram) {
      ngramRamEl.disabled = true;
      document.getElementById('ngram-ram-unsupported').style.display = '';
    }
    const cores = data.cpu_cores || 0;
    if (cores) {
      document.getElementById('cpu-moe-threads-hint').textContent =
        `0 = auto (half of ${cores} cores).`;
    }
  } catch (_) {}

  try {
    const cfg = await (await fetch('/api/config')).json();
    const saved = cfg.cpu_offload || {};
    for (const [id, key] of Object.entries(CPU_OFFLOAD_FIELDS)) {
      if (saved[key] != null) document.getElementById(id).value = saved[key];
    }
  } catch (_) {}

  const supported = !!(support.moe || support.cache);
  document.getElementById('cpu-offload-unsupported').style.display =
    supported ? 'none' : '';
  document.getElementById('cpu-offload-body').classList.toggle('disabled', !supported);
  for (const id of Object.keys(CPU_OFFLOAD_FIELDS)) {
    document.getElementById(id).disabled = !supported;
  }
  // The cache knob shipped alongside the MoE knobs, but gate it separately
  // in case a build ever carries only one of them.
  if (supported && !support.cache) document.getElementById('cpu-cache-gb').disabled = true;
  if (supported && !support.moe) {
    for (const id of ['cpu-moe-layers', 'cpu-moe-threads',
                      'cpu-draft-moe-layers', 'cpu-draft-moe-threads']) {
      document.getElementById(id).disabled = true;
    }
  }

  // Expand automatically when something is actually configured, so an
  // active offload setting isn't hidden behind a collapsed header.
  if (Object.values(getCpuOffload()).some(v => v > 0)) toggleCpuOffloadPanel();
}

function getCpuOffload() {
  const out = {};
  for (const [id, key] of Object.entries(CPU_OFFLOAD_FIELDS)) {
    const el = document.getElementById(id);
    const val = el && !el.disabled ? parseFloat(el.value) : 0;
    out[key] = Number.isFinite(val) && val > 0 ? val : 0;
  }
  return out;
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
  updateLoadButton();  // reflect actual browse state, don't hardcode disabled
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
      browseIsModel = false;
      updateLoadButton();
      return;
    }
    currentBrowsePath = data.current;
    browseIsModel = data.is_model;
    renderBreadcrumb(data.current);
    renderBrowserEntries(data);
    updateLoadButton();
    if (data.is_model) saveModelDir(data.current);
  } catch (e) {
    list.innerHTML = `<div class="browser-error">Failed to browse</div>`;
    browseIsModel = false;
    updateLoadButton();
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
  updateOffloadEligibility(data.is_model ? data.moe_offload : null);
}

// exllamav3 only offloads mul1-codebook experts; anything else falls back to
// the GPU with a note on the server console, which is indistinguishable from
// the setting being ignored. Say so next to the control instead.
function updateOffloadEligibility(info) {
  const el = document.getElementById('cpu-offload-ineligible');
  if (!el) return;
  if (!info || !info.moe) {
    // Not a MoE model (or unknown): the expert knobs simply don't apply, and
    // the CPU KV cache still does, so this isn't worth a warning.
    el.style.display = 'none';
    return;
  }
  if (info.mul1) {
    el.style.display = '';
    el.classList.remove('warn');
    el.textContent = '✓ MoE experts are mul1 — eligible for CPU offload.';
  } else {
    el.style.display = '';
    el.classList.add('warn');
    el.textContent = 'This model’s experts are not mul1-codebook, so expert '
      + 'offload will be skipped and the layers stay on GPU. The CPU KV cache '
      + 'still works.';
  }
}

async function pickDirectoryNative() {
  const initial = currentBrowsePath || '';
  try {
    const url = '/api/pick_directory' + (initial ? '?initial=' + encodeURIComponent(initial) : '');
    const res = await fetch(url);
    const data = await res.json();
    if (data.path) {
      await browseTo(data.path);
      return;
    }
  } catch (_) {}
  // Native dialog unavailable or cancelled — focus the path input as fallback.
  document.getElementById('browser-path-input').focus();
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
  const seqLength = parseInt(document.getElementById('model-seq-length').value) || null;
  const cacheSize = seqLength ? seqLength * 2 : null;
  const cacheQuant = document.getElementById('model-cache-quant').value.trim() || null;
  const useMtp = document.getElementById('use-mtp-checkbox')?.checked || false;
  const useNgram = document.getElementById('use-ngram-checkbox')?.checked || false;
  const ngramMin = useNgram
    ? (parseInt(document.getElementById('use-ngram-min')?.value) || 3) : 0;
  const cpuOffload = getCpuOffload();

  try {
    const res = await fetch('/api/model/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_dir: currentBrowsePath,
        lora_dirs: parseLoraDirs(),
        draft_model_dir: (useMtp || useNgram) ? null
          : (document.getElementById('draft-model-dir-input')?.value || '').trim() || null,
        use_mtp: useMtp,
        ngram_min: ngramMin,
        devices: gpuConfig.devices,
        device_ratios: gpuConfig.device_ratios,
        cache_size: cacheSize,
        cache_quant: cacheQuant,
        cpu_offload: cpuOffload,
        ngram_ram: document.getElementById('ngram-ram-checkbox')?.checked || false,
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
      updateDashboardButton();
      showLoraPanel(true);
      syncLoraState(data.status);
      showDraftPanel(true);
      syncDraftState(data.status);
      saveCpuOffload(cpuOffload);
    } else {
      loadingEl.textContent = 'Error: ' + (data.error || 'Unknown error');
      loadBtn.disabled = !browseIsModel;
    }
  } catch (e) {
    loadingEl.textContent = 'Error: ' + e.message;
    loadBtn.disabled = !browseIsModel;
  }
}

function saveCpuOffload(cpuOffload) {
  fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cpu_offload: cpuOffload }),
  }).catch(() => {});
}

function saveModelDir(dir) {
  fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ last_model_dir: dir }),
  }).catch(() => {});
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
  updateDashboardButton();
  showLoraPanel(false);
  showDraftPanel(false);
  document.getElementById('header-model').textContent = '';
  document.getElementById('model-info').innerHTML = '<em>No model loaded</em>';
  await browseTo(currentBrowsePath || '');
}
