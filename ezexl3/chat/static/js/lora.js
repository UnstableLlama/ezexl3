// ── LoRA Panel: dynamic add/remove/weight control ───────────────

let loraEntries = [];  // [{dir, weight}, ...]
let loraDirty = false; // true when UI state differs from server state

function initLoraPanel() {
  document.getElementById('lora-panel-toggle').onclick = toggleLoraPanel;
  document.getElementById('lora-add-btn').onclick = addLoraFromInput;
  document.getElementById('lora-apply-btn').onclick = applyLoras;
  document.getElementById('lora-add-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); addLoraFromInput(); }
  });
}

function toggleLoraPanel() {
  const body = document.getElementById('lora-panel-body');
  const chevron = document.querySelector('#lora-panel-toggle .chevron');
  const collapsed = body.style.display === 'none';
  body.style.display = collapsed ? '' : 'none';
  chevron.textContent = collapsed ? '▲' : '▼';
}

function showLoraPanel(show) {
  document.getElementById('lora-panel').style.display = show ? '' : 'none';
}

function syncLoraState(status) {
  const dirs = status.lora_dirs || [];
  const weights = status.lora_weights || [];
  loraEntries = dirs.map((d, i) => ({
    dir: d,
    weight: i < weights.length ? weights[i] : 1.0,
  }));
  loraDirty = false;
  renderLoraList();
  updateLoraBadge();
}

function renderLoraList() {
  const list = document.getElementById('lora-list');
  list.innerHTML = '';
  loraEntries.forEach((entry, i) => {
    const inactive = entry.weight <= 0;
    const el = document.createElement('div');
    el.className = 'lora-entry' + (inactive ? ' inactive' : '');

    const pctVal = Math.round(entry.weight * 100);
    const dirName = entry.dir.split('/').pop() || entry.dir;

    el.innerHTML =
      `<div class="lora-entry-header">` +
        `<span class="lora-entry-name" title="${escHtml(entry.dir)}">${escHtml(dirName)}</span>` +
        `<button class="lora-entry-remove" title="Remove" data-idx="${i}">&times;</button>` +
      `</div>` +
      `<div class="lora-weight-row">` +
        `<input type="range" min="0" max="200" step="1" value="${pctVal}" data-idx="${i}">` +
        `<span class="lora-weight-val" data-idx="${i}">${pctVal}%</span>` +
      `</div>`;

    list.appendChild(el);
  });

  list.querySelectorAll('.lora-entry-remove').forEach(btn => {
    btn.onclick = () => {
      const idx = parseInt(btn.dataset.idx);
      loraEntries.splice(idx, 1);
      loraDirty = true;
      renderLoraList();
      updateLoraBadge();
    };
  });

  list.querySelectorAll('input[type="range"]').forEach(slider => {
    const idx = parseInt(slider.dataset.idx);
    const valSpan = list.querySelector(`.lora-weight-val[data-idx="${idx}"]`);
    slider.oninput = () => {
      const pct = parseInt(slider.value);
      valSpan.textContent = pct + '%';
      loraEntries[idx].weight = pct / 100;
      loraDirty = true;
      updateLoraBadge();
      slider.closest('.lora-entry').classList.toggle('inactive', pct <= 0);
    };
    if (valSpan) {
      valSpan.onclick = () => {
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'lora-weight-val-input';
        input.value = slider.value;
        valSpan.replaceWith(input);
        input.focus();
        input.select();
        let committed = false;
        const commit = () => {
          if (committed) return;
          committed = true;
          let v = parseInt(input.value);
          if (isNaN(v)) v = parseInt(slider.value);
          v = Math.max(0, Math.min(200, v));
          slider.value = v;
          valSpan.textContent = v + '%';
          input.replaceWith(valSpan);
          loraEntries[idx].weight = v / 100;
          loraDirty = true;
          updateLoraBadge();
          slider.closest('.lora-entry').classList.toggle('inactive', v <= 0);
        };
        input.addEventListener('blur', commit);
        input.addEventListener('keydown', e => {
          if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
          if (e.key === 'Escape') { committed = true; valSpan.textContent = slider.value + '%'; input.replaceWith(valSpan); }
        });
      };
    }
  });
}

function addLoraFromInput() {
  const input = document.getElementById('lora-add-input');
  const dir = input.value.trim();
  if (!dir) return;
  loraEntries.push({ dir: dir, weight: 1.0 });
  input.value = '';
  loraDirty = true;
  renderLoraList();
  updateLoraBadge();
}

function updateLoraBadge() {
  const badge = document.getElementById('lora-panel-badge');
  const active = loraEntries.filter(e => e.weight > 0).length;
  const total = loraEntries.length;
  badge.textContent = active > 0 ? `${active} active` : (total > 0 ? '0 active' : 'none');
  badge.className = 'panel-badge' + (active > 0 ? ' loaded' : '');

  const applyBtn = document.getElementById('lora-apply-btn');
  if (loraDirty) {
    applyBtn.textContent = 'Apply LoRAs *';
    applyBtn.disabled = false;
  } else {
    applyBtn.textContent = 'Apply LoRAs';
    applyBtn.disabled = true;
  }
}

async function applyLoras() {
  const btn = document.getElementById('lora-apply-btn');
  const statusEl = document.getElementById('lora-status');
  btn.disabled = true;
  btn.textContent = 'Applying...';
  statusEl.style.display = '';
  statusEl.textContent = 'Updating LoRA adapters...';

  const payload = loraEntries.map(e => ({ dir: e.dir, weight: e.weight }));

  try {
    const res = await fetch('/api/loras/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ loras: payload }),
    });
    const data = await res.json();
    if (data.ok) {
      loraDirty = false;
      syncLoraState(data.status);
      statusEl.textContent = 'LoRAs applied successfully.';
      updateLoraModelInfo(data.status);
      setTimeout(() => { statusEl.style.display = 'none'; }, 2000);
    } else {
      statusEl.textContent = 'Error: ' + (data.error || 'Unknown error');
      btn.disabled = false;
      btn.textContent = 'Apply LoRAs *';
    }
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
    btn.disabled = false;
    btn.textContent = 'Apply LoRAs *';
  }
}

function updateLoraModelInfo(status) {
  const active = status.lora_count || 0;
  const loraInfo = active
    ? `LoRAs: ${active}<br>`
    : '';
  document.getElementById('model-info').innerHTML =
    `<strong>${status.model_name}</strong><br>` +
    `Context: ${(status.context_length || 0).toLocaleString()} tokens<br>` +
    loraInfo +
    `${status.model_dir || ''}`;
}
