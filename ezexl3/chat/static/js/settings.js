// ── Settings: Sidebar sync, sliders, toggles, banned strings ────

function populateUI(status) {
  // Mode dropdown
  const sel = document.getElementById('s-mode');
  sel.innerHTML = '';
  for (const [k, desc] of Object.entries(modes)) {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = `${k} — ${desc}`;
    sel.appendChild(opt);
  }
  sel.value = settings.mode || 'chatml';

  // System prompt
  document.getElementById('s-system').value = settings.system_prompt || '';

  // Sliders
  setSlider('s-temp',  'v-temp',  settings.temperature, v => v.toFixed(2));
  setSlider('s-topk',  'v-topk',  settings.top_k,       v => String(Math.round(v)));
  setSlider('s-topp',  'v-topp',  settings.top_p,        v => v.toFixed(2));
  setSlider('s-minp',  'v-minp',  settings.min_p,        v => v.toFixed(3));
  setSlider('s-rep',   'v-rep',   settings.repetition_penalty, v => v.toFixed(2));
  setSlider('s-maxtok','v-maxtok',settings.max_response_tokens, v => String(Math.round(v)));

  // Toggles
  setToggle('t-think',   settings.think);
  setToggle('t-nothink', settings.no_think);
  setToggle('t-amnesia', settings.amnesia);

  // Think budget
  const tb = document.getElementById('s-thinkbudget');
  tb.value = settings.think_budget != null ? settings.think_budget : '';

  // Banned strings
  renderBans();

  // Model info
  if (status.loaded) {
    document.getElementById('model-info').innerHTML =
      `<strong>${status.model_name}</strong><br>` +
      `Context: ${(status.context_length || 0).toLocaleString()} tokens<br>` +
      `${status.model_dir || ''}`;
  } else {
    document.getElementById('model-info').innerHTML = '<em>No model loaded</em>';
  }
  document.getElementById('header-model').textContent = status.model_name || '';
}

// ── Slider helpers ──────────────────────────────────────────────
function setSlider(sliderId, valId, value, fmt) {
  const sl = document.getElementById(sliderId);
  const vl = document.getElementById(valId);
  sl.value = value;
  vl.textContent = fmt(Number(value));
  sl.oninput = () => { vl.textContent = fmt(Number(sl.value)); };
  sl.onchange = () => syncSettings();

  // Click on value to enter directly
  vl.onclick = () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'slider-val-input';
    input.value = sl.value;
    vl.replaceWith(input);
    input.focus();
    input.select();

    let committed = false;
    const commit = () => {
      if (committed) return;
      committed = true;
      let v = parseFloat(input.value);
      if (isNaN(v)) v = Number(sl.value);
      v = Math.max(Number(sl.min), Math.min(Number(sl.max), v));
      sl.value = v;
      vl.textContent = fmt(v);
      input.replaceWith(vl);
      syncSettings();
    };
    const cancel = () => {
      if (committed) return;
      committed = true;
      vl.textContent = fmt(Number(sl.value));
      input.replaceWith(vl);
    };
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
      if (e.key === 'Escape') { e.preventDefault(); cancel(); }
    });
  };
}

// ── Toggle helpers ──────────────────────────────────────────────
function setToggle(id, value) {
  const el = document.getElementById(id);
  el.classList.toggle('on', !!value);
  el.onclick = () => {
    el.classList.toggle('on');
    syncSettings();
  };
}

// ── Sync settings to server ─────────────────────────────────────
let syncTimer = null;
function syncSettings() {
  clearTimeout(syncTimer);
  syncTimer = setTimeout(async () => {
    const s = {
      mode: document.getElementById('s-mode').value,
      system_prompt: document.getElementById('s-system').value,
      temperature: parseFloat(document.getElementById('s-temp').value),
      top_k: parseInt(document.getElementById('s-topk').value),
      top_p: parseFloat(document.getElementById('s-topp').value),
      min_p: parseFloat(document.getElementById('s-minp').value),
      repetition_penalty: parseFloat(document.getElementById('s-rep').value),
      max_response_tokens: parseInt(document.getElementById('s-maxtok').value),
      think: document.getElementById('t-think').classList.contains('on'),
      no_think: document.getElementById('t-nothink').classList.contains('on'),
      amnesia: document.getElementById('t-amnesia').classList.contains('on'),
      banned_strings: settings.banned_strings || [],
    };
    const tb = document.getElementById('s-thinkbudget').value;
    s.think_budget = tb ? parseInt(tb) : null;
    settings = s;
    await fetch('/api/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(s),
    });
  }, 300);
}

// ── Banned strings ──────────────────────────────────────────────
function renderBans() {
  const el = document.getElementById('banned-chips');
  el.innerHTML = '';
  (settings.banned_strings || []).forEach((s, i) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.innerHTML = `<span>${escHtml(s)}</span><span class="x" onclick="removeBan(${i})">&times;</span>`;
    el.appendChild(chip);
  });
}
function addBan() {
  const inp = document.getElementById('banned-input');
  const v = inp.value.trim();
  if (!v) return;
  if (!settings.banned_strings) settings.banned_strings = [];
  settings.banned_strings.push(v);
  inp.value = '';
  renderBans();
  syncSettings();
}
function removeBan(i) {
  settings.banned_strings.splice(i, 1);
  renderBans();
  syncSettings();
}
