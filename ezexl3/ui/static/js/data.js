// ── Data tab: live measurement table + SVG graph ─────────────────

let dataPollingInterval = null;
let lastRowCount = 0;
let activeTab = "command";  // "command" | "data" | "evals"

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-content").forEach(p => {
    p.classList.remove("active");
  });

  if (tab === "command") {
    document.getElementById("form-panel").classList.add("active");
    document.getElementById("command-desc").style.display = "";
  } else if (tab === "data") {
    document.getElementById("data-panel").classList.add("active");
    document.getElementById("command-desc").style.display = "none";
    refreshData();
  } else if (tab === "evals") {
    document.getElementById("evals-tab-panel").classList.add("active");
    document.getElementById("command-desc").style.display = "none";
    refreshEvals();
  }
}


function getModelDir() {
  const el = document.getElementById("field-models");
  return el ? el.value.trim() : "";
}


async function refreshData() {
  const modelDir = getModelDir();
  if (!modelDir) {
    showDataEmpty("Enter a model directory and run a command to see measurements.");
    return;
  }
  await Promise.all([fetchTable(modelDir), fetchGraph(modelDir)]);
}


async function fetchTable(modelDir) {
  try {
    const res = await fetch(`/api/data?model_dir=${encodeURIComponent(modelDir)}`);
    const data = await res.json();
    renderTable(data.rows || []);
  } catch (e) {
    showDataEmpty("Failed to load measurement data.");
  }
}


function renderTable(rows) {
  const tbody = document.getElementById("data-table-body");
  const empty = document.getElementById("data-empty");
  const table = document.getElementById("data-table");

  if (!rows.length) {
    tbody.innerHTML = "";
    empty.style.display = "";
    table.style.display = "none";
    return;
  }

  empty.style.display = "none";
  table.style.display = "";

  tbody.innerHTML = rows.map(r => {
    const bpw = r.weights || "";
    const kl = r["KL Div"] || "";
    const ppl = r["PPL r-100"] || "";
    const gib = r["GiB"] || "";
    return `<tr>
      <td>${esc(bpw)}</td>
      <td>${kl ? esc(kl) : '<span class="data-pending">...</span>'}</td>
      <td>${ppl ? esc(ppl) : '<span class="data-pending">...</span>'}</td>
      <td>${gib ? esc(gib) : '<span class="data-pending">...</span>'}</td>
    </tr>`;
  }).join("");

  lastRowCount = rows.filter(r => r["KL Div"] && r["PPL r-100"]).length;
}


async function fetchGraph(modelDir) {
  const graphEl = document.getElementById("data-graph");
  const placeholder = document.getElementById("data-graph-placeholder");

  try {
    const res = await fetch(`/api/graph?model_dir=${encodeURIComponent(modelDir)}`);
    if (!res.ok) {
      graphEl.innerHTML = "";
      placeholder.style.display = "";
      return;
    }
    const svg = await res.text();
    placeholder.style.display = "none";
    graphEl.innerHTML = svg;
  } catch (e) {
    graphEl.innerHTML = "";
    placeholder.style.display = "";
  }
}


function showDataEmpty(msg) {
  document.getElementById("data-table-body").innerHTML = "";
  document.getElementById("data-empty").textContent = msg;
  document.getElementById("data-empty").style.display = "";
  document.getElementById("data-table").style.display = "none";
  document.getElementById("data-graph").innerHTML = "";
  document.getElementById("data-graph-placeholder").style.display = "";
}


function startDataPolling() {
  stopDataPolling();
  dataPollingInterval = setInterval(() => {
    if (activeTab === "data") {
      refreshData();
    } else if (activeTab === "evals") {
      refreshEvals();
    }
  }, 4000);
}


function stopDataPolling() {
  if (dataPollingInterval) {
    clearInterval(dataPollingInterval);
    dataPollingInterval = null;
  }
}


function initDataSplitter() {
  const splitter = document.getElementById("data-splitter");
  const tableWrap = document.getElementById("data-table-wrap");
  const content = document.getElementById("data-content");

  let startX = 0;
  let startWidth = 0;

  function onMouseDown(e) {
    e.preventDefault();
    startX = e.clientX;
    startWidth = tableWrap.getBoundingClientRect().width;
    splitter.classList.add("dragging");
    document.body.classList.add("data-splitter-active");
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }

  function onMouseMove(e) {
    const contentRect = content.getBoundingClientRect();
    const delta = e.clientX - startX;
    const newWidth = startWidth + delta;
    const maxWidth = contentRect.width - 180 - 6;
    const clamped = Math.max(180, Math.min(newWidth, maxWidth));
    tableWrap.style.flexBasis = clamped + "px";
  }

  function onMouseUp() {
    splitter.classList.remove("dragging");
    document.body.classList.remove("data-splitter-active");
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);
  }

  splitter.addEventListener("mousedown", onMouseDown);
}


function initEvalsSplitter() {
  const splitter = document.getElementById("evals-splitter");
  const tables = document.getElementById("evals-tables");
  const body = document.getElementById("evals-body");
  if (!splitter || !tables || !body) return;

  let startX = 0;
  let startWidth = 0;

  function onMouseDown(e) {
    e.preventDefault();
    startX = e.clientX;
    startWidth = tables.getBoundingClientRect().width;
    splitter.classList.add("dragging");
    document.body.classList.add("data-splitter-active");
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }

  function onMouseMove(e) {
    const bodyRect = body.getBoundingClientRect();
    const delta = e.clientX - startX;
    const newWidth = startWidth + delta;
    const maxWidth = bodyRect.width - 180 - 6;
    const clamped = Math.max(180, Math.min(newWidth, maxWidth));
    tables.style.flexBasis = clamped + "px";
  }

  function onMouseUp() {
    splitter.classList.remove("dragging");
    document.body.classList.remove("data-splitter-active");
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);
  }

  splitter.addEventListener("mousedown", onMouseDown);
}


function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}


// ── Evals tab: perf tables + catbench grid ───────────────────────

let evalsLastBpws = [];
let evalsLastCatbenchKey = "";

function getEvalsKind() {
  const el = document.getElementById("evals-kind-select");
  return el ? el.value : "perf";
}

function setEvalsViewMode(kind) {
  // Toggle visibility of perf-specific vs. catbench-specific elements.
  const bpwGroup = document.getElementById("evals-bpw-group");
  const perfBody = document.getElementById("evals-body");
  const empty = document.getElementById("evals-empty");
  const grid = document.getElementById("evals-catbench-grid");

  if (kind === "catbench") {
    if (bpwGroup) bpwGroup.style.display = "none";
    if (perfBody) perfBody.style.display = "none";
    if (empty) empty.style.display = "none";
    if (grid) grid.style.display = "";
  } else {
    if (bpwGroup) bpwGroup.style.display = "";
    if (grid) { grid.style.display = "none"; grid.innerHTML = ""; }
    evalsLastCatbenchKey = "";
  }
}

async function refreshEvals() {
  const modelDir = getModelDir();
  if (!modelDir) {
    showEvalsEmpty("Enter a model directory first.");
    return;
  }
  const kind = getEvalsKind();
  if (kind === "catbench") {
    await fetchCatbenchGrid(modelDir);
    return;
  }
  const select = document.getElementById("evals-bpw-select");
  const selectedBpw = select.value;
  // Fetch the list of BPWs (and optionally the selected BPW's data)
  await fetchEvalsData(modelDir, selectedBpw || null);
}


async function fetchCatbenchGrid(modelDir) {
  const grid = document.getElementById("evals-catbench-grid");
  try {
    const res = await fetch(
      `/api/catbench-file?model_dir=${encodeURIComponent(modelDir)}`
    );
    const json = await res.json();
    const items = json.items || [];

    if (!items.length) {
      // Only clobber innerHTML when we actually have to — avoids the
      // "flash" you'd otherwise get on every 4 s poll.
      if (evalsLastCatbenchKey !== "__empty__") {
        grid.innerHTML =
          '<div class="evals-empty">No catbench results yet. ' +
          'Run catbench (<code>-cb</code>) to begin.</div>';
        evalsLastCatbenchKey = "__empty__";
      }
      return;
    }

    // Skip re-render if the file list is unchanged. Each <img> already
    // has a stable src, so the browser won't re-fetch on repaint.
    const key = items.map(it => it.file).join("|");
    if (key === evalsLastCatbenchKey) return;
    evalsLastCatbenchKey = key;

    grid.innerHTML = items.map(it => {
      const src = `/api/catbench-file?model_dir=${encodeURIComponent(modelDir)}&file=${encodeURIComponent(it.file)}`;
      return (
        '<figure class="evals-catbench-tile">'
        + `  <div class="evals-catbench-img"><img src="${src}" alt="${esc(it.label)}"></div>`
        + `  <figcaption>${esc(it.label)}</figcaption>`
        + '</figure>'
      );
    }).join("");
  } catch (e) {
    if (evalsLastCatbenchKey !== "__error__") {
      grid.innerHTML = '<div class="evals-empty">Failed to load catbench grid.</div>';
      evalsLastCatbenchKey = "__error__";
    }
  }
}


async function fetchEvalsData(modelDir, bpw) {
  const select = document.getElementById("evals-bpw-select");
  try {
    let url = `/api/perf-data?model_dir=${encodeURIComponent(modelDir)}`;
    if (bpw) url += `&bpw=${encodeURIComponent(bpw)}`;
    const res = await fetch(url);
    const json = await res.json();

    // Update BPW dropdown if the list changed
    const newBpws = json.bpws || [];
    if (JSON.stringify(newBpws) !== JSON.stringify(evalsLastBpws)) {
      evalsLastBpws = newBpws;
      const prev = select.value;
      select.innerHTML = '<option value="">Select a BPW...</option>';
      for (const b of newBpws) {
        const opt = document.createElement("option");
        opt.value = b;
        opt.textContent = b;
        select.appendChild(opt);
      }
      // Restore selection if it still exists
      if (prev && newBpws.includes(prev)) {
        select.value = prev;
      }
    }

    if (!newBpws.length) {
      showEvalsEmpty("No performance data yet. Run a perf eval (-perf) to begin.");
      return;
    }

    // If no BPW selected yet, auto-select the first one
    if (!select.value && newBpws.length) {
      select.value = newBpws[0];
      // Re-fetch with the selected BPW
      await fetchEvalsData(modelDir, select.value);
      return;
    }

    const data = json.data || {};
    const bpwData = data[select.value];
    if (!bpwData) {
      showEvalsEmpty("No data for selected BPW.");
      return;
    }

    renderEvalsData(bpwData);
    fetchEvalsChart(modelDir, select.value);
  } catch (e) {
    showEvalsEmpty("Failed to load performance data.");
  }
}


async function fetchEvalsChart(modelDir, bpw) {
  const chartEl = document.getElementById("evals-chart");

  if (!bpw) {
    chartEl.innerHTML = "";
    return;
  }

  try {
    const url = `/api/perf-graph?model_dir=${encodeURIComponent(modelDir)}&bpw=${encodeURIComponent(bpw)}`;
    const res = await fetch(url);
    if (!res.ok) {
      chartEl.innerHTML = "";
      return;
    }
    chartEl.innerHTML = await res.text();
  } catch (e) {
    chartEl.innerHTML = "";
  }
}


function renderEvalsData(bpwData) {
  const tables = document.getElementById("evals-tables");
  const empty = document.getElementById("evals-empty");

  const prefill = bpwData.prefill || [];
  const gen = bpwData.generation || [];

  if (!prefill.length && !gen.length) {
    showEvalsEmpty("No data for selected BPW.");
    return;
  }

  empty.style.display = "none";
  document.getElementById("evals-body").style.display = "";

  const prefillBody = document.getElementById("evals-prefill-body");
  prefillBody.innerHTML = prefill.map(r =>
    `<tr><td>${esc(String(r.context_length))}</td><td>${esc(String(r.tokens_per_second))}</td></tr>`
  ).join("");

  const genBody = document.getElementById("evals-gen-body");
  genBody.innerHTML = gen.map(r =>
    `<tr><td>${esc(String(r.context_length))}</td><td>${esc(String(r.tokens_per_second))}</td></tr>`
  ).join("");
}


function showEvalsEmpty(msg) {
  document.getElementById("evals-body").style.display = "none";
  document.getElementById("evals-chart").innerHTML = "";
  const empty = document.getElementById("evals-empty");
  empty.textContent = msg;
  empty.style.display = "";
}


function initEvalsTab() {
  const select = document.getElementById("evals-bpw-select");
  if (select) {
    select.addEventListener("change", () => {
      if (activeTab === "evals") refreshEvals();
    });
  }
  const kindSelect = document.getElementById("evals-kind-select");
  if (kindSelect) {
    kindSelect.addEventListener("change", () => {
      setEvalsViewMode(kindSelect.value);
      if (activeTab === "evals") refreshEvals();
    });
    // Initialize show/hide based on current selection
    setEvalsViewMode(kindSelect.value);
  }
}
