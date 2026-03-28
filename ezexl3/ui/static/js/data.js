// ── Data tab: live measurement table + SVG graph ─────────────────

let dataPollingInterval = null;
let lastRowCount = 0;
let activeTab = "command";  // "command" | "data"

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
    }
  }, 4000);
}


function stopDataPolling() {
  if (dataPollingInterval) {
    clearInterval(dataPollingInterval);
    dataPollingInterval = null;
  }
}


function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}
