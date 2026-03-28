// ── Initialization, navigation, event wiring ────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  // Fetch templates and GPU info
  try {
    const [tRes, gRes] = await Promise.all([
      fetch("/api/templates"),
      fetch("/api/gpus"),
    ]);
    const tData = await tRes.json();
    const gData = await gRes.json();
    templates = tData.templates || [];
    gpus = gData.gpus || [];
  } catch (e) {
    console.error("Failed to fetch initial data:", e);
  }

  // Render GPU info
  renderGpuInfo();

  // Set up command navigation
  const navBtns = document.querySelectorAll(".nav-btn");
  for (const btn of navBtns) {
    btn.addEventListener("click", () => {
      const cmd = btn.dataset.cmd;
      if (cmd === "chat") {
        // Open chat in new tab on port 8800
        window.open("http://127.0.0.1:8800", "_blank");
        return;
      }
      selectCommand(cmd);
    });
  }

  // Run / Stop / Clear buttons
  document.getElementById("run-btn").addEventListener("click", runCommand);
  document.getElementById("stop-btn").addEventListener("click", stopJob);
  document.getElementById("clear-btn").addEventListener("click", clearTerminal);

  // Browser modal
  document.getElementById("browser-close-btn").addEventListener("click", closeBrowser);
  document.getElementById("browser-select-btn").addEventListener("click", selectBrowserPath);
  document.getElementById("browser-path-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      browseTo(e.target.value.trim());
    }
  });

  // Sidebar toggle
  document.getElementById("toggle-sidebar").addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("collapsed");
  });

  // Select initial command
  selectCommand("repo");

  // Check for running job
  checkRunningJob();
});


function selectCommand(cmd) {
  activeCommand = cmd;

  // Update nav highlight
  document.querySelectorAll(".nav-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.cmd === cmd);
  });

  // Update header
  const schema = COMMANDS[cmd];
  document.getElementById("command-title").textContent = schema.label;
  document.getElementById("command-desc").textContent = schema.description;

  // Render form
  renderForm(cmd);
}


function renderGpuInfo() {
  const el = document.getElementById("gpu-info");
  if (!gpus.length) {
    el.innerHTML = '<span class="text-dim">No GPUs detected</span>';
    return;
  }
  el.innerHTML = gpus.map(g =>
    `<div class="gpu-item">GPU ${g.index}: ${g.name} <span class="text-dim">${g.vram_gb} GB</span></div>`
  ).join("");
}


async function checkRunningJob() {
  try {
    const res = await fetch("/api/run/status");
    const data = await res.json();
    if (data.status === "running" && data.job_id) {
      activeJobId = data.job_id;
      jobRunning = true;
      updateRunButton();
      appendTerminal(`Reconnecting to running job...\n`, "term-cmd");
      await streamJob(data.job_id);
    }
  } catch (e) {
    // ignore
  }
}
