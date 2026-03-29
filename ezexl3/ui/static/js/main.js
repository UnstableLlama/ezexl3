// ── Initialization, navigation, event wiring ────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  // Fetch templates, GPU info, and saved config
  try {
    const [tRes, gRes, cRes] = await Promise.all([
      fetch("/api/templates"),
      fetch("/api/gpus"),
      fetch("/api/config"),
    ]);
    const tData = await tRes.json();
    const gData = await gRes.json();
    const cData = await cRes.json();
    templates = tData.templates || [];
    gpus = gData.gpus || [];
    savedConfig = cData || {};
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
        launchChat();
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

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Data panel horizontal splitter
  initDataSplitter();

  // Splitter drag
  initSplitter();

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
  document.getElementById("tab-command-label").textContent = schema.label;
  document.getElementById("command-desc").textContent = schema.description;

  // Switch back to command tab when changing commands
  switchTab("command");

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


function flashNavWarning(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("show");
  void el.offsetWidth; // force reflow to restart animation
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1500);
}

async function launchChat() {
  const chatBtn = document.querySelector('.nav-btn[data-cmd="chat"]');
  if (jobRunning) {
    flashNavWarning("chat-flash");
    return;
  }
  chatBtn.disabled = true;
  chatBtn.classList.add("disabled");

  try {
    const res = await fetch("/api/chat/launch", { method: "POST" });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      chatBtn.disabled = false;
      chatBtn.classList.remove("disabled");
      return;
    }
    // Dashboard is shutting down, redirect to the chat UI on same port
    const url = data.url || window.location.origin;
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#a0a0a0;font-size:14px">Launching chat UI...</div>';
    // Wait for chat server to come up, then redirect with cache bust
    const tryRedirect = () => {
      fetch(url + "/api/status", { cache: "no-store" }).then(r => {
        if (r.ok) window.location.replace(url + "/?t=" + Date.now());
        else setTimeout(tryRedirect, 500);
      }).catch(() => setTimeout(tryRedirect, 500));
    };
    setTimeout(tryRedirect, 1500);
  } catch (e) {
    chatBtn.disabled = false;
    chatBtn.classList.remove("disabled");
  }
}


function updateChatButton() {
  const chatBtn = document.querySelector('.nav-btn[data-cmd="chat"]');
  if (!chatBtn) return;
  chatBtn.disabled = jobRunning;
  chatBtn.classList.toggle("disabled", jobRunning);
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


function initSplitter() {
  const splitter = document.getElementById("splitter");
  const content = document.getElementById("content");
  const formPanel = document.getElementById("upper-panel");

  let startY = 0;
  let startHeight = 0;

  function onMouseDown(e) {
    e.preventDefault();
    startY = e.clientY;
    startHeight = formPanel.getBoundingClientRect().height;
    splitter.classList.add("dragging");
    document.body.classList.add("splitter-active");
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }

  function onMouseMove(e) {
    const contentRect = content.getBoundingClientRect();
    const delta = e.clientY - startY;
    const newHeight = startHeight + delta;
    // Clamp: min 80px for form, leave at least 80px for terminal + splitter
    const maxHeight = contentRect.height - 80 - 6;
    const clamped = Math.max(80, Math.min(newHeight, maxHeight));
    formPanel.style.height = clamped + "px";
  }

  function onMouseUp() {
    splitter.classList.remove("dragging");
    document.body.classList.remove("splitter-active");
    document.removeEventListener("mousemove", onMouseMove);
    document.removeEventListener("mouseup", onMouseUp);
  }

  splitter.addEventListener("mousedown", onMouseDown);
}
