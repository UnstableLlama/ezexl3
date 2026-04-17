// ── Terminal output panel + SSE streaming ────────────────────────

const terminalEl = () => document.getElementById("terminal-output");
const terminalStatus = () => document.getElementById("terminal-status");

let abortController = null;

function clearTerminal() {
  terminalEl().textContent = "";
  terminalStatus().textContent = "";
  terminalStatus().className = "terminal-status";
}

function appendTerminal(text, className) {
  const el = terminalEl();

  // Handle \r (carriage return): replace a progress line
  if (text.includes("\r")) {
    const parts = text.split("\r");
    // Anything before first \r appends normally
    if (parts[0]) _appendChunk(el, parts[0], className);
    // Last part after final \r updates a progress line
    const replacement = parts[parts.length - 1];
    _updateProgressLine(el, replacement);
  } else {
    // Regular text — finalize all active progress lines first
    _finalizeAllProgress(el);
    _appendChunk(el, text, className);
  }
  // Auto-scroll
  el.scrollTop = el.scrollHeight;
}

function _appendChunk(el, text, className) {
  if (className) {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    el.appendChild(span);
  } else {
    el.appendChild(document.createTextNode(text));
  }
}

function _updateProgressLine(el, text) {
  // Extract key prefix to support per-item progress lines
  // Matches: "gpu0:", "filename.ext:", "...filename.ext:" (tqdm upload bars)
  const cleaned = text.replace(/\n$/, "");
  const m = cleaned.match(/^(\s*(?:\.{3})?\S+?:)\s*/);
  const key = m ? m[1].trim() : "_default";
  const display = cleaned;

  // Find or create the ephemeral progress element for this key
  let prog = el.querySelector(`.term-progress[data-key="${CSS.escape(key)}"]`);
  if (!prog) {
    prog = document.createElement("span");
    prog.className = "term-progress";
    prog.dataset.key = key;
    el.appendChild(prog);
  }
  prog.textContent = display;
}

function _finalizeAllProgress(el) {
  // Convert all active progress spans into permanent text, preserving
  // the last value each one was showing. This is what keeps the final
  // perf result ("prefill @32768: 1057 t/s") and the like visible in
  // scrollback once a normal log line arrives — replacing with a bare
  // "\n" (the previous behavior) erased the value and showed a blank
  // line instead.
  const progs = el.querySelectorAll(".term-progress");
  if (progs.length) {
    for (const prog of progs) {
      const finalText = prog.textContent + "\n";
      prog.replaceWith(document.createTextNode(finalText));
    }
  }
}

async function runUploadAction(action) {
  if (jobRunning) return;

  // Metadata gate: require MODEL + USER to be locked and non-empty before
  // we do anything — these drive the repo namespace and naming.
  const cmd = COMMANDS.upload;
  if (typeof allVisibleMetaFieldsReady === "function" && !allVisibleMetaFieldsReady(cmd)) {
    clearTerminal();
    appendTerminal("Lock the Model Name and Quantized By fields in the metadata panel before uploading.\n", "term-stderr");
    terminalStatus().textContent = "Metadata not locked";
    terminalStatus().className = "terminal-status error";
    return;
  }

  const args = collectArgs();
  if (args.error) {
    terminalStatus().textContent = args.error;
    terminalStatus().className = "terminal-status error";
    return;
  }

  const isDryRun = args.includes("-dr") || args.includes("--dry-run");

  // Check HF auth — skipped in dry-run mode (preview doesn't touch HF)
  if (!isDryRun) {
    try {
      const authRes = await fetch("/api/hf-auth");
      const authData = await authRes.json();
      if (!authData.authenticated) {
        clearTerminal();
        appendTerminal("Not logged in to HuggingFace. Run `hf login` in your terminal first.\n", "term-stderr");
        terminalStatus().textContent = "HF auth required";
        terminalStatus().className = "terminal-status error";
        return;
      }
    } catch (e) {
      // Continue anyway — the CLI will check auth too
    }
  }

  // Append --create-only for "create" action (ignored by the CLI in dry-run
  // mode, but harmless to include)
  if (action === "create") {
    args.push("--create-only");
  }

  // Run as normal command
  clearTerminal();
  const cmdStr = `ezexl3 upload ${args.join(" ")}`;
  appendTerminal(`$ ${cmdStr}\n`, "term-cmd");
  appendTerminal("\n");

  jobRunning = true;
  updateRunButton();
  const runningLabel = isDryRun
    ? "Dry run..."
    : action === "create" ? "Creating repos..." : "Uploading...";
  terminalStatus().textContent = runningLabel;
  terminalStatus().className = "terminal-status running";

  // Reveal more terminal real estate now that a job is starting.
  // The boot-time default is upper-panel = 90%; snap to 50/50 here.
  const upperPanel = document.getElementById("upper-panel");
  if (upperPanel) upperPanel.style.height = "50%";

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: "upload", args }),
    });
    const data = await res.json();

    if (data.error) {
      appendTerminal(`Error: ${data.error}\n`, "term-stderr");
      jobRunning = false;
      updateRunButton();
      terminalStatus().textContent = "Failed to start";
      terminalStatus().className = "terminal-status error";
      return;
    }

    activeJobId = data.job_id;
    await streamJob(data.job_id);
  } catch (e) {
    appendTerminal(`Connection error: ${e.message}\n`, "term-stderr");
    jobRunning = false;
    updateRunButton();
    terminalStatus().textContent = "Connection error";
    terminalStatus().className = "terminal-status error";
  }
}


async function runCommand() {
  if (jobRunning) return;

  const args = collectArgs();
  if (args.error) {
    terminalStatus().textContent = args.error;
    terminalStatus().className = "terminal-status error";
    return;
  }

  clearTerminal();
  if (typeof clearDataView === "function") clearDataView();
  const cmdStr = `ezexl3 ${activeCommand} ${args.join(" ")}`;
  appendTerminal(`$ ${cmdStr}\n`, "term-cmd");
  appendTerminal("\n");

  jobRunning = true;
  updateRunButton();
  startDataPolling();
  terminalStatus().textContent = "Running...";
  terminalStatus().className = "terminal-status running";

  // Reveal more terminal real estate now that a job is starting.
  // The boot-time default is upper-panel = 90%; snap to 50/50 here.
  const upperPanel = document.getElementById("upper-panel");
  if (upperPanel) upperPanel.style.height = "50%";

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: activeCommand, args }),
    });
    const data = await res.json();

    if (data.error) {
      appendTerminal(`Error: ${data.error}\n`, "term-stderr");
      jobRunning = false;
      updateRunButton();
      terminalStatus().textContent = "Failed to start";
      terminalStatus().className = "terminal-status error";
      return;
    }

    activeJobId = data.job_id;
    await streamJob(data.job_id);
  } catch (e) {
    appendTerminal(`Connection error: ${e.message}\n`, "term-stderr");
    jobRunning = false;
    updateRunButton();
    terminalStatus().textContent = "Connection error";
    terminalStatus().className = "terminal-status error";
  }
}


async function streamJob(jobId) {
  abortController = new AbortController();

  try {
    const res = await fetch(`/api/run/${jobId}/stream`, {
      signal: abortController.signal,
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // Keep incomplete line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") continue;

        try {
          const event = JSON.parse(payload);
          if (event.type === "stdout") {
            checkMetadataWait(event.text);
            checkBandwidth(event.text);
            if (!event.text.includes("<<EZEXL3:")) {
              appendTerminal(event.text);
            }
          } else if (event.type === "stderr") {
            appendTerminal(event.text, "term-stderr");
          } else if (event.type === "exit") {
            appendTerminal(`\n`);
            if (event.code === 0) {
              appendTerminal(`Process exited successfully (code 0)\n`, "term-success");
              terminalStatus().textContent = "Completed";
              terminalStatus().className = "terminal-status success";
            } else {
              appendTerminal(`Process exited with code ${event.code}\n`, "term-error");
              terminalStatus().textContent = `Exit code ${event.code}`;
              terminalStatus().className = "terminal-status error";
            }
          }
        } catch (e) {
          console.warn("SSE: malformed event payload", { payload, error: e });
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      appendTerminal(`\nStream disconnected: ${e.message}\n`, "term-stderr");
    }
  } finally {
    jobRunning = false;
    // Safety: no matter how the job ended, the freeze window is over.
    metadataFrozen = false;
    if (typeof syncMetaLockState === "function") syncMetaLockState();
    activeJobId = null;
    abortController = null;
    updateRunButton();
    stopDataPolling();
    refreshData();
  }
}


async function stopJob() {
  if (!activeJobId) return;
  try {
    await fetch(`/api/run/${activeJobId}/stop`, { method: "POST" });
    appendTerminal("\nProcess terminated by user\n", "term-stderr");
    terminalStatus().textContent = "Stopped";
    terminalStatus().className = "terminal-status error";
  } catch (e) {
    console.warn("stopJob: stop request failed", { jobId: activeJobId, error: e });
  }
  if (abortController) {
    abortController.abort();
  }
}


function updateRunButton() {
  const runBtn = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  const createBtn = document.getElementById("create-repos-btn");
  const uploadBtn = document.getElementById("upload-btn");
  if (jobRunning) {
    runBtn.disabled = true;
    createBtn.disabled = true;
    uploadBtn.disabled = true;
    stopBtn.style.display = "";
  } else {
    runBtn.disabled = false;
    createBtn.disabled = false;
    uploadBtn.disabled = false;
    stopBtn.style.display = "none";
  }
  if (typeof updateChatButton === "function") updateChatButton();
  if (typeof syncMetaLockState === "function") syncMetaLockState();
  if (typeof syncFormLockState === "function") syncFormLockState();
  // Re-apply the metadata gate — on upload tab this keeps the Create/Upload
  // buttons disabled unless MODEL and USER are locked.
  if (typeof updateMetadataConfirm === "function") updateMetadataConfirm();
}


function checkMetadataWait(text) {
  const match = text.match(/<<EZEXL3:WAITING_METADATA:(.+?)>>/);
  if (match) {
    if (typeof showMetadataWait === "function") {
      showMetadataWait(match[1]);
    }
    return;
  }
  const modelMatch = text.match(/<<EZEXL3:WAITING_MODEL_NAME:(.+?)>>/);
  if (modelMatch) {
    if (typeof showModelNameWait === "function") {
      showModelNameWait(modelMatch[1]);
    }
    return;
  }
  // The backend starts writing the README — freeze the metadata locks.
  // This also covers the "all locks were already set, no pause needed"
  // path where WAITING_METADATA is never printed.
  if (text.includes("<<EZEXL3:README_WRITING>>")) {
    metadataFrozen = true;
    if (typeof syncMetaLockState === "function") syncMetaLockState();
    return;
  }
  // README write finished — unfreeze the locks.
  if (text.includes("<<EZEXL3:README_DONE>>")) {
    metadataFrozen = false;
    if (typeof syncMetaLockState === "function") syncMetaLockState();
    return;
  }
}

function checkBandwidth(text) {
  const match = text.match(/<<EZEXL3:BANDWIDTH:([\d.]+) MB\/s>>/);
  if (!match) return;
  const speed = parseFloat(match[1]);
  if (speed >= 0.01) {
    terminalStatus().textContent = `Uploading: ${speed.toFixed(1)} MB/s`;
    terminalStatus().className = "terminal-status running";
  }
}
