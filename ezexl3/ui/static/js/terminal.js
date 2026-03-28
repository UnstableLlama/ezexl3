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
  if (className) {
    const span = document.createElement("span");
    span.className = className;
    span.textContent = text;
    el.appendChild(span);
  } else {
    el.appendChild(document.createTextNode(text));
  }
  // Auto-scroll
  el.scrollTop = el.scrollHeight;
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
  const cmdStr = `ezexl3 ${activeCommand} ${args.join(" ")}`;
  appendTerminal(`$ ${cmdStr}\n`, "term-cmd");
  appendTerminal("\n");

  jobRunning = true;
  updateRunButton();
  startDataPolling();
  terminalStatus().textContent = "Running...";
  terminalStatus().className = "terminal-status running";

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
            appendTerminal(event.text);
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
          // Skip malformed events
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      appendTerminal(`\nStream disconnected: ${e.message}\n`, "term-stderr");
    }
  } finally {
    jobRunning = false;
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
    // ignore
  }
  if (abortController) {
    abortController.abort();
  }
}


function updateRunButton() {
  const runBtn = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  if (jobRunning) {
    runBtn.disabled = true;
    runBtn.textContent = "Running...";
    stopBtn.style.display = "";
  } else {
    runBtn.disabled = false;
    runBtn.textContent = "Run";
    stopBtn.style.display = "none";
  }
}
