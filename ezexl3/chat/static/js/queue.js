// ── Prompt queue: batch DPO capture from a JSONL of prompts ─────
// Open a JSONL file of prompts and run through it as DPO duels: each
// prompt starts a FRESH conversation (no carried context), and every
// Commit/Skip advances the server-side cursor — checkpointed per file
// in the ratings dir — then auto-starts the next prompt. Leave the
// start line blank to resume from the checkpoint.

let queueState = {active: false};
// User node of the duel the queue itself started — commits of manual
// side-chat duels must not advance the queue.
let queueCurrentUserNodeId = null;

function queueEls() {
  return {
    path: document.getElementById('queue-path'),
    startLine: document.getElementById('queue-start-line'),
    openBtn: document.getElementById('queue-open-btn'),
    closeBtn: document.getElementById('queue-close-btn'),
    controls: document.getElementById('queue-controls'),
    nextBtn: document.getElementById('queue-next-btn'),
    skipBtn: document.getElementById('queue-skip-btn'),
    status: document.getElementById('queue-status'),
  };
}

function applyQueueStatus(data) {
  queueState = data && data.active ? data : {active: false};
  renderQueuePanel();
}

function renderQueuePanel(error) {
  const els = queueEls();
  if (!els.status) return;
  const q = queueState;
  els.openBtn.style.display = q.active ? 'none' : '';
  els.closeBtn.style.display = q.active ? '' : 'none';
  els.controls.style.display = q.active && !q.done ? 'flex' : 'none';
  if (error) {
    els.status.style.display = '';
    els.status.textContent = `⚠ ${error}`;
    return;
  }
  if (!q.active) {
    els.status.style.display = 'none';
    els.status.textContent = '';
    return;
  }
  els.status.style.display = '';
  els.status.textContent = q.done
    ? `Queue complete — all ${q.total} prompts served ✓`
    : `Prompt ${q.index + 1} / ${q.total} (line ${q.line})` +
      ` — ${q.remaining} left`;
}

async function openQueue() {
  const els = queueEls();
  const path = els.path.value.trim();
  if (!path) { renderQueuePanel('Enter a prompts .jsonl path'); return; }
  const body = {path};
  const startLine = parseInt(els.startLine.value, 10);
  if (startLine >= 1) body.start_line = startLine;
  try {
    const res = await fetch('/api/queue/open', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || data.error) { renderQueuePanel(data.error || `HTTP ${res.status}`); return; }
    applyQueueStatus(data);
    fetch('/api/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({queue_path: path}),
    }).catch(() => {});
    // Queue prompts are judged as DPO duels — switch modes if needed.
    if (ratingsMode !== 'dpo') setRatingsMode('dpo');
  } catch (e) {
    renderQueuePanel(e.message);
  }
}

async function closeQueue() {
  try {
    const res = await fetch('/api/queue/close', {method: 'POST'});
    applyQueueStatus(await res.json());
  } catch (e) {
    renderQueuePanel(e.message);
  }
}

// Run the current queue prompt as a fresh single-turn DPO duel: a new
// root-level conversation with no carried context.
async function queueRunCurrent() {
  if (!queueState.active || queueState.done || generating || !modelLoaded) return;
  if (typeof pendingDuel !== 'undefined' && pendingDuel) return;  // judge first
  if (typeof ratingsReady !== 'undefined') await ratingsReady;
  if (ratingsMode !== 'dpo') setRatingsMode('dpo');

  emptyState.style.display = 'none';
  const userNode = createNode('user', queueState.prompt, null);
  queueCurrentUserNodeId = userNode.id;
  tree.rootChildren.push(userNode.id);
  tree.activeRootChild = tree.rootChildren.length - 1;
  renderActiveTree();

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;
  try {
    await runDuel(userNode, []);
  } catch (e) {
    console.error('Queue duel failed:', e);
    renderActiveTree();
  }
  generating = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  sendBtn.disabled = false;
  scrollToBottom();
}

async function queueAdvance() {
  // Advance past the current prompt; the index guard server-side makes
  // a duplicate advance harmless. Returns true if a next prompt awaits.
  if (!queueState.active || queueState.done) return false;
  try {
    const res = await fetch('/api/queue/advance', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({index: queueState.index}),
    });
    const data = await res.json();
    if (!res.ok || data.error) { renderQueuePanel(data.error || `HTTP ${res.status}`); return false; }
    applyQueueStatus(data);
    return queueState.active && !queueState.done;
  } catch (e) {
    renderQueuePanel(e.message);
    return false;
  }
}

// Called by commitDuel/skipDuel after a duel is resolved. Advances the
// queue and auto-starts the next prompt — but only for the duel the
// queue itself started (manual side-chat duels don't advance it), and
// not for skips fired by a mode switch away from DPO (those land here
// with ratingsMode already changed and must not advance the cursor).
async function queueDuelResolved(userNodeId) {
  if (!queueState.active || queueState.done || ratingsMode !== 'dpo') return;
  if (!userNodeId || userNodeId !== queueCurrentUserNodeId) return;
  queueCurrentUserNodeId = null;
  if (await queueAdvance()) await queueRunCurrent();
}

async function queueSkipPrompt() {
  // Pass over the current prompt without generating or judging it.
  if (generating) return;
  if (await queueAdvance()) await queueRunCurrent();
}

async function initQueue() {
  const els = queueEls();
  if (!els.openBtn) return;
  els.openBtn.addEventListener('click', openQueue);
  els.closeBtn.addEventListener('click', closeQueue);
  els.nextBtn.addEventListener('click', queueRunCurrent);
  els.skipBtn.addEventListener('click', queueSkipPrompt);
  try {
    const cfg = await (await fetch('/api/config')).json();
    if (cfg.queue_path && !els.path.value) els.path.value = cfg.queue_path;
  } catch {}
  try {
    // Restore an already-open queue (e.g. after a page reload).
    const res = await fetch('/api/queue');
    applyQueueStatus(await res.json());
  } catch {}
}

initQueue();
