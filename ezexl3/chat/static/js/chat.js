// ── Chat: Streaming, send, regenerate, edit, continue, stop ─────

// ── Streaming helper (THE SACRED PIPELINE) ──────────────────────
async function streamResponse(message, context, bodyEl, {initialText = '', prefix = ''} = {}) {
  let fullText = initialText;
  let tpsData = null;

  const reqBody = {message, context};
  if (prefix) reqBody.prefix = prefix;
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(reqBody),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});

    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6);
      if (payload === '[DONE]') continue;

      try {
        const evt = JSON.parse(payload);
        switch (evt.type) {
          case 'token':
            fullText += evt.text;
            renderStreaming(bodyEl, fullText);
            scrollToBottom();
            break;
          case 'tps':
            tpsData = evt;
            break;
          case 'done':
            break;
          case 'error':
            fullText += `\n\n**Error:** ${evt.message}`;
            break;
        }
      } catch {}
    }
  }

  return {fullText, tpsData};
}

// ── DPO duel: generate two candidates for one user turn ────────
let duelStopped = false;  // set by stopGeneration(); abandons the duel

async function streamDuel(message, context, bodies, systemPrompts = null) {
  // One /api/chat request with n=2: the server batches both candidates
  // in a single generator pass and tags every SSE event with `cand`,
  // so both columns stream CONCURRENTLY. systemPrompts optionally
  // biases each candidate's generation (null entry = trained prompt).
  const texts = bodies.map(() => '');
  const tps = bodies.map(() => null);

  const reqBody = {message, context, n: bodies.length};
  if (systemPrompts && systemPrompts.some(Boolean)) {
    reqBody.system_prompts = systemPrompts;
  }
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(reqBody),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});

    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6);
      if (payload === '[DONE]') continue;

      try {
        const evt = JSON.parse(payload);
        const c = evt.cand || 0;
        switch (evt.type) {
          case 'token':
            texts[c] += evt.text;
            bodies[c].classList.remove('duel-waiting');
            renderStreaming(bodies[c], texts[c]);
            scrollToBottom();
            break;
          case 'tps':
            tps[c] = evt;
            break;
          case 'done':
            break;
          case 'error':
            texts[c] += `\n\n**Error:** ${evt.message}`;
            renderStreaming(bodies[c], texts[c]);
            break;
        }
      } catch {}
    }
  }

  return texts.map((fullText, i) => ({fullText, tpsData: tps[i]}));
}

async function runDuel(userNode, context) {
  // Streams N candidates side by side, adds each as a sibling assistant
  // node, and (unless stopped) arms pendingDuel so renderActiveTree shows
  // the judgment UI. Queue-driven bulk mode allows a single candidate,
  // generates every candidate under System Prompt A, and swaps the ▲/▼
  // pick UI for ✗ / Save-all.
  duelStopped = false;
  const bulk = bulkQueueActive();
  const n = bulk ? bulkCount() : duelCount();

  const duelEl = document.createElement('div');
  duelEl.className = 'duel-wrap';
  msgContainer.appendChild(duelEl);
  const bodies = [];
  for (let i = 0; i < n; i++) {
    const cand = document.createElement('div');
    cand.className = 'duel-candidate';
    cand.innerHTML =
      `<div class="duel-head"><span class="duel-label">${candLabel(i)}</span></div>` +
      '<div class="msg-body duel-waiting">&hellip;</div>';
    duelEl.appendChild(cand);
    bodies.push(cand.querySelector('.msg-body'));
  }

  const spoofs = bulk
    ? Array.from({length: n}, () => bulkGenSystem())
    : duelSystemPromptsFor(n);
  const results = await streamDuel(userNode.content, context, bodies, spoofs);

  const ids = [];
  results.forEach(({fullText, tpsData}, i) => {
    if (!fullText.trim()) return;  // stopped before any tokens
    const node = addAssistantNode(userNode.id, fullText.trim());
    if (tpsData) node.tpsData = tpsData;
    node.genSystem = spoofs[i] || null;
    ids.push(node.id);
  });

  if (ids.length === n && (n >= 2 || bulk) && !duelStopped) {
    pendingDuel = {userNodeId: userNode.id, ids, marks: {}, bulk,
                   sourceRow: bulk ? (promptQueue.current || null) : null};
  }
  renderActiveTree();
}

// ── DPO duel: regenerate the un-pinned candidates ──────────────
async function regenerateDuelCandidates() {
  // Duels: replaces every candidate NOT pinned with a ▲/▼ vote (unmarked
  // or ✗), keeping the voted candidates' text and marks. Bulk: unmarked
  // candidates are keepers (Save-all records them), so only ✗-marked
  // ones are replaced.
  if (!pendingDuel || generating || !modelLoaded) return;
  const duel = pendingDuel;
  const regenIdxs = duel.ids
    .map((id, i) => {
      if (duel.bulk) return duel.marks[id] === 'fail' ? i : -1;
      return (duel.marks[id] === 'up' || duel.marks[id] === 'down') ? -1 : i;
    })
    .filter(i => i >= 0);
  if (!regenIdxs.length) return;

  pendingDuel = null;
  duelStopped = false;

  const userNode = tree.nodes.get(duel.userNodeId);
  if (!userNode) { renderActiveTree(); return; }
  const context = getActivePathUpTo(userNode.id);

  // Drop the failed candidates from the tree; their replacements get
  // fresh node ids (any stale mark goes with them).
  for (const i of regenIdxs) {
    const id = duel.ids[i];
    const idx = userNode.children.indexOf(id);
    if (idx >= 0) userNode.children.splice(idx, 1);
    tree.nodes.delete(id);
    delete duel.marks[id];
  }

  // Render history up to the user turn, then rebuild the side-by-side
  // view: kept candidates static, failed slots streaming.
  const savedActiveChild = userNode.activeChild;
  userNode.activeChild = -1;
  renderActiveTree();
  userNode.activeChild = Math.min(savedActiveChild, userNode.children.length - 1);

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

  const duelEl = document.createElement('div');
  duelEl.className = 'duel-wrap';
  msgContainer.appendChild(duelEl);
  const streamBodies = [];
  duel.ids.forEach((id, i) => {
    const cand = document.createElement('div');
    cand.className = 'duel-candidate';
    cand.innerHTML =
      `<div class="duel-head"><span class="duel-label">${candLabel(i)}</span></div>` +
      '<div class="msg-body duel-waiting">&hellip;</div>';
    duelEl.appendChild(cand);
    const body = cand.querySelector('.msg-body');
    if (regenIdxs.includes(i)) {
      streamBodies.push(body);
    } else {
      const node = tree.nodes.get(id);
      body.classList.remove('duel-waiting');
      renderFinal(body, node ? node.content : '');
    }
  });

  try {
    // Each regenerated slot keeps its own generation prompt (A/B/… for
    // duels; System Prompt A across the board for bulk).
    const spoofs = duel.bulk
      ? Array.from({length: duel.ids.length}, () => bulkGenSystem())
      : duelSystemPromptsFor(duel.ids.length);
    const slotSpoofs = regenIdxs.map(i => spoofs[i] || null);
    const results = await streamDuel(userNode.content, context, streamBodies,
                                     slotSpoofs);
    results.forEach((res, k) => {
      const slot = regenIdxs[k];
      const text = res.fullText.trim();
      if (!text) { duel.ids[slot] = null; return; }  // stopped before tokens
      const node = addAssistantNode(userNode.id, text);
      if (res.tpsData) node.tpsData = res.tpsData;
      node.genSystem = slotSpoofs[k];
      duel.ids[slot] = node.id;
    });
  } catch (e) {
    console.error('Duel regen failed:', e);
  }

  generating = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  sendBtn.disabled = false;

  // Re-arm the duel only if both slots hold a live candidate; a stopped
  // regen leaves the surviving replies as ordinary siblings.
  if (!duelStopped && duel.ids.every(Boolean)) {
    pendingDuel = duel;
  }
  renderActiveTree();
  inputBox.focus();
  scrollToBottom();
}

// ── Send message ────────────────────────────────────────────────
async function sendMessage() {
  const text = inputBox.value.trim();
  if (!text || generating || !modelLoaded) return;
  if (typeof ratingsReady !== 'undefined') await ratingsReady;

  inputBox.value = '';
  inputBox.style.height = 'auto';
  emptyState.style.display = 'none';

  // Get context from tree BEFORE adding the new node
  const context = getActivePath();

  // Add user message to tree
  const userNode = addUserNode(text);

  // Render tree (shows user message, empty assistant placeholder will be added)
  renderActiveTree();

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

  if (ratingsMode === 'dpo') {
    try {
      await runDuel(userNode, context);
    } catch (e) {
      console.error('Duel failed:', e);
      renderActiveTree();
    }
  } else {
    // Create assistant placeholder in DOM
    const assistantEl = createMsgEl('assistant', '');
    msgContainer.appendChild(assistantEl);
    const bodyEl = assistantEl.querySelector('.msg-body');

    try {
      const {fullText, tpsData} = await streamResponse(text, context, bodyEl);

      // Final render
      renderFinal(bodyEl, fullText);

      // Add assistant to tree with TPS data
      const assistNode = addAssistantNode(userNode.id, fullText.trim());
      if (tpsData) assistNode.tpsData = tpsData;

      // Re-render full tree to get proper action buttons + TPS badge
      renderActiveTree();

    } catch (e) {
      renderFinal(bodyEl, `\n\n**Error:** ${e.message}`);
    }
  }

  generating = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  sendBtn.disabled = false;
  inputBox.focus();
  scrollToBottom();
}

// ── Regenerate response ─────────────────────────────────────────
async function regenerateResponse(assistantNodeId) {
  if (generating) return;
  if (typeof ratingsReady !== 'undefined') await ratingsReady;

  const assistNode = tree.nodes.get(assistantNodeId);
  if (!assistNode || assistNode.role !== 'assistant') return;

  const userNode = tree.nodes.get(assistNode.parentId);
  if (!userNode) return;

  // Context is everything BEFORE this user turn
  const context = getActivePathUpTo(userNode.id);

  const savedActiveChild = userNode.activeChild;
  userNode.activeChild = -1;
  renderActiveTree();
  userNode.activeChild = savedActiveChild;

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

  if (ratingsMode === 'dpo') {
    try {
      await runDuel(userNode, context);
    } catch (e) {
      console.error('Duel failed:', e);
      renderActiveTree();
    }
  } else {
    // Create assistant placeholder
    const assistantEl = createMsgEl('assistant', '');
    msgContainer.appendChild(assistantEl);
    const bodyEl = assistantEl.querySelector('.msg-body');

    try {
      const {fullText, tpsData} = await streamResponse(userNode.content, context, bodyEl);
      renderFinal(bodyEl, fullText);

      // Add as new sibling assistant node
      const newAssist = addAssistantNode(userNode.id, fullText.trim());
      if (tpsData) newAssist.tpsData = tpsData;

      renderActiveTree();
    } catch (e) {
      renderFinal(bodyEl, `\n\n**Error:** ${e.message}`);
    }
  }

  generating = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  sendBtn.disabled = false;
  inputBox.focus();
  scrollToBottom();
}

// ── Edit user message ───────────────────────────────────────────
function startEdit(userNodeId) {
  if (generating) return;
  const node = tree.nodes.get(userNodeId);
  if (!node || node.role !== 'user') return;

  editingNodeId = userNodeId;
  inputBox.value = node.content;
  inputBox.style.height = 'auto';
  inputBox.style.height = Math.min(inputBox.scrollHeight, 200) + 'px';
  inputBox.focus();
  inputBox.setSelectionRange(0, node.content.length);
  inputBox.style.borderColor = '#f59e0b';
  msgContainer.classList.add('editing');
}

function cancelInputEdit() {
  editingNodeId = null;
  inputBox.value = '';
  inputBox.style.height = 'auto';
  inputBox.style.borderColor = '';
  msgContainer.classList.remove('editing');
}

async function submitEdit(userNodeId, newText) {
  newText = newText.trim();
  if (!newText || generating) return;

  const oldNode = tree.nodes.get(userNodeId);
  if (!oldNode) { cancelInputEdit(); return; }

  cancelInputEdit();
  emptyState.style.display = 'none';

  // Context is everything BEFORE this user node
  const context = getActivePathUpTo(userNodeId);

  // Create sibling user node
  const parent = oldNode.parentId ? tree.nodes.get(oldNode.parentId) : null;
  const newUserNode = createNode('user', newText, oldNode.parentId);

  if (!oldNode.parentId) {
    tree.rootChildren.push(newUserNode.id);
    tree.activeRootChild = tree.rootChildren.length - 1;
  } else if (parent) {
    parent.children.push(newUserNode.id);
    parent.activeChild = parent.children.length - 1;
  }

  renderActiveTree();

  // Create assistant placeholder
  const assistantEl = createMsgEl('assistant', '');
  msgContainer.appendChild(assistantEl);
  const bodyEl = assistantEl.querySelector('.msg-body');

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

  try {
    const {fullText, tpsData} = await streamResponse(newText, context, bodyEl);
    renderFinal(bodyEl, fullText);
    const newAssist = addAssistantNode(newUserNode.id, fullText.trim());
    if (tpsData) newAssist.tpsData = tpsData;
    renderActiveTree();
  } catch (e) {
    renderFinal(bodyEl, `\n\n**Error:** ${e.message}`);
  }

  generating = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  sendBtn.disabled = false;
  inputBox.focus();
  scrollToBottom();
}

// ── Edit assistant message (reply spoofing) ─────────────────────
function startEditAssistant(assistantNodeId) {
  if (generating) return;
  const node = tree.nodes.get(assistantNodeId);
  if (!node || node.role !== 'assistant') return;

  const msgEl = document.querySelector(`[data-node-id="${assistantNodeId}"]`);
  if (!msgEl) return;
  const bodyEl = msgEl.querySelector('.msg-body');
  if (!bodyEl) return;

  const textarea = document.createElement('textarea');
  textarea.className = 'edit-assistant-textarea';
  textarea.value = node.content;
  textarea.style.width = '100%';
  textarea.style.minHeight = '100px';

  const btnRow = document.createElement('div');
  btnRow.className = 'edit-btns';
  const saveBtn2 = document.createElement('button');
  saveBtn2.className = 'save-btn';
  saveBtn2.textContent = 'Save';
  saveBtn2.addEventListener('click', () => saveAssistantEdit(assistantNodeId));
  const cancelBtn2 = document.createElement('button');
  cancelBtn2.className = 'cancel-btn';
  cancelBtn2.textContent = 'Cancel';
  cancelBtn2.addEventListener('click', () => renderActiveTree());
  btnRow.appendChild(saveBtn2);
  btnRow.appendChild(cancelBtn2);

  bodyEl.innerHTML = '';
  bodyEl.appendChild(textarea);
  bodyEl.appendChild(btnRow);
  textarea.focus();
  msgContainer.classList.add('editing');
}

function saveAssistantEdit(assistantNodeId) {
  const node = tree.nodes.get(assistantNodeId);
  if (!node) return;
  const textarea = document.querySelector(`[data-node-id="${assistantNodeId}"] .edit-assistant-textarea`);
  if (!textarea) return;
  const newText = textarea.value.trim();
  if (!newText) return;

  const userNode = tree.nodes.get(node.parentId);
  if (!userNode) return;

  const newNode = createNode('assistant', newText, node.parentId);
  userNode.children.push(newNode.id);
  userNode.activeChild = userNode.children.length - 1;

  renderActiveTree();
}

// ── Continue generation ─────────────────────────────────────────
async function continueGeneration(assistantNodeId) {
  if (generating) return;

  const assistNode = tree.nodes.get(assistantNodeId);
  if (!assistNode || assistNode.role !== 'assistant') return;

  const userNode = tree.nodes.get(assistNode.parentId);
  if (!userNode) return;

  const existingText = assistNode.content;
  const context = getActivePathUpTo(userNode.id);

  const savedActiveChild = userNode.activeChild;
  userNode.activeChild = -1;
  renderActiveTree();
  userNode.activeChild = savedActiveChild;

  const assistantEl = createMsgEl('assistant', '');
  msgContainer.appendChild(assistantEl);
  const bodyEl = assistantEl.querySelector('.msg-body');

  // Show existing text immediately
  renderStreaming(bodyEl, existingText);

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

  try {
    const {fullText, tpsData} = await streamResponse(
      userNode.content, context, bodyEl,
      {initialText: existingText, prefix: existingText}
    );
    renderFinal(bodyEl, fullText);

    const newAssist = addAssistantNode(userNode.id, fullText.trim());
    if (tpsData) newAssist.tpsData = tpsData;

    renderActiveTree();
  } catch (e) {
    renderFinal(bodyEl, `\n\n**Error:** ${e.message}`);
  }

  generating = false;
  sendBtn.style.display = 'flex';
  stopBtn.style.display = 'none';
  sendBtn.disabled = false;
  inputBox.focus();
  scrollToBottom();
}

// ── Stop generation ─────────────────────────────────────────────
async function stopGeneration() {
  duelStopped = true;  // a stopped duel is abandoned, not recorded
  await fetch('/api/stop', {method: 'POST'});
}

// ── Copy message ────────────────────────────────────────────────
function copyMessage(nodeId) {
  const node = tree.nodes.get(nodeId);
  if (!node) return;
  navigator.clipboard.writeText(node.content).then(() => {
    const btn = document.querySelector(`[data-copy="${nodeId}"]`);
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 1000);
    }
  }).catch(() => {});
}
