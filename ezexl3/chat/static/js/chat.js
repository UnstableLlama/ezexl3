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

// ── Send message ────────────────────────────────────────────────
async function sendMessage() {
  const text = inputBox.value.trim();
  if (!text || generating) return;

  inputBox.value = '';
  inputBox.style.height = 'auto';
  emptyState.style.display = 'none';

  // Get context from tree BEFORE adding the new node
  const context = getActivePath();

  // Add user message to tree
  const userNode = addUserNode(text);

  // Render tree (shows user message, empty assistant placeholder will be added)
  renderActiveTree();

  // Create assistant placeholder in DOM
  const assistantEl = createMsgEl('assistant', '');
  msgContainer.appendChild(assistantEl);
  const bodyEl = assistantEl.querySelector('.msg-body');

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

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

  // Create assistant placeholder
  const assistantEl = createMsgEl('assistant', '');
  msgContainer.appendChild(assistantEl);
  const bodyEl = assistantEl.querySelector('.msg-body');

  generating = true;
  sendBtn.style.display = 'none';
  stopBtn.style.display = 'flex';
  sendBtn.disabled = true;

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
