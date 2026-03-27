// ── Rendering: Markdown, Think Tags, DOM ────────────────────────

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true,
});

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderMarkdown(text) {
  if (!text.trim()) return '';
  try {
    return DOMPurify.sanitize(marked.parse(text));
  } catch {
    return `<p>${escHtml(text)}</p>`;
  }
}

function renderThinkContent(text, isOpen) {
  const label = isOpen ? 'Thinking...' : 'Thought';
  const openClass = isOpen ? ' open' : '';
  return `<div class="think-block">` +
    `<button class="think-toggle" onclick="this.nextElementSibling.classList.toggle('open');this.textContent=this.nextElementSibling.classList.contains('open')?'▼ ${label}':'▶ ${label}'">${isOpen ? '▼' : '▶'} ${label}</button>` +
    `<div class="think-content${openClass}">${renderMarkdown(text)}</div>` +
    `</div>`;
}

function processThinkTags(text) {
  let html = '';
  let remaining = text;
  let thinkOpen = false;

  while (remaining.length > 0) {
    const openIdx = remaining.indexOf('<think>');
    if (openIdx === -1) {
      if (thinkOpen) {
        html += renderThinkContent(remaining, true);
        remaining = '';
        thinkOpen = true;
      } else {
        html += renderMarkdown(remaining);
        remaining = '';
      }
    } else {
      if (openIdx > 0) {
        html += renderMarkdown(remaining.substring(0, openIdx));
      }
      remaining = remaining.substring(openIdx + 7);

      const closeIdx = remaining.indexOf('</think>');
      if (closeIdx === -1) {
        html += renderThinkContent(remaining, true);
        remaining = '';
        thinkOpen = true;
      } else {
        html += renderThinkContent(remaining.substring(0, closeIdx), false);
        remaining = remaining.substring(closeIdx + 8);
      }
    }
  }

  return {html, thinkOpen};
}

function renderStreaming(el, text) {
  const {html, thinkOpen} = processThinkTags(text);
  el.innerHTML = html + '<span class="cursor"></span>';
}

function renderFinal(el, text) {
  const {html} = processThinkTags(text);
  el.innerHTML = html;
}

function createMsgEl(role, text) {
  const msg = document.createElement('div');
  msg.className = 'msg';
  msg.innerHTML = `<div class="msg-role ${role}">${role}</div><div class="msg-body"></div>`;
  if (role === 'user' && text) {
    msg.querySelector('.msg-body').textContent = text;
  }
  return msg;
}

function scrollToBottom() {
  if (!document.getElementById('autoscroll').checked) return;
  const area = document.getElementById('chat-area');
  area.scrollTop = area.scrollHeight;
}

// ── Tree-aware rendering ────────────────────────────────────────
function renderActiveTree() {
  msgContainer.innerHTML = '';
  msgContainer.classList.remove('editing');

  const nodeIds = getActiveNodeIds();
  if (nodeIds.length === 0) {
    emptyState.style.display = '';
    msgContainer.appendChild(emptyState);
    return;
  }

  emptyState.style.display = 'none';

  for (const nodeId of nodeIds) {
    const node = tree.nodes.get(nodeId);
    if (!node) continue;

    const sibInfo = getSiblingInfo(nodeId);

    // Outer wrapper with side arrows
    const wrapEl = document.createElement('div');
    wrapEl.className = 'msg-wrap';

    // Left arrow: go to previous branch (only if current > 1)
    if (sibInfo && sibInfo.current > 1) {
      const leftArrow = document.createElement('div');
      leftArrow.className = 'msg-arrow msg-arrow-left';
      leftArrow.innerHTML = `<button onclick="switchBranch('${nodeId}', -1)" title="Previous version">&#x2039;</button>`;
      wrapEl.appendChild(leftArrow);
    }

    // Right arrow: go to next branch, or regen if at the end
    if (node.role === 'assistant') {
      const rightArrow = document.createElement('div');
      rightArrow.className = 'msg-arrow msg-arrow-right';
      if (sibInfo && sibInfo.current < sibInfo.total) {
        rightArrow.innerHTML = `<button onclick="switchBranch('${nodeId}', 1)" title="Next version">&#x203A;</button>`;
      } else {
        rightArrow.innerHTML = `<button class="regen-arrow" onclick="regenerateResponse('${nodeId}')" title="Regenerate">&#x203A;</button>`;
      }
      wrapEl.appendChild(rightArrow);
    }

    // The message grid
    const msgEl = document.createElement('div');
    msgEl.className = 'msg';
    msgEl.dataset.nodeId = nodeId;

    // Role label
    const roleEl = document.createElement('div');
    roleEl.className = `msg-role ${node.role}`;
    roleEl.textContent = node.role;
    msgEl.appendChild(roleEl);

    // Header row: branch counter + action buttons
    const headerEl = document.createElement('div');
    headerEl.className = 'msg-header';

    if (sibInfo) {
      headerEl.innerHTML += `<span class="branch-counter">${sibInfo.current} / ${sibInfo.total}</span>`;
    }

    // Action buttons (inline in header)
    if (node.role === 'user') {
      headerEl.innerHTML +=
        `<div class="msg-actions-inline">` +
        `<button onclick="startEdit('${nodeId}')">Edit</button>` +
        `<button class="danger" onclick="deleteNode('${nodeId}')">Delete</button>` +
        `</div>`;
    } else {
      headerEl.innerHTML +=
        `<div class="msg-actions-inline">` +
        `<button onclick="startEditAssistant('${nodeId}')">Edit</button>` +
        `<button onclick="regenerateResponse('${nodeId}')">Regen</button>` +
        `<button onclick="continueGeneration('${nodeId}')">Continue</button>` +
        `<button data-copy="${nodeId}" onclick="copyMessage('${nodeId}')">Copy</button>` +
        `<button class="danger" onclick="deleteNode('${nodeId}')">Delete</button>` +
        `</div>`;
    }

    msgEl.appendChild(headerEl);

    // Message body
    const bodyEl = document.createElement('div');
    bodyEl.className = 'msg-body';
    if (node.role === 'user') {
      bodyEl.textContent = node.content;
    } else {
      renderFinal(bodyEl, node.content);
    }
    msgEl.appendChild(bodyEl);

    // TPS badge below message body (for assistant messages with stats)
    if (node.role === 'assistant' && node.tpsData) {
      const t = node.tpsData;
      const ctx = t.prompt_tokens ? `${t.prompt_tokens.toLocaleString()} ctx` : '';
      const cached = t.cached_tokens ? ` (${t.cached_tokens.toLocaleString()} cached)` : '';
      let parts = [];
      if (ctx) parts.push(ctx + cached);
      parts.push(`${t.new_tokens} gen · ${t.tps} t/s · ${t.elapsed}s`);
      if (t.prefill_tps) parts.push(`prefill ${t.prefill_tps} t/s`);
      const tpsEl = document.createElement('div');
      tpsEl.className = 'msg-tps';
      tpsEl.textContent = parts.join(' · ');
      msgEl.appendChild(tpsEl);
    }

    wrapEl.appendChild(msgEl);
    msgContainer.appendChild(wrapEl);
  }

  scrollToBottom();
}
