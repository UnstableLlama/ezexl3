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

function highlightDialogue(html) {
  // Split on <pre>...</pre> and <code>...</code> to avoid replacing inside code blocks
  return html.replace(/(<pre[\s>][\s\S]*?<\/pre>|<code[\s>][\s\S]*?<\/code>)|(?:&quot;|"|[\u201C\u201D])(.+?)(?:&quot;|"|[\u201C\u201D])/g,
    (match, codeBlock, inner) => {
      if (codeBlock) return codeBlock; // pass code blocks through unchanged
      return `<span class="dialogue">\u201C${inner}\u201D</span>`;
    });
}

function renderMarkdown(text) {
  if (!text.trim()) return '';
  try {
    return DOMPurify.sanitize(highlightDialogue(marked.parse(text)));
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
    renderFinal(msg.querySelector('.msg-body'), text);
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

    // A pending DPO duel renders as a side-by-side pick UI covering both
    // candidates (only one of them is ever on the active path).
    if (typeof pendingDuel !== 'undefined' && pendingDuel &&
        (nodeId === pendingDuel.a || nodeId === pendingDuel.b)) {
      renderDuelChoice(msgContainer);
      continue;
    }

    const sibInfo = getSiblingInfo(nodeId);

    // Outer wrapper with side arrows
    const wrapEl = document.createElement('div');
    wrapEl.className = 'msg-wrap';

    // Left arrow: go to previous branch (only if current > 1)
    if (sibInfo && sibInfo.current > 1) {
      const leftArrow = document.createElement('div');
      leftArrow.className = 'msg-arrow msg-arrow-left';
      const leftBtn = document.createElement('button');
      leftBtn.title = 'Previous version';
      leftBtn.innerHTML = '&#x2039;';
      leftBtn.addEventListener('click', () => switchBranch(nodeId, -1));
      leftArrow.appendChild(leftBtn);
      wrapEl.appendChild(leftArrow);
    }

    // Right arrow: go to next branch, or regen if at the end
    if (node.role === 'assistant') {
      const rightArrow = document.createElement('div');
      rightArrow.className = 'msg-arrow msg-arrow-right';
      const rightBtn = document.createElement('button');
      if (sibInfo && sibInfo.current < sibInfo.total) {
        rightBtn.title = 'Next version';
        rightBtn.innerHTML = '&#x203A;';
        rightBtn.addEventListener('click', () => switchBranch(nodeId, 1));
      } else {
        rightBtn.className = 'regen-arrow';
        rightBtn.title = 'Regenerate';
        rightBtn.innerHTML = '&#x203A;';
        rightBtn.addEventListener('click', () => regenerateResponse(nodeId));
      }
      rightArrow.appendChild(rightBtn);
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

    // Rating controls (assistant only), gated by the capture mode:
    // KTO mode shows 👍/👎; DPO mode shows a "preferred" badge on
    // messages whose duel pick is recorded (click to withdraw).
    if (node.role === 'assistant' && typeof getRating === 'function') {
      const ratingSpan = document.createElement('span');
      ratingSpan.className = 'msg-rating';
      const rating = getRating(nodeId);

      function addRate(glyph, title, handler, activeCls, active) {
        const btn = document.createElement('button');
        btn.textContent = glyph;
        btn.title = title;
        if (active) btn.className = activeCls;
        btn.addEventListener('click', handler);
        ratingSpan.appendChild(btn);
      }

      if (ratingsMode === 'kto') {
        addRate('\u{1F44D}', 'Good response (KTO)',
          () => rateNode(nodeId, true), 'rated-good', rating === true);
        addRate('\u{1F44E}', 'Bad response (KTO)',
          () => rateNode(nodeId, false), 'rated-bad', rating === false);
        if (rating !== undefined) ratingSpan.classList.add('has-rating');
      } else if (pairForNode(nodeId)) {
        addRate('✓ preferred', 'DPO pair recorded — click to withdraw',
          () => removePairFor(nodeId), 'rated-pair', true);
        ratingSpan.classList.add('has-rating');
      }
      if (ratingSpan.childElementCount) headerEl.appendChild(ratingSpan);
    }

    // Action buttons (inline in header)
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'msg-actions-inline';

    function addAction(label, handler, cls) {
      const btn = document.createElement('button');
      btn.textContent = label;
      if (cls) btn.className = cls;
      btn.addEventListener('click', handler);
      actionsDiv.appendChild(btn);
      return btn;
    }

    if (node.role === 'user') {
      addAction('Edit', () => startEdit(nodeId));
      addAction('Delete', () => deleteNode(nodeId), 'danger');
    } else {
      addAction('Edit', () => startEditAssistant(nodeId));
      addAction('Regen', () => regenerateResponse(nodeId));
      addAction('Continue', () => continueGeneration(nodeId));
      const copyBtn = addAction('Copy', () => copyMessage(nodeId));
      copyBtn.dataset.copy = nodeId;
      addAction('Delete', () => deleteNode(nodeId), 'danger');
    }
    headerEl.appendChild(actionsDiv);

    msgEl.appendChild(headerEl);

    // Message body
    const bodyEl = document.createElement('div');
    bodyEl.className = 'msg-body';
    renderFinal(bodyEl, node.content);
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
      if (t.draft_accepted != null) {
        const rate = Math.round(t.draft_acceptance_rate * 100);
        parts.push(`draft ${rate}% (${t.draft_accepted}/${t.draft_accepted + t.draft_rejected})`);
      }
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

// ── DPO duel pick UI ────────────────────────────────────────────
function renderDuelChoice(container) {
  const block = document.createElement('div');
  block.className = 'duel-block';

  const wrap = document.createElement('div');
  wrap.className = 'duel-wrap';
  for (const [id, label] of [[pendingDuel.a, 'A'], [pendingDuel.b, 'B']]) {
    const node = tree.nodes.get(id);
    if (!node) continue;
    const cand = document.createElement('div');
    cand.className = 'duel-candidate';

    const head = document.createElement('div');
    head.className = 'duel-head';
    head.innerHTML = `<span class="duel-label">${label}</span>`;
    const pickBtn = document.createElement('button');
    pickBtn.className = 'duel-pick-btn';
    pickBtn.textContent = `Prefer ${label}`;
    pickBtn.title = 'Record the DPO pair and continue from this reply';
    pickBtn.addEventListener('click', () => resolveDuel(id, true));
    head.appendChild(pickBtn);
    cand.appendChild(head);

    const body = document.createElement('div');
    body.className = 'msg-body';
    renderFinal(body, node.content);
    cand.appendChild(body);
    wrap.appendChild(cand);
  }
  block.appendChild(wrap);

  const skip = document.createElement('div');
  skip.className = 'duel-skip';
  const skipBtn = document.createElement('button');
  skipBtn.textContent = 'Skip — continue without recording';
  skipBtn.title = 'Keep candidate B and record nothing';
  skipBtn.addEventListener('click', () => resolveDuel(pendingDuel.b, false));
  skip.appendChild(skipBtn);
  block.appendChild(skip);

  container.appendChild(block);
}
