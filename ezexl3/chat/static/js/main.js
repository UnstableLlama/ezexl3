// ── Main: Init, input handling, keyboard shortcuts ──────────────

// ── Init ────────────────────────────────────────────────────────
(async () => {
  try {
    const [statusRes, settingsRes] = await Promise.all([
      fetch('/api/status'),
      fetch('/api/settings'),
    ]);
    const status = await statusRes.json();
    settings = await settingsRes.json();
    modes = status.available_modes || {};
    modelLoaded = status.loaded;

    // Always init model panel (handles both loaded and unloaded states)
    await initModelPanel(status);
    initLoraPanel();
    initDraftPanel();

    // Always populate UI — initializes sliders, toggles, mode dropdown,
    // and model info regardless of whether a model is loaded yet.
    populateUI(status);
    updateChatEnabled();

    if (status.loaded) {
      showLoraPanel(true);
      syncLoraState(status);
      showDraftPanel(true);
      syncDraftState(status);
    }
  } catch (err) {
    console.error('Chat init failed:', err);
    document.getElementById('empty-hint').textContent =
      'Failed to connect to backend — is the server running?';
  }
})();

// ── Sidebar toggle ──────────────────────────────────────────────
document.getElementById('toggle-sidebar').onclick = () => sidebar.classList.toggle('hidden');

// ── Input handling ──────────────────────────────────────────────
inputBox.addEventListener('input', () => {
  inputBox.style.height = 'auto';
  inputBox.style.height = Math.min(inputBox.scrollHeight, 200) + 'px';
});
inputBox.addEventListener('blur', () => {
  if (editingNodeId && !inputBox.value.trim()) {
    cancelInputEdit();
  }
});
inputBox.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (editingNodeId) {
      submitEdit(editingNodeId, inputBox.value);
    } else {
      sendMessage();
    }
  }
  if (e.key === 'Escape' && editingNodeId) {
    cancelInputEdit();
  }
});
sendBtn.onclick = () => {
  if (editingNodeId) {
    submitEdit(editingNodeId, inputBox.value);
  } else {
    sendMessage();
  }
};
stopBtn.onclick = () => stopGeneration();

// ── Settings event listeners ────────────────────────────────────
document.getElementById('s-system').addEventListener('input', syncSettings);
document.getElementById('s-tplkwargs').addEventListener('input', syncSettings);
document.getElementById('s-mode').addEventListener('change', syncSettings);
document.getElementById('s-thinkbudget').addEventListener('input', syncSettings);
document.getElementById('strip-formatting').addEventListener('change',
  e => setStripFormatting(e.target.checked));
document.getElementById('banned-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); addBan(); }
});
document.getElementById('file-input').addEventListener('change', handleSessionFileLoad);

// ── Keyboard shortcuts ──────────────────────────────────────────
document.addEventListener('keydown', (e) => {
  // Ctrl+Shift+Backspace -> stop generation
  if (e.ctrlKey && e.shiftKey && e.key === 'Backspace') {
    e.preventDefault();
    stopGeneration();
    return;
  }
  // Up arrow in empty input -> edit last user message
  if (e.key === 'ArrowUp' && document.activeElement === inputBox && !inputBox.value.trim() && !editingNodeId) {
    const nodeIds = getActiveNodeIds();
    for (let i = nodeIds.length - 1; i >= 0; i--) {
      const n = tree.nodes.get(nodeIds[i]);
      if (n && n.role === 'user') {
        e.preventDefault();
        startEdit(n.id);
        return;
      }
    }
  }
});
