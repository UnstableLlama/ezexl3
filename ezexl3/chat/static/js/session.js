// ── Session: Save, Load, Export, Clear ───────────────────────────

async function clearContext() {
  await fetch('/api/clear', {method: 'POST'});
  tree.nodes.clear();
  tree.rootChildren = [];
  tree.activeRootChild = -1;
  msgContainer.innerHTML = '';
  emptyState.style.display = '';
  msgContainer.appendChild(emptyState);
}

async function saveSession() {
  const resp = await fetch('/api/session/save');
  const data = await resp.json();
  // Augment with tree data
  data.tree = {
    nodes: Object.fromEntries(tree.nodes),
    rootChildren: tree.rootChildren,
    activeRootChild: tree.activeRootChild,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'chat_session.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

function loadSession() {
  document.getElementById('file-input').click();
}

function handleSessionFileLoad(e) {
  const file = e.target.files[0];
  if (!file) return;
  file.text().then(async (text) => {
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      alert('Invalid session file: not valid JSON');
      e.target.value = '';
      return;
    }
    await fetch('/api/session/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: text,
    });
    // Reload settings
    const settingsRes = await fetch('/api/settings');
    settings = await settingsRes.json();
    const statusRes = await fetch('/api/status');
    const status = await statusRes.json();
    populateUI(status);

    // Restore tree
    if (data.tree) {
      tree.nodes = new Map(Object.entries(data.tree.nodes));
      tree.rootChildren = data.tree.rootChildren;
      tree.activeRootChild = data.tree.activeRootChild;
    } else if (data.context) {
      // Backward compat: build linear tree from flat context
      buildTreeFromFlatContext(data.context);
    }

    renderActiveTree();
    e.target.value = '';
    scrollToBottom();
  });
}

function exportMarkdown() {
  const path = getActivePath();
  if (path.length === 0) return;
  let md = '';
  for (const [user, assistant] of path) {
    md += `## User\n\n${user}\n\n`;
    if (assistant) md += `## Assistant\n\n${assistant}\n\n`;
  }
  const blob = new Blob([md.trim() + '\n'], {type: 'text/markdown'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'chat_export.md';
  a.click();
  URL.revokeObjectURL(a.href);
}
