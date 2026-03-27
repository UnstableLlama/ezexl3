// ── Conversation Tree Operations ─────────────────────────────────
// Pure tree-manipulation functions operating on the global `tree` state.

function createNode(role, content, parentId) {
  const node = {
    id: crypto.randomUUID(),
    role,
    content,
    parentId: parentId || null,
    children: [],
    activeChild: -1,
    tpsData: null,
  };
  tree.nodes.set(node.id, node);
  return node;
}

function getActiveLeaf() {
  if (tree.rootChildren.length === 0 || tree.activeRootChild < 0) return null;
  let nodeId = tree.rootChildren[tree.activeRootChild];
  let node = tree.nodes.get(nodeId);
  while (node && node.activeChild >= 0 && node.activeChild < node.children.length) {
    nodeId = node.children[node.activeChild];
    node = tree.nodes.get(nodeId);
  }
  return node;
}

function getActivePath() {
  // Returns [[user_content, assistant_content_or_null], ...] for backend context
  const path = [];
  if (tree.rootChildren.length === 0 || tree.activeRootChild < 0) return path;

  let childIds = tree.rootChildren;
  let activeIdx = tree.activeRootChild;

  while (activeIdx >= 0 && activeIdx < childIds.length) {
    const userNode = tree.nodes.get(childIds[activeIdx]);
    if (!userNode || userNode.role !== 'user') break;

    if (userNode.activeChild >= 0 && userNode.activeChild < userNode.children.length) {
      const assistNode = tree.nodes.get(userNode.children[userNode.activeChild]);
      if (assistNode) {
        path.push([userNode.content, assistNode.content]);
        childIds = assistNode.children;
        activeIdx = assistNode.activeChild;
        continue;
      }
    }
    // User node with no (active) assistant response
    path.push([userNode.content, null]);
    break;
  }
  return path;
}

function getActivePathUpTo(nodeId) {
  // Returns context tuples for everything BEFORE nodeId on the active path
  const fullPath = getActiveNodeIds();
  const path = [];
  for (let i = 0; i < fullPath.length; i++) {
    if (fullPath[i] === nodeId) break;
    const node = tree.nodes.get(fullPath[i]);
    if (node.role === 'user') {
      // Look ahead for assistant
      if (i + 1 < fullPath.length && fullPath[i + 1] !== nodeId) {
        const next = tree.nodes.get(fullPath[i + 1]);
        if (next && next.role === 'assistant') {
          path.push([node.content, next.content]);
        }
      }
    }
  }
  return path;
}

function getActiveNodeIds() {
  // Returns flat array of node IDs on the active path
  const ids = [];
  if (tree.rootChildren.length === 0 || tree.activeRootChild < 0) return ids;

  let childIds = tree.rootChildren;
  let activeIdx = tree.activeRootChild;

  while (activeIdx >= 0 && activeIdx < childIds.length) {
    const nodeId = childIds[activeIdx];
    const node = tree.nodes.get(nodeId);
    if (!node) break;
    ids.push(nodeId);
    childIds = node.children;
    activeIdx = node.activeChild;
  }
  return ids;
}

function addUserNode(content) {
  const leaf = getActiveLeaf();
  const node = createNode('user', content, leaf ? leaf.id : null);

  if (!leaf) {
    tree.rootChildren.push(node.id);
    tree.activeRootChild = tree.rootChildren.length - 1;
  } else {
    leaf.children.push(node.id);
    leaf.activeChild = leaf.children.length - 1;
  }
  return node;
}

function addAssistantNode(userNodeId, content) {
  const userNode = tree.nodes.get(userNodeId);
  const node = createNode('assistant', content, userNodeId);
  userNode.children.push(node.id);
  userNode.activeChild = userNode.children.length - 1;
  return node;
}

function deleteSubtree(nodeId) {
  const node = tree.nodes.get(nodeId);
  if (!node) return;
  for (const childId of [...node.children]) {
    deleteSubtree(childId);
  }
  tree.nodes.delete(nodeId);
}

function deleteNode(nodeId) {
  if (generating) return;
  const node = tree.nodes.get(nodeId);
  if (!node) return;

  if (!node.parentId) {
    const idx = tree.rootChildren.indexOf(nodeId);
    if (idx >= 0) {
      tree.rootChildren.splice(idx, 1);
      deleteSubtree(nodeId);
      if (tree.rootChildren.length === 0) {
        tree.activeRootChild = -1;
      } else {
        tree.activeRootChild = Math.min(tree.activeRootChild, tree.rootChildren.length - 1);
      }
    }
  } else {
    const parent = tree.nodes.get(node.parentId);
    if (parent) {
      const idx = parent.children.indexOf(nodeId);
      if (idx >= 0) {
        parent.children.splice(idx, 1);
        deleteSubtree(nodeId);
        if (parent.children.length === 0) {
          parent.activeChild = -1;
        } else {
          parent.activeChild = Math.min(parent.activeChild, parent.children.length - 1);
        }
      }
    }
  }
  renderActiveTree();
}

function switchBranch(nodeId, direction) {
  const node = tree.nodes.get(nodeId);
  if (!node) return;

  if (!node.parentId) {
    const idx = tree.rootChildren.indexOf(nodeId);
    const newIdx = idx + direction;
    if (newIdx >= 0 && newIdx < tree.rootChildren.length) {
      tree.activeRootChild = newIdx;
    }
  } else {
    const parent = tree.nodes.get(node.parentId);
    if (!parent) return;
    const newIdx = parent.activeChild + direction;
    if (newIdx >= 0 && newIdx < parent.children.length) {
      parent.activeChild = newIdx;
    }
  }
  renderActiveTree();
}

function getSiblingInfo(nodeId) {
  const node = tree.nodes.get(nodeId);
  if (!node) return null;

  let siblings, idx;
  if (!node.parentId) {
    siblings = tree.rootChildren;
    idx = siblings.indexOf(nodeId);
  } else {
    const parent = tree.nodes.get(node.parentId);
    if (!parent) return null;
    siblings = parent.children;
    idx = siblings.indexOf(nodeId);
  }
  if (siblings.length <= 1) return null;
  return { current: idx + 1, total: siblings.length };
}

function buildTreeFromFlatContext(contextTuples) {
  tree.nodes.clear();
  tree.rootChildren = [];
  tree.activeRootChild = -1;

  for (const [userText, assistText] of contextTuples) {
    const userNode = addUserNode(userText);
    if (assistText != null) {
      addAssistantNode(userNode.id, assistText);
    }
  }
}
