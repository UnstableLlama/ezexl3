// ── Global State ────────────────────────────────────────────────
// Shared mutable state used across all chat UI modules.

let generating = false;
const msgContainer = document.getElementById('messages');
const emptyState = document.getElementById('empty-state');
const inputBox = document.getElementById('input-box');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const sidebar = document.getElementById('sidebar');
let settings = {};
let modes = {};
let editingNodeId = null;

// ── Conversation Tree ───────────────────────────────────────────
// Each node: { id, role, content, parentId, children: [], activeChild: -1 }
const tree = {
  nodes: new Map(),
  rootChildren: [],     // IDs of first-level user nodes
  activeRootChild: -1,
};
