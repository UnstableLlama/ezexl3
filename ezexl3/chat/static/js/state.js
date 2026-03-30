// ── Global State ────────────────────────────────────────────────
// Shared mutable state used across all chat UI modules.

let generating = false;
let modelLoaded = false;
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

// ── Chat enable/disable based on model state ────────────────────
function updateChatEnabled() {
  if (modelLoaded) {
    inputBox.disabled = false;
    inputBox.placeholder = 'Type a message... (Enter to send, Shift+Enter for newline)';
    sendBtn.disabled = false;
    document.getElementById('empty-hint').textContent = 'Send a message to start chatting';
  } else {
    inputBox.disabled = true;
    inputBox.placeholder = 'Load a model to begin chatting';
    sendBtn.disabled = true;
    document.getElementById('empty-hint').textContent = 'Load a model in the sidebar to begin';
  }
}
