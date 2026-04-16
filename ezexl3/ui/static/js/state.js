// ── Global state ────────────────────────────────────────────────

let activeCommand = "repo";
let activeJobId = null;
let jobRunning = false;
let templates = [];
let gpus = [];
let savedConfig = {};
let metadataWaitingDir = null;
// True only while the README is actively being written (between the
// README_WRITING marker and the README_DONE marker).
// This is the one window where the metadata lock buttons are frozen.
let metadataFrozen = false;
