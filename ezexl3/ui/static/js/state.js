// ── Global state ────────────────────────────────────────────────

let activeCommand = "repo";
let activeJobId = null;
let jobRunning = false;
let templates = [];
let gpus = [];
let savedConfig = {};
let metadataWaitingDir = null;
