// ── Dynamic form rendering from command schemas ─────────────────

function renderForm(commandKey) {
  const cmd = COMMANDS[commandKey];
  if (!cmd) return;

  const container = document.getElementById("form-fields");
  container.innerHTML = "";

  const required = cmd.fields.filter(f => f.required && f.type !== "boolean" && !f.section);
  const optional = cmd.fields.filter(f => !f.required && !f.section && (f.type !== "boolean" || f.toggleable));
  const booleans = cmd.fields.filter(f => f.type === "boolean" && !f.toggleable && !f.section);
  const evals = cmd.fields.filter(f => f.section === "evals");

  // Required fields
  if (required.length) {
    for (const field of required) {
      container.appendChild(createFieldEl(field));
    }
  }

  // Optional fields
  if (optional.length) {
    const heading = document.createElement("div");
    heading.className = "form-section-label";
    heading.textContent = "Options";
    container.appendChild(heading);
    for (const field of optional) {
      container.appendChild(createFieldEl(field));
    }
  }

  // Boolean flags in a compact grid
  if (booleans.length) {
    const heading = document.createElement("div");
    heading.className = "form-section-label";
    heading.textContent = "Flags";
    container.appendChild(heading);
    const grid = document.createElement("div");
    grid.className = "toggle-grid";
    for (const field of booleans) {
      grid.appendChild(createToggleEl(field));
    }
    container.appendChild(grid);
  }

  // Side panels (right column)
  renderEvalsPanel(commandKey);
  renderMetadataPanel(commandKey);

  // Show/hide right panels container
  const rightPanels = document.getElementById("right-panels");
  const hasRight = cmd.hasMetadata || evals.length > 0;
  rightPanels.style.display = hasRight ? "" : "none";
}


function createFieldEl(field) {
  const row = document.createElement("div");
  row.className = "form-row";

  const labelRow = document.createElement("div");
  labelRow.className = "form-label-row";
  const label = document.createElement("label");
  label.className = "form-label";
  label.textContent = field.label;
  if (field.required) {
    const star = document.createElement("span");
    star.className = "required-star";
    star.textContent = " *";
    label.appendChild(star);
  }
  labelRow.appendChild(label);

  if (field.toggleable) {
    const toggle = document.createElement("label");
    toggle.className = "field-toggle";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `toggle-${field.name}`;
    toggle.appendChild(cb);
    const slider = document.createElement("span");
    slider.className = "toggle-slider";
    toggle.appendChild(slider);
    labelRow.appendChild(toggle);
  }
  row.appendChild(labelRow);

  if (field.help) {
    const help = document.createElement("div");
    help.className = "form-help";
    help.textContent = field.help;
    row.appendChild(help);
  }

  // Helper: also expose field.help as a hover tooltip on the input itself
  function attachHelpTooltip(el) {
    if (el && field.help) el.title = field.help;
  }

  let input;

  if (field.type === "boolean" && field.toggleable) {
    // Toggleable boolean — the toggle in the label row IS the input, no extra field needed
    // Create a hidden checkbox to serve as the field value for collectArgs
    input = document.createElement("input");
    input.type = "checkbox";
    input.id = `field-${field.name}`;
    input.style.display = "none";
    row.appendChild(input);
    // Sync the label-row toggle with the hidden field checkbox
    const toggleCb = row.querySelector(`#toggle-${field.name}`);
    if (toggleCb) {
      // defaultOn: start checked (e.g. KL Div and PPL are on by default)
      if (field.defaultOn) {
        toggleCb.checked = true;
        input.checked = true;
      }
      toggleCb.addEventListener("change", () => { input.checked = toggleCb.checked; });
    }
    return row;
  } else if (field.type === "path") {
    const wrap = document.createElement("div");
    wrap.className = "path-input-wrap";
    input = document.createElement("input");
    input.type = "text";
    input.className = "form-input";
    input.id = `field-${field.name}`;
    input.placeholder = "/path/to/model";
    // Restore saved model directory
    if (field.name === "models" && savedConfig.last_model_dir) {
      input.value = savedConfig.last_model_dir;
    }
    // Save model directory on change
    if (field.name === "models") {
      input.addEventListener("change", () => {
        const v = input.value.trim();
        if (v) saveModelDir(v);
      });
    }
    const browseBtn = document.createElement("button");
    browseBtn.className = "browse-btn";
    browseBtn.textContent = "Browse";
    browseBtn.addEventListener("click", () => openBrowser(field.name));
    wrap.appendChild(input);
    wrap.appendChild(browseBtn);
    row.appendChild(wrap);
  } else if (field.type === "select") {
    input = document.createElement("select");
    input.className = "form-select";
    input.id = `field-${field.name}`;
    for (const c of field.choices) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      if (c === field.default) opt.selected = true;
      input.appendChild(opt);
    }
    row.appendChild(input);
  } else if (field.type === "template") {
    input = document.createElement("select");
    input.className = "form-select";
    input.id = `field-${field.name}`;
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "(default: basic)";
    input.appendChild(none);
    for (const t of templates) {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      input.appendChild(opt);
    }
    row.appendChild(input);
  } else if (field.type === "number") {
    input = document.createElement("input");
    input.type = "number";
    input.className = "form-input";
    input.id = `field-${field.name}`;
    if (field.placeholder) input.placeholder = field.placeholder;
    if (field.default !== undefined) input.value = field.default;
    row.appendChild(input);
  } else if (field.type === "csv" && field.bpwPaintFlags) {
    // Tokenized BPW input with paint-mode flag highlighting
    const wrap = document.createElement("div");
    wrap.className = "bpw-token-wrap";

    input = document.createElement("input");
    input.type = "text";
    input.className = "form-input bpw-token-input";
    input.id = `field-${field.name}`;
    if (field.placeholder) input.placeholder = field.placeholder;

    const tokenDisplay = document.createElement("div");
    tokenDisplay.className = "bpw-token-display";
    tokenDisplay.id = `tokens-${field.name}`;

    // Sync: when input changes, rebuild tokens
    input.addEventListener("input", () => rebuildBpwTokens(field.name, field.bpwPaintFlags));
    input.addEventListener("change", () => rebuildBpwTokens(field.name, field.bpwPaintFlags));

    wrap.appendChild(input);
    wrap.appendChild(tokenDisplay);
    row.appendChild(wrap);
  } else {
    // text, csv
    input = document.createElement("input");
    input.type = "text";
    input.className = "form-input";
    input.id = `field-${field.name}`;
    if (field.placeholder) input.placeholder = field.placeholder;
    if (field.default !== undefined) input.value = field.default;
    // Auto-populate CUDA devices from detected GPUs
    if (field.name === "devices" && gpus.length > 0) {
      input.value = gpus.map(g => g.index).join(",");
    }
    row.appendChild(input);
  }

  // Expose field.help as a hover tooltip on the input element too
  attachHelpTooltip(input);

  // Wire up toggleable: start disabled (or enabled if defaultOn), grey out until toggled on
  if (field.toggleable) {
    const toggleCb = row.querySelector(`#toggle-${field.name}`);
    const setEnabled = (on) => {
      row.classList.toggle("field-disabled", !on);
      const inp = row.querySelector(`#field-${field.name}`);
      if (inp) inp.disabled = !on;
    };
    if (field.defaultOn) {
      setEnabled(true);
      if (toggleCb) toggleCb.checked = true;
    } else {
      setEnabled(false);
    }
    toggleCb.addEventListener("change", () => setEnabled(toggleCb.checked));
  }

  return row;
}


// ── BPW Paint Mode System ────────────────────────────────────────
// Per-BPW flag state: { fieldName: { bpwStr: Set<flagName> } }
const bpwFlagState = {};
// Global (non-paint) toggles for buttons rendered alongside paint
// buttons but applied universally instead of per-BPW: { fieldName: Set<flagName> }
const bpwGlobalState = {};

// BPW range: 1.0 to 8.0 inclusive. Below 1 is incoherent but allowed,
// above 8 isn't supported by exllamav3 and will error out.
function isValidBpw(s) {
  if (!s) return false;
  const n = Number(s);
  return Number.isFinite(n) && n >= 1 && n <= 8;
}
// Active paint mode: { fieldName, flagName } or null
let activePaint = null;

function togglePaintMode(fieldName, flagName, btn) {
  const allBtns = btn.parentElement.querySelectorAll(".bpw-paint-btn");

  if (activePaint && activePaint.fieldName === fieldName && activePaint.flagName === flagName) {
    // Deactivate current paint mode
    btn.classList.remove("active");
    activePaint = null;
  } else {
    // Deactivate any other, activate this one
    allBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    activePaint = { fieldName, flagName };
  }
}

function rebuildBpwTokens(fieldName, paintFlags) {
  const input = document.getElementById(`field-${fieldName}`);
  const display = document.getElementById(`tokens-${fieldName}`);
  if (!input || !display) return;

  const raw = input.value.replace(/\s+/g, "");
  const parts = raw.split(",").filter(Boolean);

  // Initialize state for this field if needed
  if (!bpwFlagState[fieldName]) bpwFlagState[fieldName] = {};
  // Clean up flags for BPWs no longer in the input
  const currentSet = new Set(parts);
  for (const key of Object.keys(bpwFlagState[fieldName])) {
    if (!currentSet.has(key)) delete bpwFlagState[fieldName][key];
  }

  display.innerHTML = "";

  const validParts = parts.filter(isValidBpw);
  validParts.forEach((bpw, idx) => {
    const token = document.createElement("span");
    token.className = "bpw-token";
    token.textContent = bpw;
    token.dataset.bpw = bpw;

    // Apply flag colors
    const flags = bpwFlagState[fieldName][bpw] || new Set();
    applyTokenColor(token, flags, paintFlags);

    token.addEventListener("mousedown", (e) => {
      if (jobRunning) return;
      e.preventDefault();
      onTokenClick(fieldName, bpw, paintFlags);
    });
    display.appendChild(token);

    // Add comma separator (not the last one)
    if (idx < validParts.length - 1) {
      const sep = document.createElement("span");
      sep.className = "bpw-token-sep";
      sep.textContent = ",";
      display.appendChild(sep);
    }
  });

  // Append paint buttons inline after the tokens
  if (validParts.length > 0) {
    const paintWrap = document.createElement("div");
    paintWrap.className = "bpw-paint-buttons";
    for (const pf of paintFlags) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "bpw-paint-btn";
      if (pf.isGlobal) btn.classList.add("bpw-paint-btn-global");
      btn.dataset.paintFlag = pf.name;
      btn.dataset.paintColor = pf.color;
      btn.textContent = pf.label;
      btn.style.setProperty("--paint-color", pf.color);
      if (pf.tooltip) btn.title = pf.tooltip;
      if (jobRunning) btn.disabled = true;
      if (pf.isGlobal) {
        // Global toggle (e.g. -pm) — independent on/off, not paint mode.
        if (bpwGlobalState[fieldName]?.has(pf.name)) {
          btn.classList.add("active");
        }
        btn.addEventListener("mousedown", (e) => {
          if (jobRunning) return;
          e.preventDefault();
          if (!bpwGlobalState[fieldName]) bpwGlobalState[fieldName] = new Set();
          const set = bpwGlobalState[fieldName];
          if (set.has(pf.name)) set.delete(pf.name);
          else set.add(pf.name);
          btn.classList.toggle("active");
        });
      } else {
        // Restore active state if this paint mode is currently on
        if (activePaint && activePaint.fieldName === fieldName && activePaint.flagName === pf.name) {
          btn.classList.add("active");
        }
        // Use mousedown so the click isn't eaten by the BPW input's blur
        // when the user clicks straight from typing into the entry field.
        btn.addEventListener("mousedown", (e) => {
          if (jobRunning) return;
          e.preventDefault();
          togglePaintMode(fieldName, pf.name, btn);
        });
      }
      paintWrap.appendChild(btn);
    }
    display.appendChild(paintWrap);
  }
}

function onTokenClick(fieldName, bpw, paintFlags) {
  if (!bpwFlagState[fieldName]) bpwFlagState[fieldName] = {};
  if (!bpwFlagState[fieldName][bpw]) bpwFlagState[fieldName][bpw] = new Set();

  const flags = bpwFlagState[fieldName][bpw];

  if (activePaint && activePaint.fieldName === fieldName) {
    // -opt can only be painted on fractional BPWs
    const paintDef = paintFlags.find(p => p.name === activePaint.flagName);
    if (paintDef && paintDef.fractionalOnly && !bpw.includes(".")) {
      // Silently ignore click on non-fractional BPW for fractional-only flags
      return;
    }
    // Toggle the active paint flag on this BPW
    if (flags.has(activePaint.flagName)) {
      flags.delete(activePaint.flagName);
    } else {
      flags.add(activePaint.flagName);
    }
  } else {
    // No paint mode active: clear all flags on this token
    flags.clear();
  }

  rebuildBpwTokens(fieldName, paintFlags);
}

function lighten(hex, amount) {
  // Lighten a hex color by mixing toward white
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const nr = Math.min(255, Math.round(r + (255 - r) * amount));
  const ng = Math.min(255, Math.round(g + (255 - g) * amount));
  const nb = Math.min(255, Math.round(b + (255 - b) * amount));
  return `#${nr.toString(16).padStart(2,"0")}${ng.toString(16).padStart(2,"0")}${nb.toString(16).padStart(2,"0")}`;
}

function applyTokenColor(token, flags, paintFlags) {
  // Reset
  token.style.backgroundColor = "";
  token.style.backgroundImage = "";
  token.style.color = "";
  token.style.border = "";
  token.style.outline = "";
  token.style.outlineOffset = "";
  token.classList.remove("bpw-token-flagged");

  if (flags.size === 0) return;

  token.classList.add("bpw-token-flagged");

  // Separate border-only flags (opt) from stripe flags (hq, hb8)
  const hasOpt = flags.has("opt");
  const stripeFlags = [...flags].filter(f => f !== "opt");

  // Apply red outline for -opt (additive — stacks with stripe patterns)
  if (hasOpt) {
    token.style.outline = "3px solid #d94a4a";
    token.style.outlineOffset = "-1px";
  }

  if (stripeFlags.length === 0) {
    // -opt only: red border, no stripe fill
    if (hasOpt) token.style.color = "";
    return;
  }

  if (stripeFlags.length === 1) {
    const pf = paintFlags.find(p => p.name === stripeFlags[0]);
    if (pf) {
      token.style.color = "#fff";
      // Accessibility: stripe patterns for color-impaired distinction
      // -hq = horizontal stripes, -hb8 = vertical stripes
      // 2px color, 3px black — consistent sizing, black slightly larger
      const dir = pf.name === "hq" ? "180deg" : "90deg";
      token.style.backgroundImage =
        `repeating-linear-gradient(${dir}, ${pf.color} 0px, ${pf.color} 2px, #000 2px, #000 6px)`;
    }
  } else if (stripeFlags.length >= 2) {
    // Both hq + hb8: teal/cyan with checkerboard (intersection of horizontal + vertical)
    const c = "#00897b";
    token.style.color = "#fff";
    token.style.backgroundImage =
      `repeating-linear-gradient(180deg, ${c} 0px, ${c} 2px, transparent 2px, transparent 6px), ` +
      `repeating-linear-gradient(90deg, ${c} 0px, ${c} 2px, transparent 2px, transparent 6px)`;
    token.style.backgroundColor = "#000";
  }
}

function getBpwFlags(fieldName) {
  return bpwFlagState[fieldName] || {};
}


function syncFormLockState() {
  const container = document.getElementById("form-fields");
  if (!container) return;

  // Disable/enable all form inputs
  for (const el of container.querySelectorAll("input, select, button.browse-btn")) {
    el.disabled = jobRunning;
  }
  // Disable/enable all toggle sliders
  for (const el of container.querySelectorAll(".toggle-item input, .field-toggle input")) {
    el.disabled = jobRunning;
  }
  // Disable/enable paint buttons
  for (const btn of document.querySelectorAll(".bpw-paint-btn")) {
    btn.disabled = jobRunning;
  }
  // Disable/enable token clicks via CSS class
  for (const display of document.querySelectorAll(".bpw-token-display")) {
    display.classList.toggle("locked", jobRunning);
  }
  // Visual dim
  container.classList.toggle("form-locked", jobRunning);
}


function createToggleEl(field) {
  const item = document.createElement("label");
  item.className = "toggle-item";

  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.id = `field-${field.name}`;
  item.appendChild(cb);

  const slider = document.createElement("span");
  slider.className = "toggle-slider";
  item.appendChild(slider);

  const text = document.createElement("span");
  text.className = "toggle-label-text";
  text.textContent = field.label;
  if (field.help) text.title = field.help;
  item.appendChild(text);

  return item;
}


function collectArgs() {
  const cmd = COMMANDS[activeCommand];
  if (!cmd) return [];

  const args = [];
  const missing = [];

  for (const field of cmd.fields) {
    const el = document.getElementById(`field-${field.name}`);
    if (!el) continue;

    if (field.type === "boolean") {
      if (field.invertFlag) {
        // Inverted: emit flag when UNchecked (e.g. --no-kl when KL toggle is off)
        if (!el.checked) args.push(field.flag);
      } else {
        if (el.checked) args.push(field.flag);
      }
      continue;
    }

    // Skip toggleable fields that are toggled off
    if (field.toggleable) {
      const toggleCb = document.getElementById(`toggle-${field.name}`);
      if (!toggleCb || !toggleCb.checked) continue;
    }

    const val = el.value.trim();

    if (field.required && !val) {
      missing.push(field.label);
      el.classList.add("input-error");
      continue;
    }
    el.classList.remove("input-error");

    if (!val) {
      // Toggleable fields with no value still emit the flag (e.g. -cb defaults to 3)
      if (field.toggleable) args.push(field.flag);
      continue;
    }

    if (field.type === "csv") {
      // Normalize: strip spaces around commas so "-d 0, 1" becomes "-d 0,1"
      let csv = val.replace(/\s*,\s*/g, ",").replace(/\s+/g, ",");
      // For BPW fields, drop out-of-range values (1-8 only)
      if (field.bpwPaintFlags) {
        csv = csv.split(",").filter(v => isValidBpw(v)).join(",");
        if (!csv) continue;
      }
      args.push(field.flag, csv);
      // Emit per-BPW paint flags (e.g. -hq 4,6 -hb8 8)
      // and collect any global toggles (e.g. -pm) to append at the end.
      if (field.bpwPaintFlags) {
        const flagState = getBpwFlags(field.name);
        const globalSet = bpwGlobalState[field.name] || new Set();
        for (const pf of field.bpwPaintFlags) {
          if (pf.isGlobal) {
            if (globalSet.has(pf.name)) args.push(pf.flag);
            continue;
          }
          const flagged = Object.entries(flagState)
            .filter(([, flags]) => flags.has(pf.name))
            .map(([bpw]) => bpw);
          if (flagged.length > 0) {
            args.push(pf.flag, flagged.join(","));
          }
        }
      }
    } else if (field.type === "select" || field.type === "template") {
      if (val) {
        args.push(field.flag, val);
      }
    } else if (field.type === "number") {
      args.push(field.flag, val);
    } else {
      // text, path
      args.push(field.flag, val);
    }
  }

  if (missing.length) {
    return { error: `Required: ${missing.join(", ")}` };
  }
  return args;
}


function saveModelDir(dir) {
  savedConfig.last_model_dir = dir;
  fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ last_model_dir: dir }),
  }).catch(() => {});
}


// ── Metadata panel (README fields with lock buttons) ─────────────

const META_FIELDS = [
  { key: "AUTHOR", label: "Author", help: "Model author / organization" },
  { key: "MODEL", label: "Model Name", help: "Base model name" },
  { key: "REPOLINK", label: "Repo Link", help: "Link to original model" },
  { key: "USER", label: "Quantized By", help: "Your HuggingFace username" },
];

function renderMetadataPanel(commandKey) {
  const panel = document.getElementById("metadata-panel");
  const cmd = COMMANDS[commandKey];

  if (!cmd || !cmd.hasMetadata) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "";
  // Preserve waiting state across re-renders
  if (metadataWaitingDir) {
    panel.classList.add("metadata-waiting");
    document.getElementById("metadata-confirm").style.display = "";
  }

  const container = document.getElementById("metadata-fields");
  container.innerHTML = "";

  for (const f of META_FIELDS) {
    const div = document.createElement("div");
    div.className = "meta-field";
    div.dataset.key = f.key;

    const row = document.createElement("div");
    row.className = "meta-field-row";

    const label = document.createElement("label");
    label.className = "meta-label";
    label.textContent = f.label;
    row.appendChild(label);
    div.appendChild(row);

    if (f.help) {
      const help = document.createElement("div");
      help.className = "meta-help";
      help.textContent = f.help;
      div.appendChild(help);
    }

    // Input + lock button side by side
    const inputRow = document.createElement("div");
    inputRow.className = "meta-input-row";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "meta-input";
    input.id = `meta-${f.key}`;
    input.addEventListener("input", updateMetadataConfirm);
    inputRow.appendChild(input);

    const lockBtn = document.createElement("button");
    lockBtn.type = "button";
    lockBtn.className = "meta-lock";
    lockBtn.dataset.key = f.key;
    lockBtn.title = "Lock this field";
    lockBtn.textContent = "\u{1F513}";  // unlocked
    lockBtn.addEventListener("click", () => toggleMetaLock(f.key));
    inputRow.appendChild(lockBtn);

    div.appendChild(inputRow);
    container.appendChild(div);
  }

  // Wire model dir changes to reload defaults
  const modelInput = document.getElementById("field-models");
  if (modelInput) {
    modelInput.addEventListener("change", () => loadMetadataDefaults(true));
  }

  // Apply run-locked state if a job is already running
  syncMetaLockState();
  updateMetadataConfirm();
  loadMetadataDefaults(false);
}


function toggleMetaLock(key) {
  const lockBtn = document.querySelector(`.meta-lock[data-key="${key}"]`);
  if (!lockBtn) return;

  const field = lockBtn.closest(".meta-field");
  const input = document.getElementById(`meta-${key}`);
  const isLocked = field.classList.contains("locked");

  if (isLocked) {
    // Only prevent unlocking while the README is actively being written.
    // During quant/measure/etc. the user can freely lock & unlock.
    if (metadataFrozen) return;
    // Unlock
    field.classList.remove("locked");
    lockBtn.classList.remove("locked");
    lockBtn.textContent = "\u{1F513}";  // unlocked
    lockBtn.title = "Lock this field";
    if (input) input.readOnly = false;
  } else {
    // Lock
    field.classList.add("locked");
    lockBtn.classList.add("locked");
    lockBtn.textContent = "\u{1F512}";  // locked
    lockBtn.title = "Unlock this field";
    if (input) input.readOnly = true;
  }

  // Sync lock state to disk so backend always sees current UI state
  saveMetadata();
  updateMetadataConfirm();
}


function syncMetaLockState() {
  // Visually mark locked fields as frozen only during the README write
  // window (between Resume click and README_DONE). Outside that window
  // — even during the rest of the run — users can freely toggle locks.
  const locks = document.querySelectorAll(".meta-lock.locked");
  for (const btn of locks) {
    btn.classList.toggle("run-locked", metadataFrozen);
  }
}


async function loadMetadataDefaults(force) {
  const modelDir = getModelDir();
  if (!modelDir) return;

  try {
    const res = await fetch(`/api/metadata?model_dir=${encodeURIComponent(modelDir)}`);
    if (!res.ok) return;
    const data = await res.json();
    for (const f of META_FIELDS) {
      const input = document.getElementById(`meta-${f.key}`);
      if (!input) continue;
      // Don't overwrite locked fields
      const field = input.closest(".meta-field");
      if (field && field.classList.contains("locked")) continue;
      if (data[f.key] && (!input.value || force)) {
        input.value = data[f.key];
      }
    }
    updateMetadataConfirm();
    // Sync on-disk state with current UI (clears stale _locked from previous sessions)
    saveMetadata();
  } catch (e) { /* non-critical */ }
}


async function saveMetadata() {
  const dir = getModelDir() || metadataWaitingDir;
  if (!dir) return;

  const meta = { model_dir: dir };
  const locked = {};
  for (const f of META_FIELDS) {
    const input = document.getElementById(`meta-${f.key}`);
    const field = document.querySelector(`.meta-field[data-key="${f.key}"]`);
    meta[f.key] = input ? input.value.trim() : "";
    locked[f.key] = !!(field && field.classList.contains("locked"));
  }
  meta._locked = locked;

  try {
    await fetch("/api/metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(meta),
    });
  } catch (e) { /* non-critical */ }
}


async function confirmMetadata() {
  // Save with _confirm flag to clear the _waiting state on disk
  const dir = getModelDir() || metadataWaitingDir;
  if (dir) {
    const meta = { model_dir: dir, _confirm: true };
    const locked = {};
    for (const f of META_FIELDS) {
      const input = document.getElementById(`meta-${f.key}`);
      const field = document.querySelector(`.meta-field[data-key="${f.key}"]`);
      meta[f.key] = input ? input.value.trim() : "";
      locked[f.key] = !!(field && field.classList.contains("locked"));
    }
    meta._locked = locked;
    try {
      await fetch("/api/metadata", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(meta),
      });
    } catch (e) { /* non-critical */ }
  }

  const panel = document.getElementById("metadata-panel");
  panel.classList.remove("metadata-waiting");
  document.getElementById("metadata-confirm").style.display = "none";
  metadataWaitingDir = null;
  // Backend will start writing the README momentarily — freeze the locks
  // until the README_DONE marker comes back through the stream.
  metadataFrozen = true;
  syncMetaLockState();
}


function updateMetadataConfirm() {
  const allLocked = META_FIELDS.every(f => {
    const field = document.querySelector(`.meta-field[data-key="${f.key}"]`);
    const input = document.getElementById(`meta-${f.key}`);
    return field && field.classList.contains("locked") && input && input.value.trim();
  });

  const btn = document.getElementById("metadata-confirm");
  if (btn) btn.disabled = !allLocked;

  // Pre-fill mode: auto-save when all locked and not waiting for pipeline
  if (allLocked && !metadataWaitingDir) {
    saveMetadata();
  }
}


function renderEvalsPanel(commandKey) {
  const panel = document.getElementById("evals-panel");
  const container = document.getElementById("evals-fields");
  container.innerHTML = "";

  const cmd = COMMANDS[commandKey];
  const evals = cmd ? cmd.fields.filter(f => f.section === "evals") : [];

  if (!evals.length) {
    panel.style.display = "none";
    return;
  }

  panel.style.display = "";

  for (const field of evals) {
    // Compact: omit help text in side panel
    const compactField = Object.assign({}, field, { help: null });
    if (field.type === "boolean") {
      Object.assign(compactField, { toggleable: true });
    }
    container.appendChild(createFieldEl(compactField));
  }
}


function showMetadataWait(modelDir) {
  metadataWaitingDir = modelDir;
  // If we ever get back here (e.g. README retry), make sure the freeze
  // window is considered closed — the pipeline is waiting for input now.
  metadataFrozen = false;

  // If current command doesn't have a metadata panel, switch to readme
  if (!COMMANDS[activeCommand]?.hasMetadata) {
    selectCommand("readme");
  }

  switchTab("command");

  const panel = document.getElementById("metadata-panel");
  if (panel) {
    panel.style.display = "";
    panel.classList.add("metadata-waiting");
  }

  // Allow lock toggles even though a job is running (user needs to lock fields)
  const locks = document.querySelectorAll(".meta-lock");
  for (const btn of locks) {
    btn.classList.remove("run-locked");
  }

  const btn = document.getElementById("metadata-confirm");
  if (btn) {
    btn.style.display = "";
    btn.addEventListener("click", confirmMetadata);
  }

  // Populate with defaults the subprocess wrote
  loadMetadataDefaults(true);
  updateMetadataConfirm();
}
