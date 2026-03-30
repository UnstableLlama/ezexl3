// ── Dynamic form rendering from command schemas ─────────────────

function renderForm(commandKey) {
  const cmd = COMMANDS[commandKey];
  if (!cmd) return;

  const container = document.getElementById("form-fields");
  container.innerHTML = "";

  const required = cmd.fields.filter(f => f.required && f.type !== "boolean");
  const optional = cmd.fields.filter(f => !f.required && f.type !== "boolean" || (f.type === "boolean" && f.toggleable));
  const booleans = cmd.fields.filter(f => f.type === "boolean" && !f.toggleable);

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

  // Metadata panel for commands that generate READMEs
  renderMetadataPanel(commandKey);
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
  if (field.headerToggle) {
    const ht = field.headerToggle;
    const wrap = document.createElement("label");
    wrap.className = "header-toggle";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `field-${ht.name}`;
    wrap.appendChild(cb);
    const slider = document.createElement("span");
    slider.className = "toggle-slider";
    wrap.appendChild(slider);
    const text = document.createElement("span");
    text.className = "header-toggle-label";
    text.textContent = ht.label;
    wrap.appendChild(text);
    labelRow.appendChild(wrap);
  }
  row.appendChild(labelRow);

  if (field.help) {
    const help = document.createElement("div");
    help.className = "form-help";
    help.textContent = field.help;
    row.appendChild(help);
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

  // Wire up toggleable: start disabled, grey out until toggled on
  if (field.toggleable) {
    const toggleCb = row.querySelector(`#toggle-${field.name}`);
    const setEnabled = (on) => {
      row.classList.toggle("field-disabled", !on);
      const inp = row.querySelector(`#field-${field.name}`);
      if (inp) inp.disabled = !on;
    };
    setEnabled(false);
    toggleCb.addEventListener("change", () => setEnabled(toggleCb.checked));
  }

  return row;
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

    // Collect headerToggle if checked
    if (field.headerToggle) {
      const htEl = document.getElementById(`field-${field.headerToggle.name}`);
      if (htEl && htEl.checked) {
        args.push(field.headerToggle.flag);
      }
    }

    if (field.type === "boolean") {
      if (el.checked) {
        args.push(field.flag);
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
      args.push(field.flag, val.replace(/\s*,\s*/g, ",").replace(/\s+/g, ","));
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


// ── Metadata panel (README fields with pin checkboxes) ───────────

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

    const pin = document.createElement("label");
    pin.className = "meta-pin";
    pin.title = "Pin this field";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "meta-pin-cb";
    cb.dataset.key = f.key;
    cb.addEventListener("change", () => {
      div.classList.toggle("pinned", cb.checked);
      updateMetadataConfirm();
    });
    const icon = document.createElement("span");
    icon.className = "meta-pin-icon";
    icon.innerHTML = "&#x1F4CC;";
    pin.appendChild(cb);
    pin.appendChild(icon);
    row.appendChild(pin);
    div.appendChild(row);

    if (f.help) {
      const help = document.createElement("div");
      help.className = "meta-help";
      help.textContent = f.help;
      div.appendChild(help);
    }

    const input = document.createElement("input");
    input.type = "text";
    input.className = "meta-input";
    input.id = `meta-${f.key}`;
    input.addEventListener("input", updateMetadataConfirm);
    div.appendChild(input);

    container.appendChild(div);
  }

  // Wire model dir changes to reload defaults
  const modelInput = document.getElementById("field-models");
  if (modelInput) {
    modelInput.addEventListener("change", () => loadMetadataDefaults(true));
  }

  updateMetadataConfirm();
  loadMetadataDefaults(false);
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
      if (input && data[f.key]) {
        // Only overwrite if empty or force-refreshing
        if (!input.value || force) {
          input.value = data[f.key];
        }
      }
    }
  } catch (e) { /* non-critical */ }
}


async function saveMetadata() {
  const dir = getModelDir() || metadataWaitingDir;
  if (!dir) return;

  const meta = { model_dir: dir };
  for (const f of META_FIELDS) {
    const input = document.getElementById(`meta-${f.key}`);
    meta[f.key] = input ? input.value.trim() : "";
  }

  try {
    await fetch("/api/metadata", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(meta),
    });
  } catch (e) { /* non-critical */ }
}


async function confirmMetadata() {
  await saveMetadata();

  const panel = document.getElementById("metadata-panel");
  panel.classList.remove("metadata-waiting");
  document.getElementById("metadata-confirm").style.display = "none";
  metadataWaitingDir = null;
}


function updateMetadataConfirm() {
  const allPinned = META_FIELDS.every(f => {
    const cb = document.querySelector(`.meta-pin-cb[data-key="${f.key}"]`);
    const input = document.getElementById(`meta-${f.key}`);
    return cb && cb.checked && input && input.value.trim();
  });

  const btn = document.getElementById("metadata-confirm");
  if (btn) btn.disabled = !allPinned;

  // Pre-fill mode: auto-save when all pinned and not waiting for pipeline
  if (allPinned && !metadataWaitingDir) {
    saveMetadata();
  }
}


function showMetadataWait(modelDir) {
  metadataWaitingDir = modelDir;

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

  const btn = document.getElementById("metadata-confirm");
  if (btn) {
    btn.style.display = "";
    btn.addEventListener("click", confirmMetadata);
  }

  // Populate with defaults the subprocess wrote
  loadMetadataDefaults(true);
  updateMetadataConfirm();
}
