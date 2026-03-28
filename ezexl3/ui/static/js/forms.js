// ── Dynamic form rendering from command schemas ─────────────────

function renderForm(commandKey) {
  const cmd = COMMANDS[commandKey];
  if (!cmd) return;

  const container = document.getElementById("form-fields");
  container.innerHTML = "";

  const required = cmd.fields.filter(f => f.required && f.type !== "boolean");
  const optional = cmd.fields.filter(f => !f.required && f.type !== "boolean");
  const booleans = cmd.fields.filter(f => f.type === "boolean");

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
}


function createFieldEl(field) {
  const row = document.createElement("div");
  row.className = "form-row";

  const label = document.createElement("label");
  label.className = "form-label";
  label.textContent = field.label;
  if (field.required) {
    const star = document.createElement("span");
    star.className = "required-star";
    star.textContent = " *";
    label.appendChild(star);
  }
  row.appendChild(label);

  if (field.help) {
    const help = document.createElement("div");
    help.className = "form-help";
    help.textContent = field.help;
    row.appendChild(help);
  }

  let input;

  if (field.type === "path") {
    const wrap = document.createElement("div");
    wrap.className = "path-input-wrap";
    input = document.createElement("input");
    input.type = "text";
    input.className = "form-input";
    input.id = `field-${field.name}`;
    input.placeholder = "/path/to/model";
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

    if (field.type === "boolean") {
      if (el.checked) {
        args.push(field.flag);
      }
      continue;
    }

    const val = el.value.trim();

    if (field.required && !val) {
      missing.push(field.label);
      el.classList.add("input-error");
      continue;
    }
    el.classList.remove("input-error");

    if (!val) continue;

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
