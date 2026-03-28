// ── File/directory browser ───────────────────────────────────────

let browserTargetField = null;

function openBrowser(fieldName) {
  browserTargetField = fieldName;
  const modal = document.getElementById("browser-modal");
  modal.style.display = "flex";

  // Start from current field value or home
  const input = document.getElementById(`field-${fieldName}`);
  const startPath = input && input.value.trim() ? input.value.trim() : "";
  browseTo(startPath);
}

function closeBrowser() {
  document.getElementById("browser-modal").style.display = "none";
  browserTargetField = null;
}

function selectBrowserPath() {
  const pathEl = document.getElementById("browser-current-path");
  if (browserTargetField && pathEl) {
    const input = document.getElementById(`field-${browserTargetField}`);
    if (input) {
      input.value = pathEl.textContent;
      input.classList.remove("input-error");
    }
  }
  closeBrowser();
}

async function browseTo(path) {
  const list = document.getElementById("browser-entries");
  const pathEl = document.getElementById("browser-current-path");
  const pathInput = document.getElementById("browser-path-input");
  const selectBtn = document.getElementById("browser-select-btn");
  const indicator = document.getElementById("browser-indicator");

  list.innerHTML = '<div class="browser-loading">Loading...</div>';

  try {
    const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
    const res = await fetch(url);
    const data = await res.json();

    if (data.error) {
      list.innerHTML = `<div class="browser-error">${data.error}</div>`;
      return;
    }

    pathEl.textContent = data.current;
    pathInput.value = data.current;

    // Model indicator
    if (data.is_model) {
      indicator.textContent = "Valid model directory (config.json found)";
      indicator.className = "browser-indicator model-valid";
      indicator.style.display = "";
      selectBtn.disabled = false;
    } else {
      indicator.textContent = "";
      indicator.style.display = "none";
      selectBtn.disabled = false; // Allow selecting any directory
    }

    list.innerHTML = "";

    // Parent directory
    if (data.parent) {
      const parentEl = document.createElement("div");
      parentEl.className = "browser-entry browser-dir";
      parentEl.textContent = "..";
      parentEl.addEventListener("click", () => browseTo(data.parent));
      list.appendChild(parentEl);
    }

    for (const entry of data.entries) {
      const el = document.createElement("div");
      if (entry.type === "dir") {
        el.className = "browser-entry browser-dir";
        el.textContent = entry.name + "/";
        el.addEventListener("click", () => browseTo(data.current + "/" + entry.name));
      } else {
        el.className = "browser-entry browser-file";
        el.textContent = entry.name;
      }
      list.appendChild(el);
    }

    if (!data.entries.length && !data.parent) {
      list.innerHTML = '<div class="browser-empty">Empty directory</div>';
    }
  } catch (e) {
    list.innerHTML = `<div class="browser-error">Error: ${e.message}</div>`;
  }
}
