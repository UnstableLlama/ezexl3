import csv
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional

from ezexl3.graph_svg import generate_iceblink_svg

_META_FILENAME = ".ezexl3_readme_meta.json"
_META_KEYS = ("AUTHOR", "MODEL", "REPOLINK", "USER")


def get_hf_username() -> str:
    """Try to get huggingface username from huggingface-cli."""
    try:
        import subprocess
        result = subprocess.run(["huggingface-cli", "whoami"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.splitlines()[0].strip()
    except Exception:
        pass
    return os.environ.get("USER", "USER")


def _metadata_path(model_dir: str) -> str:
    return os.path.join(model_dir, _META_FILENAME)


def _read_saved_metadata(model_dir: str) -> Optional[Dict[str, str]]:
    path = _metadata_path(model_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_saved_metadata(model_dir: str, meta: Dict) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(_metadata_path(model_dir), "w") as f:
        json.dump(meta, f, indent=2)


def _compute_defaults(model_dir: str) -> Dict[str, str]:
    """Compute sensible default metadata from the model directory name."""
    model_name = os.path.basename(os.path.abspath(model_dir))
    parts = model_name.split("-", 1)
    author = parts[0] if len(parts) > 1 else "AUTHOR"
    model = parts[1] if len(parts) > 1 else model_name
    user = get_hf_username()
    repolink = f"https://huggingface.co/{author}/{model}"
    return {"AUTHOR": author, "MODEL": model, "REPOLINK": repolink, "USER": user}


def _wait_for_dashboard_metadata(model_dir: str, defaults: Dict[str, str]) -> Dict[str, str]:
    """Write defaults with a waiting flag and poll until the dashboard confirms."""
    waiting_meta = dict(defaults)
    waiting_meta["_waiting"] = True
    _write_saved_metadata(model_dir, waiting_meta)

    print(f"\n<<EZEXL3:WAITING_METADATA:{model_dir}>>")
    print("⏳ Waiting for README metadata from dashboard...")
    print("   Review the metadata fields, lock all four, and click 'Resume'")
    sys.stdout.flush()

    poll_count = 0
    while True:
        time.sleep(1)
        poll_count += 1
        if poll_count % 30 == 0:
            print("⏳ Still waiting for metadata...")
            sys.stdout.flush()
        saved = _read_saved_metadata(model_dir)
        if saved and not saved.get("_waiting"):
            print("📝 Metadata received from dashboard")
            result = {k: saved.get(k, defaults.get(k, "")) for k in _META_KEYS}
            result["QUANT_METHOD"] = "exl3"
            result["QUANT_TOOL"] = "exllamav3"
            return result


def _all_fields_locked(saved: Dict) -> bool:
    """Check if all metadata fields were locked in the dashboard."""
    locked = saved.get("_locked", {})
    return all(locked.get(k) for k in _META_KEYS)


def prompt_metadata(model_dir: str, bpws: List[str], interactive: bool = True) -> Dict[str, str]:
    """Collect README metadata from saved file, dashboard, or interactive prompts."""
    defaults = _compute_defaults(model_dir)

    # Check for saved metadata — only auto-use if ALL fields are locked
    saved = _read_saved_metadata(model_dir)
    if (saved and not saved.get("_waiting")
            and all(saved.get(k) for k in _META_KEYS)
            and _all_fields_locked(saved)):
        print(f"📝 Using saved README metadata from {_META_FILENAME}")
        result = {k: saved[k] for k in _META_KEYS}
        result["QUANT_METHOD"] = "exl3"
        result["QUANT_TOOL"] = "exllamav3"
        return result

    if not interactive:
        defaults["QUANT_METHOD"] = "exl3"
        defaults["QUANT_TOOL"] = "exllamav3"
        return defaults

    # Non-TTY (dashboard subprocess): wait for metadata via file
    if not sys.stdin.isatty():
        # Use saved values as starting point if they exist
        starting = {k: (saved or defaults).get(k, defaults[k]) for k in _META_KEYS}
        return _wait_for_dashboard_metadata(model_dir, starting)

    # Interactive TTY: prompt user
    print("\n📝 Please provide metadata for the README (ENTER to use defaults):")

    author = input(f"Author [{defaults['AUTHOR']}]: ").strip() or defaults["AUTHOR"]
    model = input(f"Model [{defaults['MODEL']}]: ").strip() or defaults["MODEL"]
    repolink = input(f"Repo Link [{defaults['REPOLINK']}]: ").strip() or defaults["REPOLINK"]
    user = input(f"Quantized By (HuggingFace Username) [{defaults['USER']}]: ").strip() or defaults["USER"]

    return {
        "AUTHOR": author,
        "MODEL": model,
        "REPOLINK": repolink,
        "USER": user,
        "QUANT_METHOD": "exl3",
        "QUANT_TOOL": "exllamav3",
    }


def _discover_rows_without_measurements(model_dir: str, bpws_hint: Optional[List[str]] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    bpws: List[str] = []
    if bpws_hint:
        bpws.extend([str(x) for x in bpws_hint])

    if os.path.isdir(model_dir):
        for item in os.listdir(model_dir):
            path = os.path.join(model_dir, item)
            if not os.path.isdir(path):
                continue
            if item.startswith("w-"):
                continue
            try:
                float(item)
                bpws.append(item)
            except Exception:
                continue

    seen = set()
    ordered_bpws: List[str] = []
    for b in bpws:
        if b in seen:
            continue
        seen.add(b)
        ordered_bpws.append(b)

    def _bpw_order(v: str) -> float:
        try:
            return float(v)
        except Exception:
            return 9999.0

    for b in sorted(ordered_bpws, key=_bpw_order):
        rows.append({"weights": b, "GiB": "x", "KL Div": "x", "PPL r-100": "x"})

    rows.append({"weights": "bf16", "GiB": "x", "KL Div": "x", "PPL r-100": "x"})
    return rows


def _build_catbench_grid(model_dir: str) -> str:
    """Build an HTML table grid of catbench SVG thumbnails.

    Scans {model_dir}/catbench/ for canonical SVGs (e.g. 2.00bpw.svg, bf16.svg)
    and arranges them in rows of 4, matching turboderp's format.
    """
    catbench_dir = os.path.join(model_dir, "catbench")
    if not os.path.isdir(catbench_dir):
        return ""

    # Find canonical SVGs (not _1, _2 variants)
    svgs: List[tuple] = []  # (sort_key, label, filename)
    for fn in os.listdir(catbench_dir):
        if not fn.endswith(".svg"):
            continue
        # Skip numbered variants like 2.00bpw_1.svg
        if re.search(r"_\d+\.svg$", fn):
            continue

        if fn == "bf16.svg":
            svgs.append((9999.0, "BF16", fn))
        elif fn.endswith("bpw.svg"):
            bpw_str = fn.replace("bpw.svg", "")
            try:
                val = float(bpw_str)
                svgs.append((val, f"{val:.2f} bpw", fn))
            except ValueError:
                continue

    if not svgs:
        return ""

    svgs.sort(key=lambda x: x[0])

    cols_per_row = 4
    rows_html = []

    for i in range(0, len(svgs), cols_per_row):
        cells = []
        for _, label, fn in svgs[i:i + cols_per_row]:
            rel_path = f"catbench/{fn}"
            cells.append(
                f'    <td align="center">\n'
                f'      <a href="{rel_path}">\n'
                f'        <img src="{rel_path}" alt="{label}" width="160">\n'
                f'      </a>\n'
                f'      <div>{label}</div>\n'
                f'    </td>'
            )
        rows_html.append("  <tr>\n" + "\n".join(cells) + "\n  </tr>")

    return '<table align="center">\n' + "\n".join(rows_html) + "\n</table>"


def run_readme(
    model_dir: str,
    template_name: Optional[str] = None,
    interactive: bool = True,
    include_graph: bool = True,
    include_measurements: bool = True,
    bpws_hint: Optional[List[str]] = None,
    include_catbench: bool = False,
) -> None:
    """
    Generate README.md for the model repository based on measurement CSV and template.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(pkg_dir, "templates")

    if not template_name:
        template_name = "basic"

    possible_names = [
        template_name,
        f"{template_name}.md",
    ]

    lookup_names: List[str] = []
    for name in possible_names:
        if name not in lookup_names:
            lookup_names.append(name)

    base_name = template_name
    if base_name.endswith(".md"):
        base_name = base_name[:-3]

    if not base_name.endswith("TemplateREADME"):
        lookup_names.append(f"{base_name}TemplateREADME.md")
    if not base_name.endswith("README"):
        lookup_names.append(f"{base_name}README.md")
    if not base_name.endswith("Template"):
        lookup_names.append(f"{base_name}Template.md")

    template_path = None
    for name in lookup_names:
        path = os.path.join(templates_dir, name)
        if os.path.exists(path) and os.path.isfile(path):
            template_path = path
            break

    if not template_path:
        print(f"🔴 Template not found in {templates_dir} for '{template_name}'")
        print(f"   Tried: {', '.join(lookup_names)}")
        return

    with open(template_path, "r") as f:
        template = f.read()

    rows: List[Dict[str, str]] = []
    if include_measurements:
        from ezexl3.measure import default_csv_path

        csv_path = default_csv_path(model_dir)
        if not os.path.exists(csv_path):
            print(f"🔴 CSV not found: {csv_path}. Cannot generate README.")
            return

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            print(f"🔴 CSV is empty: {csv_path}. Cannot generate README.")
            return
    else:
        rows = _discover_rows_without_measurements(model_dir, bpws_hint=bpws_hint)

    def sort_key(r):
        w = r["weights"]
        if w == "bf16":
            return 100.0
        try:
            return float(w)
        except Exception:
            return 200.0

    rows.sort(key=sort_key)

    bpws = [r["weights"] for r in rows if r["weights"] != "bf16"]

    meta = prompt_metadata(model_dir, bpws, interactive=interactive)

    formatted_labels: Dict[str, str] = {}
    first_bpw = None

    for r in rows:
        w = r["weights"]
        if w == "bf16":
            formatted_labels[w] = "bf16"
        else:
            try:
                val = float(w)
                label = f"{val:.2f}bpw"
                formatted_labels[w] = label
                if first_bpw is None:
                    first_bpw = label
            except Exception:
                formatted_labels[w] = w
                
    quant_repo_link = f"https://huggingface.co/{meta['USER']}/{meta['MODEL']}-{meta['QUANT_METHOD']}"

    table_rows = []
    for r in rows:
        w = r["weights"]
        label = formatted_labels[w]

        gib = r.get("GiB", "x")
        try:
            gib = f"{float(gib):.2f}"
        except Exception:
            pass

        kl = r.get("KL Div", "x")
        try:
            kl = f"{float(kl):.4f}"
        except Exception:
            pass

        ppl = r.get("PPL r-100", "x")
        try:
            ppl = f"{float(ppl):.4f}"
        except Exception:
            pass
          
        if w == "bf16":
            revision_link = meta["REPOLINK"].rstrip("/")
        else:
            revision_link = f"{quant_repo_link.rstrip('/')}/tree/{label}"

        if include_measurements:
            row_html = f"""            <tr>
              <td><a class=\"link-style\" href=\"{revision_link}\">{label}</a></td>
              <td>{gib}</td>
              <td>{kl}</td>
              <td>{ppl}</td>
            </tr>"""
        else:
            row_html = f"""            <tr>
              <td><a class=\"link-style\" href=\"{revision_link}\">{label}</a></td>
              <td>{gib}</td>
            </tr>"""
        table_rows.append(row_html)

    table_body = "\n".join(table_rows)
    template = re.sub(r"<tbody>.*?</tbody>", f"<tbody>\n{table_body}\n          </tbody>", template, flags=re.DOTALL)

    if include_measurements:
        table_head = """          <thead>
            <tr>
              <th>REVISION</th>
              <th>GiB</th>
              <th>KL DIV</th>
              <th>PPL</th>
            </tr>
          </thead>"""
    else:
        table_head = """          <thead>
            <tr>
              <th>REVISION</th>
              <th>GiB</th>
            </tr>
          </thead>"""
    template = re.sub(r"<thead>.*?</thead>", table_head, template, flags=re.DOTALL)

    if include_graph:
        graph_filename = f"{os.path.basename(os.path.abspath(model_dir)).lower()}.svg"
        graph_path = os.path.join(model_dir, graph_filename)
        try:
            from ezexl3.measure import default_csv_path
            generate_iceblink_svg(csv_path=default_csv_path(model_dir), out_svg=graph_path, title=f"{meta['MODEL']}-{meta['QUANT_METHOD']}")
        except Exception as e:
            print(f"⚠️ Graph generation skipped: {e}")
        meta["GRAPH_FILE"] = graph_filename
    else:
        template = re.sub(r"\s*<img class=\"repo-graph\"[^>]*>\s*", "\n", template)
        meta["GRAPH_FILE"] = ""

    for k, v in meta.items():
        template = template.replace(f"{{{{{k}}}}}", str(v))

    default_rev = first_bpw or formatted_labels.get("bf16", "REVISION")
    template = template.replace("{{DEFAULT_REVISION}}", default_rev)

    # Fill or remove the SVG Catbench panel (defined in templates)
    _catbench_panel_re = (
        r'\s*<div class="content-panel">\s*'
        r'<div class="panel-title">SVG Catbench</div>.*?'
        r'\{\{CATBENCH_CONTENT\}\}.*?</div>\s*</div>'
    )
    if include_catbench:
        catbench_html = _build_catbench_grid(model_dir)
        if catbench_html:
            template = template.replace("{{CATBENCH_CONTENT}}", catbench_html)
        else:
            template = re.sub(_catbench_panel_re, "", template, flags=re.DOTALL)
    else:
        template = re.sub(_catbench_panel_re, "", template, flags=re.DOTALL)

    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(template)

    print(f"✅ Generated {readme_path}")
