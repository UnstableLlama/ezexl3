import csv
import os
import re
from typing import List, Dict, Optional

from ezexl3.graph_svg import generate_iceblink_svg


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


def prompt_metadata(model_dir: str, bpws: List[str], interactive: bool = True) -> Dict[str, str]:
    """Interactively prompt user for README metadata with smart defaults."""
    model_name = os.path.basename(os.path.abspath(model_dir))

    parts = model_name.split("-", 1)
    default_author = parts[0] if len(parts) > 1 else "AUTHOR"
    default_model = parts[1] if len(parts) > 1 else model_name

    default_user = get_hf_username()
    default_repolink = f"https://huggingface.co/{default_author}/{default_model}"

    if not interactive:
        return {
            "AUTHOR": default_author,
            "MODEL": default_model,
            "REPOLINK": default_repolink,
            "USER": default_user,
            "QUANT_METHOD": "exl3",
            "QUANT_TOOL": "exllamav3",
        }

    print("\n📝 Please provide metadata for the README (ENTER to use defaults):")

    author = input(f"Author [{default_author}]: ").strip() or default_author
    model = input(f"Model [{default_model}]: ").strip() or default_model

    repolink = input(f"Repo Link [{default_repolink}]: ").strip() or default_repolink

    user = input(f"Quantized By (HuggingFace Username) [{default_user}]: ").strip() or default_user

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


def run_readme(
    model_dir: str,
    template_name: Optional[str] = None,
    interactive: bool = True,
    include_graph: bool = True,
    include_measurements: bool = True,
    bpws_hint: Optional[List[str]] = None,
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

    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(template)

    print(f"✅ Generated {readme_path}")
