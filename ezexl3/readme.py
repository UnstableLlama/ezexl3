import csv
import os
import sys
import re
from typing import List, Dict, Any, Optional

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

def prompt_metadata(model_dir: str, bpws: List[str]) -> Dict[str, str]:
    """Interactively prompt user for README metadata with smart defaults."""
    model_name = os.path.basename(os.path.abspath(model_dir))
    
    # Try to guess author and model from path (e.g., meta-llama-Llama-3.2-1B-Instruct)
    # Common pattern is author-model
    parts = model_name.split("-", 1)
    default_author = parts[0] if len(parts) > 1 else "AUTHOR"
    default_model = parts[1] if len(parts) > 1 else model_name
    
    print(f"\n📝 Please provide metadata for the README (ENTER to use defaults):")
    
    author = input(f"Author [{default_author}]: ").strip() or default_author
    model = input(f"Model [{default_model}]: ").strip() or default_model
    
    default_repolink = f"https://huggingface.co/{author}/{model}"
    repolink = input(f"Repo Link [{default_repolink}]: ").strip() or default_repolink
    
    default_user = get_hf_username()
    user = input(f"Quantized By (HuggingFace Username) [{default_user}]: ").strip() or default_user
    
    return {
        "AUTHOR": author,
        "MODEL": model,
        "REPOLINK": repolink,
        "USER": user,
        "QUANT_METHOD": "EXL3",
        "QUANT_TOOL": "exllamav3",
    }

def run_readme(model_dir: str, template_name: Optional[str] = None) -> None:
    """
    Generate README.md for the model repository based on measurement CSV and template.
    """
    # Load template
    # Templates are in /templates/ relative to the project root (one level up from ezexl3/ package)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pkg_dir)
    templates_dir = os.path.join(project_root, "templates")
    
    if not template_name:
        template_name = "basic"
        
    # Search patterns
    possible_names = [
        template_name,
        f"{template_name}.md",
    ]
    
    # Let's adjust patterns to be more robust
    lookup_names = []
    for name in possible_names:
        if name not in lookup_names:
            lookup_names.append(name)
            
    # Add logic for names already ending in Template or README
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

    from ezexl3.measure import default_csv_path
    
    csv_path = default_csv_path(model_dir)
    if not os.path.exists(csv_path):
        print(f"🔴 CSV not found: {csv_path}. Cannot generate README.")
        return

    # Read CSV
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"🔴 CSV is empty: {csv_path}. Cannot generate README.")
        return

    # Sort rows for the table: BPWs first (ascending), then bf16
    def sort_key(r):
        w = r["weights"]
        if w == "bf16": return 100.0
        try: return float(w)
        except: return 200.0
    
    rows.sort(key=sort_key)
    
    bpws = [r["weights"] for r in rows if r["weights"] != "bf16"]
    
    # Get metadata
    meta = prompt_metadata(model_dir, bpws)
    
    # Sort rows for the table: BPWs first (ascending), then bf16
    formatted_labels = {}
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
            except:
                formatted_labels[w] = w
    
    # Generate table body
    # Columns in template: REVISION, SIZE (GiB), K/L DIV, PPL
    table_rows = []
    for r in rows:
        w = r["weights"]
        label = formatted_labels[w]
        
        gib = r.get("GiB", "x")
        try: gib = f"{float(gib):.2f}"
        except: pass
        
        kl = r.get("K/L Div", "x")
        try: kl = f"{float(kl):.4f}"
        except: pass
        
        ppl = r.get("PPL r-100", "x")
        try: ppl = f"{float(ppl):.4f}"
        except: pass

        rev_link = f'<a class="link-style" href="#">{label}</a>'
        
        row_html = f"""            <tr>
              <td>{rev_link}</td>
              <td>{gib}</td>
              <td>{kl}</td>
              <td>{ppl}</td>
            </tr>"""
        table_rows.append(row_html)

    table_body = "\n".join(table_rows)

    # Replace table in template
    template = re.sub(r"<tbody>.*?</tbody>", f"<tbody>\n{table_body}\n          </tbody>", template, flags=re.DOTALL)

    # Replace placeholders
    for k, v in meta.items():
        template = template.replace(f"{{{{{k}}}}}", str(v))

    # Replace DEFAULT_REVISION with the first BPW found, or "bf16" or "REVISION"
    default_rev = first_bpw or formatted_labels.get("bf16", "REVISION")
    template = template.replace("{{DEFAULT_REVISION}}", default_rev)

    # Write README.md
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(template)

    print(f"✅ Generated {readme_path}")
