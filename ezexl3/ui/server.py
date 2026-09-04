# Lightweight aiohttp web server for the ezexl3 dashboard UI.

from __future__ import annotations

import asyncio
import collections
import ipaddress
import json
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import traceback
import uuid
import webbrowser

# Strip ANSI escape sequences (cursor movement, colors, line clears)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")
from pathlib import Path

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionError

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


# ---------------------------------------------------------------------------
# Job manager — runs one subprocess at a time, buffers output for SSE
# ---------------------------------------------------------------------------

class Job:
    __slots__ = (
        "id", "cmd", "process", "output", "status", "returncode",
        "_waiters", "total_appended",
    )

    def __init__(self, job_id: str, cmd: list[str]):
        self.id = job_id
        self.cmd = cmd
        self.process: asyncio.subprocess.Process | None = None
        self.output: collections.deque[dict] = collections.deque(maxlen=50_000)
        # Monotonic count of every event ever appended, so stream subscribers
        # can track progress even when the deque rolls over.
        self.total_appended: int = 0
        self.status: str = "starting"  # starting | running | stopped | done
        self.returncode: int | None = None
        self._waiters: list[asyncio.Event] = []

    def append_event(self, event: dict) -> None:
        self.output.append(event)
        self.total_appended += 1

    def notify(self):
        for ev in self._waiters:
            ev.set()

    def new_waiter(self) -> asyncio.Event:
        ev = asyncio.Event()
        self._waiters.append(ev)
        return ev

    def remove_waiter(self, ev: asyncio.Event):
        try:
            self._waiters.remove(ev)
        except ValueError:
            pass


class JobManager:
    def __init__(self):
        self.current: Job | None = None

    async def start(self, subcommand: str, args: list[str]) -> Job:
        if self.current and self.current.status in ("starting", "running"):
            raise RuntimeError("A job is already running")

        job_id = uuid.uuid4().hex[:12]
        # Prefer the installed entry-point script; fall back to python -m
        ezexl3_bin = shutil.which("ezexl3")
        if ezexl3_bin:
            cmd = [ezexl3_bin, subcommand] + args
        else:
            cmd = [sys.executable, "-m", "ezexl3", subcommand] + args
        job = Job(job_id, cmd)
        self.current = job

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        job.process = proc
        job.status = "running"

        asyncio.create_task(self._read_stream(job, proc.stdout, "stdout"))
        asyncio.create_task(self._read_stream(job, proc.stderr, "stderr"))
        asyncio.create_task(self._wait_exit(job, proc))

        return job

    async def _read_stream(self, job: Job, stream, stream_type: str):
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            # Strip ANSI escape sequences (cursor movement, colors, line clears)
            # that our simple terminal can't handle
            text = _ANSI_RE.sub("", text)
            # Split on \n, preserve \r within lines for progress bar handling
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if not line and i == len(lines) - 1:
                    break  # trailing empty after final \n
                suffix = "\n" if i < len(lines) - 1 else ""
                segment = line + suffix
                # Skip lines that are only whitespace/newlines (tqdm padding)
                if not segment.strip():
                    continue
                event = {"type": stream_type, "text": segment}
                job.append_event(event)
            job.notify()

    async def _wait_exit(self, job: Job, proc: asyncio.subprocess.Process):
        code = await proc.wait()
        job.returncode = code
        job.status = "done"
        job.append_event({"type": "exit", "code": code})
        job.notify()

    async def stop(self, job_id: str):
        job = self.current
        if not job or job.id != job_id:
            return
        if job.process and job.process.returncode is None:
            job.status = "stopped"
            job.process.terminate()
            try:
                await asyncio.wait_for(job.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                job.process.kill()


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html", headers={
        "Content-Type": "text/html",
        "Cache-Control": "no-cache, no-store, must-revalidate",
    })


async def handle_browse(request: web.Request) -> web.Response:
    """Server-side file browser for directory selection."""
    raw = request.query.get("path", "")
    browse_path = Path(raw) if raw else Path.home()

    try:
        browse_path = browse_path.resolve()
    except (OSError, ValueError):
        return web.json_response({"error": "Invalid path"}, status=400)

    if not browse_path.is_dir():
        return web.json_response({"error": "Not a directory"}, status=400)

    entries = []
    try:
        for entry in sorted(browse_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                entries.append({"name": entry.name, "type": "dir"})
            elif entry.suffix in (".json", ".safetensors", ".jinja"):
                entries.append({"name": entry.name, "type": "file"})
    except PermissionError:
        return web.json_response({"error": "Permission denied"}, status=403)

    is_model = (browse_path / "config.json").is_file()
    parent = str(browse_path.parent) if browse_path.parent != browse_path else None

    return web.json_response({
        "current": str(browse_path),
        "parent": parent,
        "entries": entries,
        "is_model": is_model,
    })


async def handle_pick_directory(request: web.Request) -> web.Response:
    """Open the native OS directory picker dialog."""
    import asyncio
    from ezexl3.native_dialog import pick_directory

    initial = request.query.get("initial", "")
    loop = asyncio.get_event_loop()
    try:
        path = await asyncio.wait_for(
            loop.run_in_executor(None, pick_directory, initial),
            timeout=300,
        )
    except asyncio.TimeoutError:
        return web.json_response({"path": None, "error": "Dialog timed out"})
    except Exception as e:
        return web.json_response({"path": None, "error": str(e)})

    return web.json_response({"path": path})


async def handle_gpus(request: web.Request) -> web.Response:
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 1),
                })
    except Exception:
        pass
    return web.json_response({"gpus": gpus})


async def handle_templates(request: web.Request) -> web.Response:
    templates = []
    if TEMPLATES_DIR.is_dir():
        for f in sorted(TEMPLATES_DIR.iterdir()):
            if f.suffix == ".md" and "Template" in f.name:
                # Extract short name: "fireTemplateREADME.md" -> "fire"
                short = f.stem.replace("TemplateREADME", "").replace("Template", "")
                if short:
                    templates.append(short)
    return web.json_response({"templates": templates})


async def handle_run(request: web.Request) -> web.Response:
    manager: JobManager = request.app["job_manager"]
    data = await request.json()
    subcommand = data.get("command", "").strip()
    args = data.get("args", [])

    if not subcommand:
        return web.json_response({"error": "No command specified"}, status=400)

    valid_commands = {"repo", "quantize", "quant", "mtp", "qbench", "measure", "readme", "upload"}
    if subcommand not in valid_commands:
        return web.json_response({"error": f"Invalid command: {subcommand}"}, status=400)

    if manager.current and manager.current.status in ("starting", "running"):
        return web.json_response({"error": "A job is already running"}, status=409)

    try:
        job = await manager.start(subcommand, args)
        return web.json_response({
            "job_id": job.id,
            "command": " ".join(job.cmd),
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_run_stream(request: web.Request) -> web.Response:
    """SSE endpoint — streams stdout/stderr from the running job."""
    manager: JobManager = request.app["job_manager"]
    job_id = request.match_info["job_id"]

    job = manager.current
    if not job or job.id != job_id:
        return web.json_response({"error": "Job not found"}, status=404)

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    # Follow new output. The whole body is wrapped so that a client
    # disconnect at any point (initial replay, mid-stream, or final
    # DONE/EOF) is swallowed quietly instead of logging a traceback.
    waiter = job.new_waiter()
    try:
        # Track progress with a cursor that indexes into job.total_appended
        # (the monotonic counter) so we stay correct even after the bounded
        # deque rolls over. The first loop iteration replays everything
        # currently buffered; subsequent iterations send only new events.
        cursor = 0

        def _pending_events() -> list[dict]:
            """Return any events the client hasn't seen yet, in order."""
            snapshot = list(job.output)
            buffered_start = job.total_appended - len(snapshot)
            if cursor >= job.total_appended:
                return []
            start = max(0, cursor - buffered_start)
            return snapshot[start:]

        while True:
            waiter.clear()

            pending = _pending_events()
            if pending:
                for event in pending:
                    sse_data = f"data: {json.dumps(event)}\n\n"
                    await response.write(sse_data.encode("utf-8"))
                cursor = job.total_appended

            # If the job is done and we've drained everything, close.
            if job.status in ("done", "stopped") and cursor >= job.total_appended:
                break

            try:
                await asyncio.wait_for(waiter.wait(), timeout=30)
            except asyncio.TimeoutError:
                # Send keepalive so intermediaries don't idle us out.
                await response.write(b": keepalive\n\n")

        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
    except (ConnectionResetError, ConnectionAbortedError, ClientConnectionError, asyncio.CancelledError):
        # Client went away — nothing to do.
        pass
    finally:
        job.remove_waiter(waiter)

    return response


async def handle_run_stop(request: web.Request) -> web.Response:
    manager: JobManager = request.app["job_manager"]
    job_id = request.match_info["job_id"]
    await manager.stop(job_id)
    return web.json_response({"ok": True})


async def handle_run_status(request: web.Request) -> web.Response:
    manager: JobManager = request.app["job_manager"]
    job = manager.current
    if not job:
        return web.json_response({"status": "idle"})
    return web.json_response({
        "status": job.status,
        "job_id": job.id,
        "command": " ".join(job.cmd),
        "returncode": job.returncode,
    })


# ---------------------------------------------------------------------------
# Live measurement data + graph
# ---------------------------------------------------------------------------

def _resolve_db_path(model_dir: str) -> str | None:
    """Find the measurement DB for a model directory."""
    try:
        from ezexl3.measure_db import default_db_path
        db = default_db_path(model_dir)
        return db if os.path.exists(db) else None
    except ImportError:
        return None


async def handle_data(request: web.Request) -> web.Response:
    """Return current measurement rows from the SQLite DB as JSON."""
    model_dir = request.query.get("model_dir", "").strip()
    if not model_dir:
        return web.json_response({"rows": [], "error": "No model_dir"})

    db_path = _resolve_db_path(model_dir)
    if not db_path:
        return web.json_response({"rows": []})

    try:
        from ezexl3.measure_db import read_all_rows
        rows_dict = await asyncio.to_thread(read_all_rows, db_path)
        # Sort numerically by BPW
        rows = list(rows_dict.values())
        rows.sort(key=lambda r: _bpw_key(r.get("weights", "")))
        return web.json_response({"rows": rows})
    except Exception as e:
        return web.json_response({"rows": [], "error": str(e)})


async def handle_perf_data(request: web.Request) -> web.Response:
    """Return detailed per-context-length perf data from the perf DB."""
    model_dir = request.query.get("model_dir", "").strip()
    bpw = request.query.get("bpw", "").strip() or None
    if not model_dir:
        return web.json_response({"bpws": [], "data": {}})

    try:
        from ezexl3.perf_db import (
            available_bpws,
            default_perf_db_path,
            read_perf_data,
        )
        perf_db = default_perf_db_path(model_dir)
        bpws = await asyncio.to_thread(available_bpws, perf_db)
        data = await asyncio.to_thread(read_perf_data, perf_db, bpw)
        return web.json_response({"bpws": bpws, "data": data})
    except Exception as e:
        return web.json_response({"bpws": [], "data": {}, "error": str(e)})


async def handle_graph(request: web.Request) -> web.Response:
    """Generate SVG graph from current DB state and return it."""
    model_dir = request.query.get("model_dir", "").strip()
    if not model_dir:
        return web.json_response({"error": "No model_dir"}, status=400)

    db_path = _resolve_db_path(model_dir)
    if not db_path:
        return web.json_response({"error": "No measurement data yet"}, status=404)

    try:
        from ezexl3.measure_db import export_csv, read_all_rows

        # Need at least 2 numeric rows to draw
        rows = await asyncio.to_thread(read_all_rows, db_path)
        numeric = [r for r in rows.values()
                   if r.get("KL Div") and r.get("PPL r-100") and r.get("GiB")]
        if len(numeric) < 2:
            return web.json_response({"error": "Need at least 2 completed measurements"}, status=404)

        # Export to temp CSV, generate SVG, return inline
        model_name = _resolve_model_name(model_dir)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            svg_path = os.path.join(tmp, "graph.svg")
            await asyncio.to_thread(export_csv, db_path, csv_path)

            from ezexl3.graph_svg import generate_iceblink_svg
            await asyncio.to_thread(
                generate_iceblink_svg, csv_path, svg_path, model_name,
            )

            svg_content = Path(svg_path).read_text(encoding="utf-8")
        return web.Response(
            text=svg_content,
            content_type="image/svg+xml",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


def _bpw_key(label: str) -> float:
    try:
        return float(label)
    except (ValueError, TypeError):
        return float("inf")


def _resolve_model_name(model_dir: str) -> str:
    """Get display name: locked MODEL from metadata, or underscore-stripped basename."""
    meta_path = os.path.join(model_dir, ".ezexl3_readme_meta.json")
    if os.path.exists(meta_path):
        try:
            meta = json.loads(Path(meta_path).read_text("utf-8"))
            locked = meta.get("_locked") or {}
            if locked.get("MODEL") and meta.get("MODEL", "").strip():
                return meta["MODEL"]
        except Exception:
            pass
    name = os.path.basename(os.path.abspath(model_dir))
    if "_" in name:
        name = name.split("_", 1)[1]
    return name


async def handle_perf_graph(request: web.Request) -> web.Response:
    """Generate a perf chart SVG for the selected BPW and return it inline."""
    model_dir = request.query.get("model_dir", "").strip()
    bpw = request.query.get("bpw", "").strip()
    if not model_dir or not bpw:
        return web.json_response({"error": "model_dir and bpw required"}, status=400)

    try:
        from ezexl3.perf_db import default_perf_db_path
        perf_db = default_perf_db_path(model_dir)
        if not os.path.exists(perf_db):
            return web.json_response({"error": "No perf data yet"}, status=404)

        model_name = _resolve_model_name(model_dir)
        title = f"{model_name} — {bpw} BPW"

        with tempfile.TemporaryDirectory() as tmp:
            svg_path = os.path.join(tmp, "perf.svg")
            from ezexl3.graph_svg import generate_perf_svg
            await asyncio.to_thread(
                generate_perf_svg, perf_db, bpw, svg_path, title,
            )
            svg_content = Path(svg_path).read_text(encoding="utf-8")

        return web.Response(
            text=svg_content,
            content_type="image/svg+xml",
            headers={"Cache-Control": "no-cache"},
        )
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=404)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


_CATBENCH_NUMBERED_RE = re.compile(r"_\d+\.svg$")


def _scan_catbench_dir(catbench_dir: str) -> list[dict]:
    """List canonical catbench SVGs in *catbench_dir*, sorted by BPW.

    Skips numbered variants ({prefix}_1.svg, …) so only one tile appears
    per BPW. ``bf16`` sorts last. Filenames that don't match the expected
    ``{bpw}bpw.svg`` or ``bf16.svg`` layout are ignored.
    """
    if not os.path.isdir(catbench_dir):
        return []

    items: list[tuple[float, dict]] = []
    for fn in os.listdir(catbench_dir):
        if not fn.endswith(".svg") or _CATBENCH_NUMBERED_RE.search(fn):
            continue
        if fn == "bf16.svg":
            items.append((float("inf"), {"label": "BF16", "bpw": "bf16", "file": fn}))
        elif fn.endswith("bpw.svg"):
            try:
                val = float(fn[:-len("bpw.svg")])
            except ValueError:
                continue
            items.append((val, {"label": f"{val:.2f} bpw", "bpw": f"{val:.2f}", "file": fn}))

    items.sort(key=lambda t: t[0])
    return [item for _, item in items]


async def handle_catbench_file(request: web.Request) -> web.Response:
    """Serve catbench artifacts for a model dir.

    Two modes (distinguished by ``file`` query param):

    * ``?model_dir=X``            → JSON listing of canonical SVGs
    * ``?model_dir=X&file=Y.svg`` → serve that SVG file inline

    The file-serving branch normalizes the requested path and rejects
    anything that escapes ``{model_dir}/catbench/``.
    """
    model_dir = request.query.get("model_dir", "").strip()
    fname = request.query.get("file", "").strip()
    if not model_dir:
        return web.json_response({"error": "No model_dir"}, status=400)

    catbench_dir = os.path.join(model_dir, "catbench")

    # Listing mode
    if not fname:
        try:
            items = await asyncio.to_thread(_scan_catbench_dir, catbench_dir)
            return web.json_response({"items": items})
        except Exception as e:
            return web.json_response({"items": [], "error": str(e)}, status=500)

    # File-serving mode — path must resolve inside catbench_dir
    try:
        catbench_real = Path(catbench_dir).resolve()
        requested = (Path(catbench_dir) / fname).resolve()
    except (OSError, ValueError):
        return web.json_response({"error": "Invalid path"}, status=400)

    try:
        requested.relative_to(catbench_real)
    except ValueError:
        return web.json_response({"error": "Path escapes catbench dir"}, status=400)

    if requested.suffix != ".svg" or not requested.is_file():
        return web.json_response({"error": "Not found"}, status=404)

    return web.FileResponse(
        requested,
        headers={"Content-Type": "image/svg+xml", "Cache-Control": "no-cache"},
    )


async def handle_metadata_get(request: web.Request) -> web.Response:
    """Return saved README metadata or computed defaults for a model directory."""
    model_dir = request.query.get("model_dir", "").strip()
    if not model_dir:
        return web.json_response({"error": "No model_dir"}, status=400)

    meta_path = os.path.join(model_dir, ".ezexl3_readme_meta.json")
    if os.path.exists(meta_path):
        try:
            data = json.loads(Path(meta_path).read_text("utf-8"))
            return web.json_response(data)
        except Exception:
            pass

    try:
        from ezexl3.readme import _compute_defaults
        defaults = await asyncio.to_thread(_compute_defaults, model_dir)
        defaults["_defaults"] = True
        return web.json_response(defaults)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_metadata_set(request: web.Request) -> web.Response:
    """Save README metadata to the model directory, clearing any waiting flag.

    Merges over any existing metadata: only keys present in the request are
    updated, so a partial editor (e.g. the upload tab, which only shows
    MODEL and USER) won't wipe AUTHOR/REPOLINK set elsewhere.
    """
    data = await request.json()
    model_dir = data.get("model_dir", "").strip()
    if not model_dir:
        return web.json_response({"error": "No model_dir"}, status=400)

    # Reject paths that exist but aren't a directory (e.g. the user pasted
    # /path/to/model/README.md instead of /path/to/model). makedirs below
    # would raise FileExistsError in that case and crash the handler.
    if os.path.exists(model_dir) and not os.path.isdir(model_dir):
        return web.json_response(
            {"error": f"model_dir is not a directory: {model_dir}"},
            status=400,
        )

    # Legacy: clients used to POST _confirm=true when the user clicked a
    # Resume button. That button was removed — the backend now auto-resumes
    # when every field is locked — so this branch is effectively dead, but
    # left in place so any stale client that still POSTs the flag behaves
    # correctly.
    confirm = data.get("_confirm", False)

    meta_path = os.path.join(model_dir, ".ezexl3_readme_meta.json")
    existing: dict = {}
    if os.path.exists(meta_path):
        try:
            existing = json.loads(Path(meta_path).read_text("utf-8"))
        except Exception:
            existing = {}

    meta = dict(existing)
    for key in ("AUTHOR", "MODEL", "REPOLINK", "USER"):
        if key in data:
            meta[key] = data[key]

    # Merge lock state — incoming wins, existing entries for other fields kept
    existing_locked = existing.get("_locked") or {}
    incoming_locked = data.get("_locked") or {}
    meta["_locked"] = {**existing_locked, **incoming_locked}

    meta["_waiting"] = False if confirm else existing.get("_waiting", False)

    os.makedirs(model_dir, exist_ok=True)
    Path(meta_path).write_text(json.dumps(meta, indent=2), "utf-8")

    return web.json_response({"ok": True})


async def handle_chat_launch(request: web.Request) -> web.Response:
    """Shut down the dashboard and launch the chat UI in its place."""
    manager: JobManager = request.app["job_manager"]
    if manager.current and manager.current.status in ("starting", "running"):
        return web.json_response(
            {"error": "Cannot launch chat while a job is running"}, status=409,
        )

    # Find port/host from the running server
    host = request.app.get("_host", "127.0.0.1")
    port = request.app.get("_port", 8801)

    # Spawn chat server as a separate process, then shut ourselves down
    ezexl3_bin = shutil.which("ezexl3")
    if ezexl3_bin:
        chat_cmd = [ezexl3_bin, "chat", "--port", str(port), "--host", host, "--no-browser"]
    else:
        chat_cmd = [sys.executable, "-m", "ezexl3", "chat", "--port", str(port), "--host", host, "--no-browser"]

    # Store the command — the pre-registered cleanup handler will spawn it
    # after aiohttp has released the port.
    request.app["_spawn_on_exit"] = chat_cmd

    # Schedule graceful shutdown — SIGINT lets aiohttp close connections
    # cleanly, release the port, then run cleanup (which spawns chat).
    async def _graceful_exit():
        await asyncio.sleep(1)
        os.kill(os.getpid(), signal.SIGINT)
    asyncio.ensure_future(_graceful_exit())

    return web.json_response({"ok": True, "url": f"http://{host}:{port}"})


# ---------------------------------------------------------------------------
# Persistent user config (~/.config/ezexl3/ui.json)
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "ezexl3" / "ui.json"


_CONFIG_LOCK = threading.Lock()


def _read_config() -> dict:
    """Strict read: {} only when the file genuinely isn't there.

    Raises on an existing-but-unparseable file, so a read-modify-write
    can tell "nothing saved yet" from "couldn't read what's saved" —
    conflating the two is how a whole config gets replaced by one key.
    """
    p = _config_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text("utf-8"))


def _load_config() -> dict:
    """Best-effort read for display. Never raises."""
    try:
        return _read_config()
    except Exception:
        return {}


def _save_config(data: dict) -> None:
    # Write a temp file and rename it over the target: a reader sees
    # either the old file or the new one, never a half-written one. The
    # dashboard and the chat server share this file and do run at once
    # (the dashboard switch spawns one from the other), and a plain
    # write_text left a window where a reader caught the truncated file,
    # parsed it as {}, and then persisted that back over everything.
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".ui.json.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _update_config(incoming: dict) -> None:
    """Merge incoming keys into the saved config and write it back."""
    with _CONFIG_LOCK:
        try:
            cfg = _read_config()
        except Exception:
            # Corrupt or unreadable. Keep a copy instead of letting the
            # merge below silently overwrite it with near-nothing.
            p = _config_path()
            try:
                p.replace(p.with_suffix(".json.corrupt"))
            except OSError:
                pass
            cfg = {}
        cfg.update(incoming)
        _save_config(cfg)


async def handle_config_get(request: web.Request) -> web.Response:
    return web.json_response(await asyncio.to_thread(_load_config))


async def handle_config_set(request: web.Request) -> web.Response:
    incoming = await request.json()
    await asyncio.to_thread(_update_config, incoming)
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Middleware — prevent browser from caching static files across server swaps
# ---------------------------------------------------------------------------

@web.middleware
async def _no_cache_static(request: web.Request, handler):
    resp = await handler(request)
    # Do NOT touch headers on a response whose body has already started
    # streaming (our SSE handler calls prepare() itself). Modifying headers
    # after prepare() is undefined behavior in aiohttp.
    if getattr(resp, "prepared", False):
        return resp
    if "Cache-Control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------------------
# HuggingFace auth check
# ---------------------------------------------------------------------------

async def handle_hf_auth(request: web.Request) -> web.Response:
    """Check if user is authenticated with HuggingFace."""
    try:
        from huggingface_hub import HfApi
        info = await asyncio.to_thread(HfApi().whoami)
        return web.json_response({"authenticated": True, "username": info.get("name", "")})
    except Exception:
        return web.json_response({"authenticated": False, "username": ""})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> web.Application:
    app = web.Application(middlewares=[_no_cache_static])
    app["job_manager"] = JobManager()
    app["_spawn_on_exit"] = None  # set by handle_chat_launch

    async def _cleanup_spawn(app):
        cmd = app.get("_spawn_on_exit")
        if cmd:
            import subprocess
            subprocess.Popen(cmd, start_new_session=True)
    app.on_cleanup.append(_cleanup_spawn)

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/browse", handle_browse)
    app.router.add_get("/api/pick_directory", handle_pick_directory)
    app.router.add_get("/api/gpus", handle_gpus)
    app.router.add_get("/api/hf-auth", handle_hf_auth)
    app.router.add_get("/api/templates", handle_templates)
    app.router.add_post("/api/run", handle_run)
    app.router.add_get("/api/run/{job_id}/stream", handle_run_stream)
    app.router.add_post("/api/run/{job_id}/stop", handle_run_stop)
    app.router.add_get("/api/run/status", handle_run_status)
    app.router.add_get("/api/data", handle_data)
    app.router.add_get("/api/perf-data", handle_perf_data)
    app.router.add_get("/api/perf-graph", handle_perf_graph)
    app.router.add_get("/api/catbench-file", handle_catbench_file)
    app.router.add_get("/api/graph", handle_graph)
    app.router.add_get("/api/metadata", handle_metadata_get)
    app.router.add_post("/api/metadata", handle_metadata_set)
    app.router.add_post("/api/chat/launch", handle_chat_launch)
    app.router.add_get("/api/config", handle_config_get)
    app.router.add_post("/api/config", handle_config_set)
    app.router.add_static("/", STATIC_DIR, show_index=False, append_version=True)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _kill_port_holder(port: int) -> bool:
    """Try to kill whatever process is holding *port*. Returns True on success."""
    import subprocess
    try:
        pids = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True,
        ).strip().split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    if not pids:
        return False
    my_pid = str(os.getpid())
    for pid in pids:
        if pid == my_pid:
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    # Give the OS a moment to release the socket
    import time
    time.sleep(0.5)
    return True


def run_ui_server(
    host: str = "127.0.0.1",
    port: int = 8801,
    open_browser: bool = True,
):
    try:
        addr = ipaddress.ip_address(host)
        if not addr.is_loopback:
            print(
                f"\n  WARNING: Binding to non-loopback address {host}.\n"
                f"  The dashboard API has no authentication — anyone with network\n"
                f"  access can execute commands on this machine.\n",
                file=sys.stderr,
            )
    except ValueError:
        pass

    url = f"http://{host}:{port}"
    print(f"  ezexl3 dashboard: {url}")
    print(f"  Press Ctrl+C to stop\n")

    def _make_app():
        app = create_app()
        app["_host"] = host
        app["_port"] = port
        if open_browser:
            async def _open_browser(_app):
                import threading
                def _delayed_open():
                    import time
                    time.sleep(2)
                    webbrowser.open(url)
                threading.Thread(target=_delayed_open, daemon=True).start()
            app.on_startup.append(_open_browser)
        return app

    try:
        web.run_app(_make_app(), host=host, port=port, print=None,
                    reuse_address=True)
    except OSError as exc:
        if exc.errno == 98:  # EADDRINUSE
            if _kill_port_holder(port):
                print(f"  Killed stale process on port {port}, retrying...\n")
                web.run_app(_make_app(), host=host, port=port, print=None,
                            reuse_address=True)
            else:
                print(
                    f"\n  ERROR: Port {port} is already in use and could not"
                    f" be freed.\n"
                    f"  Fix:  lsof -ti :{port} | xargs kill\n"
                    f"  Or use a different port:  ezexl3 ui --port {port + 1}\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            raise
