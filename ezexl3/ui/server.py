# Lightweight aiohttp web server for the ezexl3 dashboard UI.

from __future__ import annotations

import asyncio
import collections
import ipaddress
import json
import os
import shutil
import signal
import sys
import tempfile
import uuid
import webbrowser
from pathlib import Path

from aiohttp import web

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


# ---------------------------------------------------------------------------
# Job manager — runs one subprocess at a time, buffers output for SSE
# ---------------------------------------------------------------------------

class Job:
    __slots__ = ("id", "cmd", "process", "output", "status", "returncode", "_waiters")

    def __init__(self, job_id: str, cmd: list[str]):
        self.id = job_id
        self.cmd = cmd
        self.process: asyncio.subprocess.Process | None = None
        self.output: collections.deque[dict] = collections.deque(maxlen=50_000)
        self.status: str = "starting"  # starting | running | stopped | done
        self.returncode: int | None = None
        self._waiters: list[asyncio.Event] = []

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
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            event = {"type": stream_type, "text": text}
            job.output.append(event)
            job.notify()

    async def _wait_exit(self, job: Job, proc: asyncio.subprocess.Process):
        code = await proc.wait()
        job.returncode = code
        job.status = "done"
        event = {"type": "exit", "code": code}
        job.output.append(event)
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

    valid_commands = {"repo", "quantize", "quant", "measure", "readme"}
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
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    # Replay buffered output
    cursor = 0
    for event in list(job.output):
        sse_data = f"data: {json.dumps(event)}\n\n"
        await response.write(sse_data.encode("utf-8"))
        cursor += 1

    # Follow new output
    waiter = job.new_waiter()
    try:
        while True:
            waiter.clear()
            # Send any new events since last cursor
            current_len = len(job.output)
            if cursor < current_len:
                items = list(job.output)
                for event in items[cursor:]:
                    sse_data = f"data: {json.dumps(event)}\n\n"
                    await response.write(sse_data.encode("utf-8"))
                cursor = current_len

            # If job is done and we've sent everything, close
            if job.status in ("done", "stopped") and cursor >= len(job.output):
                break

            # Wait for new data
            try:
                await asyncio.wait_for(waiter.wait(), timeout=30)
            except asyncio.TimeoutError:
                # Send keepalive
                await response.write(b": keepalive\n\n")
    except (ConnectionResetError, ConnectionAbortedError):
        pass
    finally:
        job.remove_waiter(waiter)

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
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
        model_name = os.path.basename(os.path.abspath(model_dir))
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
    """Save README metadata to the model directory, clearing any waiting flag."""
    data = await request.json()
    model_dir = data.get("model_dir", "").strip()
    if not model_dir:
        return web.json_response({"error": "No model_dir"}, status=400)

    # Only clear _waiting when the dashboard explicitly confirms (Resume click)
    confirm = data.get("_confirm", False)

    # Read existing state to preserve _waiting unless confirming
    meta_path = os.path.join(model_dir, ".ezexl3_readme_meta.json")
    existing_waiting = False
    if not confirm and os.path.exists(meta_path):
        try:
            existing = json.loads(Path(meta_path).read_text("utf-8"))
            existing_waiting = existing.get("_waiting", False)
        except Exception:
            pass

    meta = {
        "AUTHOR": data.get("AUTHOR", ""),
        "MODEL": data.get("MODEL", ""),
        "REPOLINK": data.get("REPOLINK", ""),
        "USER": data.get("USER", ""),
        "_locked": data.get("_locked", {}),
        "_waiting": existing_waiting if not confirm else False,
    }

    meta_path = os.path.join(model_dir, ".ezexl3_readme_meta.json")
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


def _load_config() -> dict:
    p = _config_path()
    if p.is_file():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _save_config(data: dict) -> None:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), "utf-8")


async def handle_config_get(request: web.Request) -> web.Response:
    return web.json_response(await asyncio.to_thread(_load_config))


async def handle_config_set(request: web.Request) -> web.Response:
    incoming = await request.json()
    cfg = await asyncio.to_thread(_load_config)
    cfg.update(incoming)
    await asyncio.to_thread(_save_config, cfg)
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Middleware — prevent browser from caching static files across server swaps
# ---------------------------------------------------------------------------

@web.middleware
async def _no_cache_static(request: web.Request, handler):
    resp = await handler(request)
    if "Cache-Control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-store"
    return resp


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
    app.router.add_get("/api/gpus", handle_gpus)
    app.router.add_get("/api/templates", handle_templates)
    app.router.add_post("/api/run", handle_run)
    app.router.add_get("/api/run/{job_id}/stream", handle_run_stream)
    app.router.add_post("/api/run/{job_id}/stop", handle_run_stop)
    app.router.add_get("/api/run/status", handle_run_status)
    app.router.add_get("/api/data", handle_data)
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
