# Lightweight aiohttp web server for the ezexl3 chat UI.

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import signal
import sys
import webbrowser
from pathlib import Path

from aiohttp import web

from .inference import ChatEngine, ChatSettings

STATIC_DIR = Path(__file__).parent / "static"

# Files that signal a valid model directory
_MODEL_MARKERS = {"config.json"}


@web.middleware
async def _no_cache_static(request: web.Request, handler):
    resp = await handler(request)
    if "Cache-Control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def create_app(engine: ChatEngine) -> web.Application:
    app = web.Application(middlewares=[_no_cache_static])
    app["engine"] = engine

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/settings", handle_get_settings)
    app.router.add_post("/api/settings", handle_set_settings)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_post("/api/stop", handle_stop)
    app.router.add_post("/api/clear", handle_clear)
    app.router.add_get("/api/session/save", handle_session_save)
    app.router.add_post("/api/session/load", handle_session_load)
    app.router.add_get("/api/gpus", handle_gpus)
    app.router.add_get("/api/browse", handle_browse)
    app.router.add_post("/api/model/load", handle_model_load)
    app.router.add_post("/api/model/unload", handle_model_unload)
    app.router.add_post("/api/ui/launch", handle_ui_launch)
    app.router.add_static("/", STATIC_DIR, show_index=False, append_version=True)

    return app


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    html_path = STATIC_DIR / "index.html"
    return web.FileResponse(html_path, headers={
        "Content-Type": "text/html",
        "Cache-Control": "no-cache, no-store, must-revalidate",
    })


async def handle_status(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    return web.json_response(engine.get_status())


async def handle_get_settings(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    return web.json_response(engine.settings.to_dict())


async def handle_set_settings(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    # Merge incoming fields into current settings
    current = engine.settings.to_dict()
    current.update(data)
    engine.settings = ChatSettings.from_dict(current)
    return web.json_response({"ok": True})


async def handle_chat(request: web.Request) -> web.Response:
    """SSE endpoint: streams token-by-token responses."""
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    message = data.get("message", "").strip()
    if "context" in data:
        engine.context = [tuple(pair) for pair in data["context"]]
    if not message:
        return web.json_response({"error": "Empty message"}, status=400)

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

    prefix = data.get("prefix", "")
    async for event in engine.generate(message, prefix=prefix):
        sse_data = f"data: {json.dumps(event)}\n\n"
        await response.write(sse_data.encode("utf-8"))

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def handle_stop(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    engine.cancel()
    return web.json_response({"ok": True})


async def handle_clear(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    engine.clear_context()
    return web.json_response({"ok": True})


async def handle_session_save(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    session_data = engine.save_session()
    return web.json_response(
        session_data,
        headers={
            "Content-Disposition": 'attachment; filename="chat_session.json"',
        },
    )


async def handle_session_load(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    engine.load_session(data)
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Model management routes
# ---------------------------------------------------------------------------

async def handle_gpus(request: web.Request) -> web.Response:
    gpus = ChatEngine.detect_gpus()
    return web.json_response({"gpus": gpus})


async def handle_browse(request: web.Request) -> web.Response:
    """Server-side file browser for model directory selection."""
    raw = request.query.get("path", "")
    browse_path = Path(raw) if raw else Path.home()

    try:
        browse_path = browse_path.resolve()
    except (OSError, ValueError):
        return web.json_response(
            {"error": "Invalid path"}, status=400,
        )

    if not browse_path.is_dir():
        return web.json_response(
            {"error": "Not a directory"}, status=400,
        )

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
        return web.json_response(
            {"error": "Permission denied"}, status=403,
        )

    is_model = (browse_path / "config.json").is_file()
    parent = str(browse_path.parent) if browse_path.parent != browse_path else None

    return web.json_response({
        "current": str(browse_path),
        "parent": parent,
        "entries": entries,
        "is_model": is_model,
    })


async def handle_model_load(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    model_dir = data.get("model_dir", "").strip()
    if not model_dir:
        return web.json_response(
            {"ok": False, "error": "model_dir is required"}, status=400,
        )
    if not Path(model_dir).is_dir():
        return web.json_response(
            {"ok": False, "error": f"Directory not found: {model_dir}"}, status=400,
        )
    if not (Path(model_dir) / "config.json").is_file():
        return web.json_response(
            {"ok": False, "error": "No config.json found — not a valid model directory"},
            status=400,
        )

    try:
        await asyncio.to_thread(
            engine.load_model,
            model_dir=model_dir,
            devices=data.get("devices"),
            device_ratios=data.get("device_ratios"),
            cache_size=data.get("cache_size"),
            cache_quant=data.get("cache_quant"),
        )
        return web.json_response({
            "ok": True,
            "status": engine.get_status(),
            "settings": engine.settings.to_dict(),
        })
    except Exception as e:
        return web.json_response(
            {"ok": False, "error": str(e)}, status=500,
        )


async def handle_model_unload(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    engine.unload()
    return web.json_response({"ok": True})


async def handle_ui_launch(request: web.Request) -> web.Response:
    """Shut down the chat server and launch the dashboard UI in its place."""
    engine: ChatEngine = request.app["engine"]
    if engine.is_loaded:
        return web.json_response(
            {"error": "Unload the model before switching to dashboard"},
            status=409,
        )
    if engine.is_generating:
        return web.json_response(
            {"error": "Cannot switch while generating"}, status=409,
        )

    host = request.app.get("_host", "127.0.0.1")
    port = request.app.get("_port", 8800)

    import shutil
    ezexl3_bin = shutil.which("ezexl3")
    if ezexl3_bin:
        ui_cmd = [ezexl3_bin, "ui", "--port", str(port), "--host", host,
                  "--no-browser"]
    else:
        ui_cmd = [sys.executable, "-m", "ezexl3", "ui", "--port", str(port),
                  "--host", host, "--no-browser"]

    await asyncio.create_subprocess_exec(
        *ui_cmd, stdout=None, stderr=None, start_new_session=True,
    )

    import threading
    def _delayed_exit():
        import time
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=_delayed_exit, daemon=True).start()

    return web.json_response({"ok": True, "url": f"http://{host}:{port}"})


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


def run_server(
    model_dir: str | None = None,
    devices: list[int] | None = None,
    device_ratios: str | None = None,
    cache_size: int | None = None,
    cache_quant: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8800,
    open_browser: bool = True,
):
    """Optionally load a model, then start the web server."""
    engine = ChatEngine(
        model_dir=model_dir,
        devices=devices,
        device_ratios=device_ratios,
        cache_size=cache_size,
        cache_quant=cache_quant,
    )

    # Warn if binding to a non-loopback address (no auth layer).
    try:
        addr = ipaddress.ip_address(host)
        if not addr.is_loopback:
            print(
                f"\n  WARNING: Binding to non-loopback address {host}.\n"
                f"  The chat API has no authentication — anyone with network\n"
                f"  access to this host can interact with the model.\n"
                f"  Consider using a reverse-proxy with auth, or an SSH tunnel.\n",
                file=sys.stderr,
            )
    except ValueError:
        pass  # hostname like "localhost" — resolved by aiohttp

    if model_dir:
        print(f"Loading model from: {model_dir}")
        engine.load()
        print()
    else:
        print("  No model specified — select one in the UI.")

    url = f"http://{host}:{port}"
    print(f"  Chat UI: {url}")
    print(f"  Press Ctrl+C to stop\n")

    def _make_app():
        app = create_app(engine)
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
                    f"  Or use a different port:  ezexl3 chat --port {port + 1}\n",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            raise
