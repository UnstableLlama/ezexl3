# Lightweight aiohttp web server for the ezexl3 chat UI.

from __future__ import annotations

import asyncio
import json
import os
import webbrowser
from pathlib import Path

from aiohttp import web

from .inference import ChatEngine, ChatSettings

STATIC_DIR = Path(__file__).parent / "static"


def create_app(engine: ChatEngine) -> web.Application:
    app = web.Application()
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

    return app


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    html_path = STATIC_DIR / "index.html"
    return web.FileResponse(html_path, headers={"Content-Type": "text/html"})


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

    async for event in engine.generate(message):
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
# Entry point
# ---------------------------------------------------------------------------

def run_server(
    model_dir: str,
    devices: list[int] | None = None,
    device_ratios: str | None = None,
    cache_size: int | None = None,
    cache_quant: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8800,
    open_browser: bool = True,
):
    """Load model then start the web server."""
    engine = ChatEngine(
        model_dir=model_dir,
        devices=devices,
        device_ratios=device_ratios,
        cache_size=cache_size,
        cache_quant=cache_quant,
    )

    print(f"Loading model from: {model_dir}")
    engine.load()
    print()

    app = create_app(engine)
    url = f"http://{host}:{port}"
    print(f"  Chat UI: {url}")
    print(f"  Press Ctrl+C to stop\n")

    if open_browser:
        webbrowser.open(url)

    web.run_app(app, host=host, port=port, print=None)
