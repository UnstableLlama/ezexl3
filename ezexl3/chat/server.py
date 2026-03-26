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
    app.router.add_post("/api/regenerate", handle_regenerate)
    app.router.add_post("/api/edit", handle_edit)
    app.router.add_post("/api/delete", handle_delete)
    app.router.add_post("/api/navigate", handle_navigate)
    app.router.add_get("/api/tree", handle_get_tree)
    app.router.add_post("/api/stop", handle_stop)
    app.router.add_post("/api/clear", handle_clear)
    app.router.add_get("/api/session/save", handle_session_save)
    app.router.add_post("/api/session/load", handle_session_load)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _stream_events(request, gen):
    """Consume an async generator and stream as SSE, then send tree snapshot."""
    engine: ChatEngine = request.app["engine"]
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

    sent_initial_tree = False
    async for event in gen:
        sse_data = f"data: {json.dumps(event)}\n\n"
        await response.write(sse_data.encode("utf-8"))
        # Send tree snapshot right after the start event so the client can
        # render the new nodes while tokens stream in
        if event.get("type") == "start" and not sent_initial_tree:
            sent_initial_tree = True
            t = {"type": "tree", "tree": engine.get_tree()}
            await response.write(f"data: {json.dumps(t)}\n\n".encode("utf-8"))
        await response.drain()

    # Send final tree snapshot so the client stays in sync
    tree_event = {"type": "tree", "tree": engine.get_tree()}
    await response.write(f"data: {json.dumps(tree_event)}\n\n".encode("utf-8"))
    await response.write(b"data: [DONE]\n\n")
    await response.drain()
    await response.write_eof()
    return response


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
    if not message:
        return web.json_response({"error": "Empty message"}, status=400)

    parent_id = data.get("parent_id")  # optional: which assistant node to follow
    return await _stream_events(request, engine.generate(message, parent_id))


async def handle_regenerate(request: web.Request) -> web.Response:
    """SSE endpoint: regenerate an assistant message (creates a new sibling)."""
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    node_id = data.get("node_id", "").strip()
    if not node_id:
        return web.json_response({"error": "Missing node_id"}, status=400)

    return await _stream_events(request, engine.regenerate(node_id))


async def handle_edit(request: web.Request) -> web.Response:
    """SSE endpoint: edit a user message and generate a new response."""
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    node_id = data.get("node_id", "").strip()
    content = data.get("content", "").strip()
    if not node_id or not content:
        return web.json_response(
            {"error": "Missing node_id or content"}, status=400
        )

    return await _stream_events(
        request, engine.edit_and_generate(node_id, content)
    )


async def handle_delete(request: web.Request) -> web.Response:
    """Delete a message node and all its descendants."""
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    node_id = data.get("node_id", "").strip()
    if not node_id:
        return web.json_response({"error": "Missing node_id"}, status=400)

    engine.delete_message(node_id)
    return web.json_response({"ok": True, "tree": engine.get_tree()})


async def handle_navigate(request: web.Request) -> web.Response:
    """Switch the active sibling at a branch point."""
    engine: ChatEngine = request.app["engine"]
    data = await request.json()
    node_id = data.get("node_id", "").strip()
    sibling_index = data.get("index")
    if not node_id or sibling_index is None:
        return web.json_response(
            {"error": "Missing node_id or index"}, status=400
        )

    engine.navigate_branch(node_id, int(sibling_index))
    return web.json_response({"ok": True, "tree": engine.get_tree()})


async def handle_get_tree(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    return web.json_response(engine.get_tree())


async def handle_stop(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    engine.cancel()
    return web.json_response({"ok": True})


async def handle_clear(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app["engine"]
    engine.clear_context()
    return web.json_response({"ok": True, "tree": engine.get_tree()})


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
    return web.json_response({"ok": True, "tree": engine.get_tree()})


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
