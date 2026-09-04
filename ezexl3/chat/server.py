# Lightweight aiohttp web server for the ezexl3 chat UI.

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import signal
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

from aiohttp import web

from .inference import ChatEngine, ChatSettings, cpu_offload_support, ngram_ram_support
from .ratings import (
    RatingsStore, default_datasets_dir, strip_think_text, valid_dataset_name,
    validate_prompt,
)

STATIC_DIR = Path(__file__).parent / "static"

# Files that signal a valid model directory
_MODEL_MARKERS = {"config.json"}

# Typed app keys (avoids NotAppKeyWarning)
_KEY_ENGINE = web.AppKey("engine", ChatEngine)
_KEY_SPAWN = web.AppKey("_spawn_on_exit", list)
_KEY_HOST = web.AppKey("_host", str)
_KEY_PORT = web.AppKey("_port", int)


@web.middleware
async def _no_cache_static(request: web.Request, handler):
    resp = await handler(request)
    if "Cache-Control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def create_app(engine: ChatEngine) -> web.Application:
    app = web.Application(middlewares=[_no_cache_static])
    app[_KEY_ENGINE] = engine
    app[_KEY_SPAWN] = None  # set by handle_ui_launch

    async def _cleanup_spawn(app):
        cmd = app.get(_KEY_SPAWN)
        if cmd:
            import subprocess
            subprocess.Popen(cmd, start_new_session=True)
    app.on_cleanup.append(_cleanup_spawn)

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
    app.router.add_get("/api/pick_directory", handle_pick_directory)
    app.router.add_post("/api/model/load", handle_model_load)
    app.router.add_post("/api/model/unload", handle_model_unload)
    app.router.add_post("/api/loras/apply", handle_loras_apply)
    app.router.add_post("/api/draft/load", handle_draft_load)
    app.router.add_post("/api/draft/unload", handle_draft_unload)
    app.router.add_post("/api/ui/launch", handle_ui_launch)
    app.router.add_get("/api/config", handle_config_get)
    app.router.add_post("/api/config", handle_config_set)
    app.router.add_get("/api/ratings", handle_ratings_get)
    app.router.add_post("/api/rate", handle_rate)
    app.router.add_post("/api/ratings/bulk", handle_ratings_bulk)
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
    engine: ChatEngine = request.app[_KEY_ENGINE]
    return web.json_response(engine.get_status())


async def handle_get_settings(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    return web.json_response(engine.settings.to_dict())


async def handle_set_settings(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    data = await request.json()
    # Merge incoming fields into current settings
    current = engine.settings.to_dict()
    current.update(data)
    engine.settings = ChatSettings.from_dict(current)
    return web.json_response({"ok": True})


async def handle_chat(request: web.Request) -> web.Response:
    """SSE endpoint: streams token-by-token responses."""
    engine: ChatEngine = request.app[_KEY_ENGINE]
    data = await request.json()
    message = data.get("message", "").strip()
    if "context" in data:
        engine.context = [tuple(pair) for pair in data["context"]]
    if not message:
        return web.json_response({"error": "Empty message"}, status=400)
    # Candidates per generation (DPO duel mode sends n>=2), batched
    # concurrently in one generator pass. Recurrent models are further
    # clamped to the cache slots allocated at load (see batch_slots).
    n = data.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 8:
        n = 1
    # Optional per-candidate system-prompt overrides (DPO generation
    # bias); None/"" entries fall back to the trained prompt.
    system_prompts = data.get("system_prompts")
    if system_prompts is not None:
        if (not isinstance(system_prompts, list)
                or len(system_prompts) != n
                or not all(s is None or isinstance(s, str)
                           for s in system_prompts)):
            return web.json_response(
                {"error": "system_prompts must be a list of n strings/nulls"},
                status=400,
            )

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
    async for event in engine.generate(message, prefix=prefix, n=n,
                                       system_prompts=system_prompts):
        sse_data = f"data: {json.dumps(event)}\n\n"
        await response.write(sse_data.encode("utf-8"))

    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def handle_stop(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    engine.cancel()
    return web.json_response({"ok": True})


async def handle_clear(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    engine.clear_context()
    return web.json_response({"ok": True})


async def handle_session_save(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    session_data = engine.save_session()
    return web.json_response(
        session_data,
        headers={
            "Content-Disposition": 'attachment; filename="chat_session.json"',
        },
    )


async def handle_session_load(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    data = await request.json()
    engine.load_session(data)
    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# Model management routes
# ---------------------------------------------------------------------------

def _parse_cpu_offload(raw) -> dict:
    """Coerce the load panel's CPU-offload block into clean numbers.

    Anything missing, negative or unparseable becomes 0, which the engine
    reads as "don't pass the argument at all".
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, cast in (
        ("moe_layers", int), ("moe_threads", int),
        ("draft_moe_layers", int), ("draft_moe_threads", int),
        ("cache_gb", float),
    ):
        try:
            val = cast(raw.get(key) or 0)
        except (TypeError, ValueError):
            val = 0
        out[key] = max(val, 0)
    return out


async def handle_gpus(request: web.Request) -> web.Response:
    gpus = ChatEngine.detect_gpus()
    return web.json_response({
        "gpus": gpus,
        "cpu_cores": os.cpu_count() or 0,
        # Which CPU-offload knobs this exllamav3 build supports, so the load
        # panel can disable controls it can't honor.
        "cpu_offload": cpu_offload_support(),
        # Whether this build can load a PLE model's n-gram table into RAM (-ngr)
        "ngram_ram": ngram_ram_support(),
    })


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
        "moe_offload": _moe_offload_eligible(browse_path) if is_model else None,
    })


def _moe_offload_eligible(model_dir: Path) -> dict:
    """Can this model's experts actually be offloaded to the CPU?

    exllamav3 only offloads mul1-codebook experts — everything else silently
    falls back to the GPU with a note on the server console, which looks
    exactly like the setting being ignored. Read the safetensors index
    (cheap, no weights loaded) so the load panel can say so up front.

    Returns {"moe": bool, "mul1": bool}; "moe" false means the control is
    irrelevant rather than broken.
    """
    try:
        index = model_dir / "model.safetensors.index.json"
        if not index.is_file():
            return {"moe": False, "mul1": False}
        keys = json.loads(index.read_text("utf-8")).get("weight_map", {}).keys()
        moe = any("expert" in k.lower() for k in keys)
        mul1 = any(k.endswith(".mul1") for k in keys)
        return {"moe": moe, "mul1": mul1}
    except Exception:
        return {"moe": False, "mul1": False}


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


async def handle_model_load(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
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
    lora_dirs = data.get("lora_dirs") or []
    if not isinstance(lora_dirs, list):
        return web.json_response(
            {"ok": False, "error": "lora_dirs must be a list of directories"},
            status=400,
        )
    for lora_dir in lora_dirs:
        if not isinstance(lora_dir, str) or not lora_dir.strip():
            return web.json_response(
                {"ok": False, "error": "Each LoRA directory must be a non-empty string"},
                status=400,
            )
        lp = Path(lora_dir)
        if not lp.is_dir():
            return web.json_response(
                {"ok": False, "error": f"LoRA directory not found: {lora_dir}"},
                status=400,
            )
        if not (lp / "adapter_config.json").is_file():
            return web.json_response(
                {"ok": False, "error": f"LoRA adapter_config.json missing: {lora_dir}"},
                status=400,
            )

    lora_weights = data.get("lora_weights") or [1.0] * len(lora_dirs)

    draft_model_dir = (data.get("draft_model_dir") or "").strip() or None
    use_mtp = bool(data.get("use_mtp"))
    ngram_min = data.get("ngram_min") or 0
    if not isinstance(ngram_min, int) or isinstance(ngram_min, bool) or ngram_min < 0:
        return web.json_response(
            {"ok": False, "error": "ngram_min must be a non-negative integer"},
            status=400,
        )
    if use_mtp and draft_model_dir:
        return web.json_response(
            {"ok": False, "error": "Specify either a draft model directory or MTP drafting, not both"},
            status=400,
        )
    if ngram_min and (use_mtp or draft_model_dir):
        return web.json_response(
            {"ok": False, "error": "Specify only one of: draft model directory, MTP drafting, or n-gram drafting"},
            status=400,
        )
    if draft_model_dir:
        dp = Path(draft_model_dir)
        if not dp.is_dir():
            return web.json_response(
                {"ok": False, "error": f"Draft model directory not found: {draft_model_dir}"},
                status=400,
            )
        if not (dp / "config.json").is_file():
            return web.json_response(
                {"ok": False, "error": "Draft model config.json missing — not a valid model directory"},
                status=400,
            )

    # Duel batch size (persisted in ui.json by the ratings UI) sets the
    # recurrent-model cache slots allocated at load. Clamp to the same
    # ceiling as the /api/chat n cap.
    cfg = await asyncio.to_thread(_load_config)
    batch_slots = cfg.get("ratings_batch", ChatEngine.BATCH_SLOTS)
    if not isinstance(batch_slots, int) or isinstance(batch_slots, bool):
        batch_slots = ChatEngine.BATCH_SLOTS
    batch_slots = max(1, min(8, batch_slots))

    try:
        await asyncio.to_thread(
            engine.load_model,
            model_dir=model_dir,
            lora_dirs=lora_dirs,
            lora_weights=lora_weights,
            draft_model_dir=draft_model_dir,
            use_mtp=use_mtp,
            ngram_min=ngram_min,
            devices=data.get("devices"),
            device_ratios=data.get("device_ratios"),
            cache_size=data.get("cache_size"),
            cache_quant=data.get("cache_quant"),
            batch_slots=batch_slots,
            cpu_offload=_parse_cpu_offload(data.get("cpu_offload")),
            ngram_ram=bool(data.get("ngram_ram")),
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
    engine: ChatEngine = request.app[_KEY_ENGINE]
    engine.unload()
    return web.json_response({"ok": True})


async def handle_loras_apply(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    if not engine.is_loaded:
        return web.json_response(
            {"ok": False, "error": "No model loaded"}, status=400,
        )
    if engine.is_generating:
        return web.json_response(
            {"ok": False, "error": "Cannot update LoRAs while generating"},
            status=409,
        )
    data = await request.json()
    lora_configs = data.get("loras", [])
    if not isinstance(lora_configs, list):
        return web.json_response(
            {"ok": False, "error": "loras must be a list"}, status=400,
        )
    for cfg in lora_configs:
        if not isinstance(cfg, dict):
            return web.json_response(
                {"ok": False, "error": "Each LoRA must be an object"},
                status=400,
            )
        d = cfg.get("dir", "")
        if not isinstance(d, str) or not d.strip():
            return web.json_response(
                {"ok": False, "error": "Each LoRA must have a non-empty 'dir'"},
                status=400,
            )
        lp = Path(d)
        if not lp.is_dir():
            return web.json_response(
                {"ok": False, "error": f"LoRA directory not found: {d}"},
                status=400,
            )
        if not (lp / "adapter_config.json").is_file():
            return web.json_response(
                {"ok": False, "error": f"LoRA adapter_config.json missing: {d}"},
                status=400,
            )
        w = cfg.get("weight", 1.0)
        if not isinstance(w, (int, float)):
            return web.json_response(
                {"ok": False, "error": f"Invalid weight for {d}"}, status=400,
            )
    try:
        await asyncio.to_thread(engine.update_loras, lora_configs)
        return web.json_response({
            "ok": True,
            "status": engine.get_status(),
        })
    except Exception as e:
        return web.json_response(
            {"ok": False, "error": str(e)}, status=500,
        )


async def handle_draft_load(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    if not engine.is_loaded:
        return web.json_response(
            {"ok": False, "error": "No model loaded"}, status=400,
        )
    if engine.is_generating:
        return web.json_response(
            {"ok": False, "error": "Cannot load draft model while generating"},
            status=409,
        )
    data = await request.json()
    mtp = bool(data.get("mtp"))
    ngram_min = data.get("ngram_min") or 0
    if not isinstance(ngram_min, int) or isinstance(ngram_min, bool) or ngram_min < 0:
        return web.json_response(
            {"ok": False, "error": "ngram_min must be a non-negative integer"},
            status=400,
        )
    draft_dir = (data.get("draft_model_dir") or "").strip()
    if sum([bool(mtp), bool(draft_dir), bool(ngram_min)]) > 1:
        return web.json_response(
            {"ok": False, "error": "Specify only one of: draft model directory, MTP drafting, or n-gram drafting"},
            status=400,
        )
    if not mtp and not ngram_min:
        if not draft_dir:
            return web.json_response(
                {"ok": False, "error": "draft_model_dir is required"}, status=400,
            )
        dp = Path(draft_dir)
        if not dp.is_dir():
            return web.json_response(
                {"ok": False, "error": f"Draft model directory not found: {draft_dir}"},
                status=400,
            )
        if not (dp / "config.json").is_file():
            return web.json_response(
                {"ok": False, "error": "Draft model config.json missing — not a valid model directory"},
                status=400,
            )
    try:
        reloaded = False
        if engine.needs_load_time_draft(draft_dir or None):
            # Recurrent models size draft headroom at cache creation, so
            # enabling a draft source means a full reload with the draft
            # configured. Reuse the current load parameters and carry the
            # chat settings across (load_model resets them).
            saved_settings = engine.settings
            await asyncio.to_thread(
                engine.load_model,
                model_dir=engine.model_dir,
                lora_dirs=list(engine.lora_dirs),
                lora_weights=list(engine.lora_weights),
                draft_model_dir=draft_dir or None,
                use_mtp=mtp,
                ngram_min=ngram_min,
                devices=engine._devices or None,
                device_ratios=engine._device_ratios,
                cache_size=engine._cache_size,
                cache_quant=engine._cache_quant,
                batch_slots=engine.batch_slots,
                cpu_offload=dict(engine._cpu_offload),
                ngram_ram=engine.ngram_ram,
            )
            engine.settings = saved_settings
            reloaded = True
        else:
            await asyncio.to_thread(
                engine.load_draft, draft_dir or None, mtp, ngram_min,
            )
        return web.json_response({
            "ok": True,
            "reloaded": reloaded,
            "status": engine.get_status(),
        })
    except Exception as e:
        return web.json_response(
            {"ok": False, "error": str(e)}, status=500,
        )


async def handle_draft_unload(request: web.Request) -> web.Response:
    engine: ChatEngine = request.app[_KEY_ENGINE]
    if not engine.is_loaded:
        return web.json_response(
            {"ok": False, "error": "No model loaded"}, status=400,
        )
    if engine.is_generating:
        return web.json_response(
            {"ok": False, "error": "Cannot unload draft model while generating"},
            status=409,
        )
    try:
        await asyncio.to_thread(engine.unload_draft)
        return web.json_response({
            "ok": True,
            "status": engine.get_status(),
        })
    except Exception as e:
        return web.json_response(
            {"ok": False, "error": str(e)}, status=500,
        )


async def handle_ui_launch(request: web.Request) -> web.Response:
    """Shut down the chat server and launch the dashboard UI in its place."""
    engine: ChatEngine = request.app[_KEY_ENGINE]
    if engine.is_loaded:
        return web.json_response(
            {"error": "Unload the model before switching to dashboard"},
            status=409,
        )
    if engine.is_generating:
        return web.json_response(
            {"error": "Cannot switch while generating"}, status=409,
        )

    host = request.app.get(_KEY_HOST, "127.0.0.1")
    port = request.app.get(_KEY_PORT, 8800)

    import shutil
    ezexl3_bin = shutil.which("ezexl3")
    if ezexl3_bin:
        ui_cmd = [ezexl3_bin, "ui", "--port", str(port), "--host", host,
                  "--no-browser"]
    else:
        ui_cmd = [sys.executable, "-m", "ezexl3", "ui", "--port", str(port),
                  "--host", host, "--no-browser"]

    # Store the command — the pre-registered cleanup handler will spawn it
    # after aiohttp has released the port.
    request.app[_KEY_SPAWN] = ui_cmd

    # Schedule graceful shutdown — SIGINT lets aiohttp close connections
    # cleanly, release the port, then run cleanup (which spawns UI).
    async def _graceful_exit():
        await asyncio.sleep(1)
        os.kill(os.getpid(), signal.SIGINT)
    asyncio.ensure_future(_graceful_exit())

    return web.json_response({"ok": True, "url": f"http://{host}:{port}"})


# ---------------------------------------------------------------------------
# Persistent user config (shared with dashboard: ~/.config/ezexl3/ui.json)
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
    # chat server and the dashboard share this file and do run at once
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
# Preference ratings (KTO / DPO data collection)
# ---------------------------------------------------------------------------

def _ratings_store() -> RatingsStore:
    cfg = _load_config()
    root = (cfg.get("ratings_dir") or "").strip() or default_datasets_dir()
    return RatingsStore(root)


async def handle_ratings_get(request: web.Request) -> web.Response:
    dataset = request.query.get("dataset", "chat")
    if not valid_dataset_name(dataset):
        return web.json_response(
            {"error": f"Invalid dataset name: {dataset!r}"}, status=400,
        )
    store = await asyncio.to_thread(_ratings_store)

    def _snapshot():
        state = store.state(dataset)
        return {
            "dataset": dataset,
            "dir": str(store.root),
            "datasets": store.list_datasets(),
            **state,
        }

    return web.json_response(await asyncio.to_thread(_snapshot))


async def handle_rate(request: web.Request) -> web.Response:
    """Record preference data for chat messages.

    Body: {dataset, prompt, kto?: {node_id, completion, label},
           pair?: {chosen: {node_id, content},
                   rejected: {node_id, content}, remove?},
           bulk?: {completions: [{node_id, content}], target,
                   gen_system?, source_row?}}

    kto upserts one thumbs row (KTO mode); pair upserts the DPO row for a
    two-candidate duel (DPO mode), keyed by the unordered node-id duo;
    bulk appends one row per completion with the generated text on the
    *target* side (review-flow "Save all"). Returns the updated snapshot.
    """
    engine: ChatEngine = request.app[_KEY_ENGINE]
    data = await request.json()

    dataset = data.get("dataset", "chat")
    if not valid_dataset_name(dataset):
        return web.json_response(
            {"error": f"Invalid dataset name: {dataset!r}"}, status=400,
        )
    kto = data.get("kto")
    pair = data.get("pair")
    bulk = data.get("bulk")
    if not (kto or pair or bulk):
        return web.json_response(
            {"error": "Nothing to record: need kto, pair, or bulk"},
            status=400,
        )
    prompt = data.get("prompt")
    err = validate_prompt(prompt)
    if err:
        return web.json_response({"error": err}, status=400)

    if kto is not None:
        if not kto.get("node_id") or not isinstance(kto.get("completion"), str):
            return web.json_response(
                {"error": "kto needs node_id and completion"}, status=400,
            )
        if kto.get("label") not in (True, False, None):
            return web.json_response(
                {"error": "kto label must be true, false, or null"}, status=400,
            )
    if pair is not None:
        chosen = pair.get("chosen") or {}
        rejected = pair.get("rejected") or {}
        if not chosen.get("node_id") or not rejected.get("node_id"):
            return web.json_response(
                {"error": "pair needs chosen.node_id and rejected.node_id"},
                status=400,
            )
        if not pair.get("remove") and (
            not isinstance(chosen.get("content"), str)
            or not isinstance(rejected.get("content"), str)
        ):
            return web.json_response(
                {"error": "pair needs chosen and rejected content"}, status=400,
            )
        for side in (chosen, rejected):
            gs = side.get("gen_system")
            if gs is not None and not isinstance(gs, str):
                return web.json_response(
                    {"error": "gen_system must be a string or null"},
                    status=400,
                )
    if bulk is not None:
        comps = bulk.get("completions")
        if (not isinstance(comps, list) or not comps
                or not all(isinstance(c, dict)
                           and isinstance(c.get("content"), str)
                           for c in comps)):
            return web.json_response(
                {"error": "bulk needs a non-empty completions list "
                          "of {node_id, content}"}, status=400,
            )
        if bulk.get("target") not in ("chosen", "rejected"):
            return web.json_response(
                {"error": "bulk target must be 'chosen' or 'rejected'"},
                status=400,
            )
        gs = bulk.get("gen_system")
        if gs is not None and not isinstance(gs, str):
            return web.json_response(
                {"error": "gen_system must be a string or null"}, status=400,
            )
        src = bulk.get("source_row")
        if src is not None and not isinstance(src, dict):
            return web.json_response(
                {"error": "source_row must be an object or null"}, status=400,
            )

    # Full model dir as provenance — the basename alone is ambiguous for
    # layouts like .../Llama-3.2-3B-Instruct/4.
    model = engine.model_dir or engine.model_name

    def _apply():
        store = _ratings_store()
        if kto is not None:
            store.rate_kto(dataset, kto["node_id"], prompt,
                           kto["completion"], kto.get("label"), model)
        if pair is not None:
            store.rate_dpo_pair(dataset, prompt, pair.get("chosen") or {},
                                pair.get("rejected") or {}, model,
                                remove=bool(pair.get("remove")))
        if bulk is not None:
            store.add_bulk_rows(dataset, prompt, bulk["completions"],
                                bulk["target"], model,
                                source_row=bulk.get("source_row"),
                                gen_system=bulk.get("gen_system"))
        state = store.state(dataset)
        return {
            "dataset": dataset,
            "dir": str(store.root),
            "datasets": store.list_datasets(),
            **state,
        }

    try:
        return web.json_response(await asyncio.to_thread(_apply))
    except (OSError, ValueError) as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_ratings_bulk(request: web.Request) -> web.Response:
    """SSE endpoint: unattended bulk generation into a preference dataset.

    Body: {dataset, rows: [{prompt, chosen?, rejected?, id?}], n,
           system_prompt?, target, carry?, strip_think?}

    Every row's prompt becomes a fresh single-turn conversation; the
    engine batches all rows × n jobs across prompts in one generator
    pool. Each completion is written to <dataset>.dpo.jsonl as it
    finishes (generated text on the *target* side; with carry=true the
    opposite side comes from the source row). Streams progress events;
    POST /api/stop cancels, keeping everything already written.
    """
    engine: ChatEngine = request.app[_KEY_ENGINE]
    data = await request.json()

    dataset = data.get("dataset", "chat")
    if not valid_dataset_name(dataset):
        return web.json_response(
            {"error": f"Invalid dataset name: {dataset!r}"}, status=400,
        )
    rows = data.get("rows")
    if (not isinstance(rows, list) or not rows
            or not all(isinstance(r, dict)
                       and isinstance(r.get("prompt"), str)
                       and r["prompt"].strip()
                       for r in rows)):
        return web.json_response(
            {"error": "rows must be a non-empty list of {prompt, …}"},
            status=400,
        )
    n = data.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 8:
        return web.json_response(
            {"error": "n must be an integer in 1..8"}, status=400,
        )
    target = data.get("target")
    if target not in ("chosen", "rejected"):
        return web.json_response(
            {"error": "target must be 'chosen' or 'rejected'"}, status=400,
        )
    system_prompt = data.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        return web.json_response(
            {"error": "system_prompt must be a string or null"}, status=400,
        )
    carry = bool(data.get("carry"))
    strip = bool(data.get("strip_think"))
    if not engine.is_loaded:
        return web.json_response({"error": "Model not loaded"}, status=400)

    # The trained prompt column: main system prompt + the row's user turn.
    main_sys = (engine.settings.system_prompt or "").strip()

    def _turns_for(prompt_text: str) -> list:
        turns = []
        if main_sys:
            turns.append({"role": "system", "content": main_sys})
        turns.append({"role": "user", "content": prompt_text})
        return turns

    # gen_system metadata: recorded only when it differs from the
    # trained prompt (mirrors duel semantics).
    gen_system = (system_prompt or "").strip() or None

    model = engine.model_dir or engine.model_name
    store = await asyncio.to_thread(_ratings_store)

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

    async def _send(event: dict):
        await response.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))

    prompts = [r["prompt"].strip() for r in rows]
    done_by_item: dict[int, list] = {}
    saved_items = set()
    rows_written = 0
    items_done = 0

    async def _flush_item(i: int):
        """Write item i's finished completions in one store call."""
        nonlocal rows_written, items_done
        comps = [c for c in done_by_item.get(i, []) if c["content"].strip()]
        saved_items.add(i)
        items_done += 1
        if not comps:
            return
        written = await asyncio.to_thread(
            store.add_bulk_rows, dataset, _turns_for(prompts[i]), comps,
            target, model,
            source_row=rows[i] if carry else {"id": rows[i].get("id")},
            gen_system=gen_system,
        )
        rows_written += written
        preview = comps[-1]["content"]
        await _send({
            "type": "saved", "item": i, "rows": written,
            "rows_written": rows_written, "items_done": items_done,
            "total_items": len(prompts),
            "preview": preview[:300],
        })

    async for event in engine.generate_bulk(prompts, n=n,
                                            system_prompt=system_prompt):
        et = event.get("type")
        if et == "row_done":
            i = event["item"]
            if event.get("eos_reason") != "cancelled":
                text = event.get("text", "")
                if strip:
                    text = strip_think_text(text)
                done_by_item.setdefault(i, []).append(
                    {"content": text.strip(), "node_id": None})
            else:
                done_by_item.setdefault(i, [])
            if (len(done_by_item[i]) >= n
                    or event.get("eos_reason") == "cancelled") \
                    and i not in saved_items:
                await _flush_item(i)
        elif et == "progress":
            await _send({**event, "rows_written": rows_written,
                         "items_done": items_done,
                         "total_items": len(prompts)})
        elif et == "error":
            await _send(event)

    # A cancelled run leaves items with some finished completions but no
    # flush trigger — save what completed.
    for i in list(done_by_item):
        if i not in saved_items:
            await _flush_item(i)

    await _send({"type": "bulk_done", "rows_written": rows_written,
                 "items_done": items_done, "total_items": len(prompts)})
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


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
    draft_model_dir: str | None = None,
    use_mtp: bool = False,
    ngram_min: int = 0,
    ngram_ram: bool = False,
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
        draft_model_dir=draft_model_dir,
        use_mtp=use_mtp,
        ngram_min=ngram_min,
        ngram_ram=ngram_ram,
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
        app[_KEY_HOST] = host
        app[_KEY_PORT] = port
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
