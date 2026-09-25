"""Auto GPU split leaves room for LoRA adapters, and a failed load never
leaves a half-resident model.

LoRA tensors are copied onto whichever device their target layer was
split to, AFTER the split, so exllamav3's own "-gs auto" (fills GPU 0 to
~0.5 GB) OOMs there while the next card is empty. The chat engine now
hands exllamav3 explicit per-device budgets that subtract the adapters'
fp16 footprint, and any exception during load() unloads everything.
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.cuda.is_available.return_value = False
    _mock_torch.cuda.device_count.return_value = 0
    sys.modules["torch"] = _mock_torch

if "exllamav3" not in sys.modules:
    sys.modules["exllamav3"] = MagicMock()

from ezexl3.chat import inference  # noqa: E402

GIB = 1024 ** 3


def _write_safetensors(path, tensors):
    """Minimal safetensors writer: {name: (dtype, n_elems, bytes_per)}."""
    header, offset = {}, 0
    for name, (dtype, n, per) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [n],
                        "data_offsets": [offset, offset + n * per]}
        offset += n * per
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * offset)


class LoraVramBytesTests(unittest.TestCase):

    def test_counts_elements_at_fp16(self):
        d = tempfile.mkdtemp()
        # 1000 fp32 + 500 fp16 elements -> all fp16 on the GPU = 3000 bytes
        _write_safetensors(os.path.join(d, "adapter_model.safetensors"), {
            "a.lora_A.weight": ("F32", 1000, 4),
            "a.lora_B.weight": ("F16", 500, 2),
        })
        self.assertEqual(inference._lora_vram_bytes([d]), 3000)

    def test_sums_sidecar_files_and_multiple_dirs(self):
        d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        _write_safetensors(os.path.join(d1, "adapter_model.safetensors"),
                           {"x": ("BF16", 10, 2)})
        _write_safetensors(os.path.join(d1, "lora_modules.safetensors"),
                           {"lm_head.lora_a": ("F32", 10, 4)})
        _write_safetensors(os.path.join(d2, "modules_to_save.safetensors"),
                           {"lm_head.weight": ("F16", 10, 2)})
        self.assertEqual(inference._lora_vram_bytes([d1, d2]), 60)

    def test_ignores_trainer_state_and_pissa_init(self):
        d = tempfile.mkdtemp()
        _write_safetensors(os.path.join(d, "pissa_init.safetensors"),
                           {"w": ("F32", 1_000_000, 4)})
        with open(os.path.join(d, "trainer_state.pt"), "wb") as f:
            f.write(b"\0" * 4096)
        self.assertEqual(inference._lora_vram_bytes([d]), 0)

    def test_empty_and_missing(self):
        self.assertEqual(inference._lora_vram_bytes(None), 0)
        self.assertEqual(inference._lora_vram_bytes([]), 0)
        self.assertEqual(inference._lora_vram_bytes(["/nonexistent/lora"]), 0)

    def test_unparseable_file_falls_back_to_disk_size(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
            f.write(b"garbage")
        self.assertEqual(inference._lora_vram_bytes([d]), 7)


class AutoGpuSplitTests(unittest.TestCase):

    def _split(self, free_gb, extra=0, **kw):
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.mem_get_info.side_effect = [
            (int(g * GIB), 24 * GIB) for g in free_gb]
        with patch.object(inference.torch, "cuda", cuda):
            return inference._auto_gpu_split(len(free_gb), extra, **kw)

    def test_subtracts_headroom_and_lora_from_each_device(self):
        # 23 GB free each, 3.5 GB adapter: 23 - 0.5 - 3.5 = 19 GB budgets
        self.assertEqual(self._split([23, 23], int(3.5 * GIB)), "19.00,19.00")

    def test_uses_free_not_total(self):
        # Another process holds most of GPU 0
        self.assertEqual(self._split([4, 23]), "3.50,22.50")

    def test_device_without_room_is_zeroed_not_negative(self):
        self.assertEqual(self._split([2, 23], int(3 * GIB)), "0.00,19.50")

    def test_no_room_anywhere_returns_none(self):
        self.assertIsNone(self._split([1, 1], int(3 * GIB)))

    def test_no_cuda_returns_none(self):
        cuda = MagicMock()
        cuda.is_available.return_value = False
        with patch.object(inference.torch, "cuda", cuda):
            self.assertIsNone(inference._auto_gpu_split(2, 0))


class LoadUsesSplitTests(unittest.TestCase):

    def _engine(self, devices, lora_dirs=None, ratios=None):
        e = inference.ChatEngine(model_dir="/m", devices=devices,
                                 device_ratios=ratios)
        e.lora_dirs = lora_dirs or []
        e._load_inner = MagicMock()
        return e

    def _load(self, engine, split="19.00,19.00"):
        cuda = MagicMock()
        cuda.is_available.return_value = True
        with patch.object(inference, "_import_model_init", MagicMock()), \
             patch.object(inference, "_auto_gpu_split", return_value=split) as auto, \
             patch.object(inference, "_lora_vram_bytes", return_value=123) as lb, \
             patch.object(inference.torch, "cuda", cuda):
            engine.load()
        return auto, lb

    def test_multi_gpu_auto_passes_computed_budgets(self):
        e = self._engine([0, 1], ["/lora"])
        auto, lb = self._load(e)
        lb.assert_called_once_with(["/lora"])
        auto.assert_called_once_with(2, 123)
        self.assertEqual(e._load_inner.call_args.args[2], "19.00,19.00")

    def test_user_split_wins(self):
        e = self._engine([0, 1], ["/lora"], ratios="10,20")
        auto, _ = self._load(e)
        auto.assert_not_called()
        self.assertEqual(e._load_inner.call_args.args[2], "10,20")

    def test_single_gpu_skips_split(self):
        e = self._engine([0], ["/lora"])
        auto, _ = self._load(e)
        auto.assert_not_called()
        self.assertIsNone(e._load_inner.call_args.args[2])

    def test_unmeasurable_falls_back_to_exllamav3_auto(self):
        e = self._engine([0, 1])
        self._load(e, split=None)
        self.assertIsNone(e._load_inner.call_args.args[2])


class FailedLoadCleanupTests(unittest.TestCase):

    def test_exception_mid_load_unloads_everything(self):
        e = inference.ChatEngine(model_dir="/m", devices=[0])
        e.lora_dirs = ["/lora"]

        def half_load(*a):
            # Model, cache and generator are up; the LoRA step then OOMs.
            e.model = object()
            e.cache = object()
            e.generator = object()
            raise RuntimeError("CUDA out of memory")

        e._load_inner = half_load
        with patch.object(inference, "_import_model_init", MagicMock()), \
             patch.object(inference.torch, "cuda", MagicMock()):
            with self.assertRaises(RuntimeError):
                e.load()
        self.assertFalse(e.is_loaded)
        self.assertIsNone(e.model)
        self.assertIsNone(e.cache)
        self.assertEqual(e.loras, [])
        self.assertEqual(e.lora_dirs, [])

    def test_load_model_unloads_even_without_generator(self):
        # Leftover from a load that died before the generator existed.
        e = inference.ChatEngine(model_dir="/m", devices=[0])
        e.model = object()
        e.unload = MagicMock(wraps=e.unload)
        e.load = MagicMock()
        e.load_model("/m2")
        e.unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class FailedLoadServerTests(unittest.TestCase):
    """POST /api/model/load: a load that raises still returns 500, and the
    engine is unloaded once the traceback is gone."""

    def test_handler_unloads_after_failure(self):
        import asyncio
        import tempfile
        from aiohttp.test_utils import TestClient, TestServer
        from ezexl3.chat.server import create_app

        model_dir = tempfile.mkdtemp()
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write("{}")
        engine = inference.ChatEngine(model_dir=None)
        engine.load_model = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        engine.unload = MagicMock()

        async def run():
            with patch("ezexl3.chat.server._load_config", return_value={}):
                async with TestClient(TestServer(create_app(engine))) as client:
                    resp = await client.post(
                        "/api/model/load", json={"model_dir": model_dir})
                    return resp.status, await resp.json()

        status, data = asyncio.run(run())
        self.assertEqual(status, 500)
        self.assertFalse(data["ok"])
        self.assertIn("out of memory", data["error"])
        engine.unload.assert_called_once()

    def test_load_outlives_client_disconnect(self):
        """A client that drops mid-load (Firefox on a network change) must
        not orphan the load: it still finishes, a failure still unloads,
        and /api/status reports the outcome."""
        import asyncio
        import threading
        from aiohttp.test_utils import TestClient, TestServer
        from ezexl3.chat.server import create_app

        model_dir = tempfile.mkdtemp()
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write("{}")
        release = threading.Event()

        def slow_fail(**kwargs):
            release.wait(10)
            raise RuntimeError("CUDA out of memory")

        engine = inference.ChatEngine(model_dir=None)
        engine.load_model = MagicMock(side_effect=slow_fail)
        engine.unload = MagicMock()
        engine.get_status = MagicMock(return_value={"loaded": False})

        async def run():
            with patch("ezexl3.chat.server._load_config", return_value={}):
                async with TestClient(TestServer(create_app(engine))) as client:
                    req = asyncio.ensure_future(client.post(
                        "/api/model/load", json={"model_dir": model_dir}))
                    for _ in range(100):
                        if engine.load_model.called:
                            break
                        await asyncio.sleep(0.02)
                    req.cancel()  # browser drops the connection
                    busy = await (await client.get("/api/status")).json()
                    second = await client.post(
                        "/api/model/load", json={"model_dir": model_dir})
                    release.set()
                    for _ in range(100):
                        done = await (await client.get("/api/status")).json()
                        if not done["loading"]:
                            break
                        await asyncio.sleep(0.02)
                    return busy, second.status, done

        busy, second_status, done = asyncio.run(run())
        self.assertTrue(busy["loading"])
        self.assertEqual(second_status, 409)
        self.assertFalse(done["loading"])
        self.assertIn("out of memory", done["load_error"])
        engine.unload.assert_called_once()
