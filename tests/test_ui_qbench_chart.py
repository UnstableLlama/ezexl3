"""The Results tab serves qbench's own chart renders, not a home-grown SVG."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from ezexl3 import qbench
from ezexl3.ui.server import handle_qbench_chart


def _get(query: str):
    return asyncio.run(handle_qbench_chart(make_mocked_request("GET", f"/api/qbench-chart?{query}")))


class ChartListing(unittest.TestCase):
    def test_lists_existing_charts_in_display_order_preferring_qbench_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            qb = Path(tmp, "qbench")
            qb.mkdir()
            (qb / "qb_kld_hist_combined.png").write_bytes(b"hist")
            (qb / "qb_kld.png").write_bytes(b"kld-fresh")
            Path(tmp, "qb_kld.png").write_bytes(b"kld-root-copy")
            Path(tmp, "qb_ppl.png").write_bytes(b"ppl-root-only")
            Path(tmp, "unrelated.png").write_bytes(b"nope")

            items = qbench.list_charts(tmp)
            self.assertEqual([i["file"] for i in items], ["qb_kld.png", "qb_ppl.png", "qb_kld_hist_combined.png"])
            self.assertEqual(items[0]["label"], "KL vs bpw")
            self.assertTrue(all(isinstance(i["mtime"], int) for i in items))
            self.assertEqual(qbench.chart_path(tmp, "qb_kld.png"), str(qb / "qb_kld.png"))
            self.assertEqual(qbench.chart_path(tmp, "qb_ppl.png"), os.path.join(tmp, "qb_ppl.png"))
            self.assertIsNone(qbench.chart_path(tmp, "qb_kld_spread.png"))
            self.assertIsNone(qbench.chart_path(tmp, "unrelated.png"))

    def test_ui_charts_are_all_qbench_outputs(self):
        outputs = set(qbench._OUTPUT_FILES.values())
        for name, _ in qbench.UI_CHARTS:
            self.assertIn(name, outputs)


class ChartEndpoint(unittest.TestCase):
    def test_listing_and_serving(self):
        with tempfile.TemporaryDirectory() as tmp:
            qb = Path(tmp, "qbench")
            qb.mkdir()
            (qb / "qb_kld.png").write_bytes(b"png")

            resp = _get(f"model_dir={tmp}")
            self.assertEqual(resp.status, 200)
            self.assertIn(b'"qb_kld.png"', resp.body)

            resp = _get(f"model_dir={tmp}&file=qb_kld.png")
            self.assertIsInstance(resp, web.FileResponse)
            self.assertEqual(str(resp._path), str(qb / "qb_kld.png"))
            self.assertEqual(resp.headers["Content-Type"], "image/png")

    def test_only_known_chart_names_are_reachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            qb = Path(tmp, "qbench")
            qb.mkdir()
            (qb / "project.yml").write_text("title: x\n")
            (qb / "qb_ppl.png").write_bytes(b"png")
            for fname in ("project.yml", "../config.json", "qb_kld.png", "/etc/passwd"):
                resp = _get(f"model_dir={tmp}&file={fname}")
                self.assertEqual(resp.status, 404, fname)
                self.assertNotIsInstance(resp, web.FileResponse, fname)

    def test_missing_model_dir_is_a_client_error(self):
        self.assertEqual(_get("").status, 400)
        resp = _get("model_dir=/nonexistent/path")
        self.assertEqual(resp.status, 200)
        self.assertIn(b'"items": []', resp.body)


if __name__ == "__main__":
    unittest.main()
