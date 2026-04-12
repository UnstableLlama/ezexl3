"""Tests for the perf detail database (ezexl3/perf_db.py),
the extract_perf_detail() extractor (ezexl3/evals.py),
and perf chart generation (ezexl3/graph_svg.py)."""

import os
import tempfile

import pytest

from ezexl3.perf_db import (
    available_bpws,
    default_perf_db_path,
    read_perf_data,
    upsert_perf_results,
)
from ezexl3.evals import extract_perf_detail


# ---------------------------------------------------------------------------
# perf_db basic operations
# ---------------------------------------------------------------------------

class TestPerfDbUpsertAndRead:
    def test_upsert_and_read_single_bpw(self, tmp_path):
        db = str(tmp_path / "test.db")
        prefill = [(256, 1715.46), (512, 2410.60), (1024, 2839.02)]
        gen = [(0, 49.43), (256, 45.97), (512, 45.25)]
        upsert_perf_results(db, "4", prefill, gen)

        data = read_perf_data(db, "4")
        assert "4" in data
        assert len(data["4"]["prefill"]) == 3
        assert len(data["4"]["generation"]) == 3
        assert data["4"]["prefill"][0]["context_length"] == 256
        assert data["4"]["prefill"][0]["tokens_per_second"] == 1715.46
        assert data["4"]["generation"][0]["context_length"] == 0
        assert data["4"]["generation"][0]["tokens_per_second"] == 49.43

    def test_upsert_replaces_existing_bpw(self, tmp_path):
        db = str(tmp_path / "test.db")
        upsert_perf_results(db, "4", [(256, 1000.0)], [(0, 50.0)])
        # Overwrite with new data
        upsert_perf_results(db, "4", [(256, 2000.0), (512, 3000.0)], [(0, 60.0)])

        data = read_perf_data(db, "4")
        assert len(data["4"]["prefill"]) == 2
        assert data["4"]["prefill"][0]["tokens_per_second"] == 2000.0

    def test_read_all_bpws(self, tmp_path):
        db = str(tmp_path / "test.db")
        upsert_perf_results(db, "4", [(256, 1000.0)], [(0, 50.0)])
        upsert_perf_results(db, "6", [(256, 800.0)], [(0, 45.0)])
        upsert_perf_results(db, "bf16", [(256, 1500.0)], [(0, 55.0)])

        data = read_perf_data(db)
        assert set(data.keys()) == {"4", "6", "bf16"}

    def test_read_nonexistent_bpw_returns_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        upsert_perf_results(db, "4", [(256, 1000.0)], [(0, 50.0)])

        data = read_perf_data(db, "8")
        assert data == {}

    def test_read_nonexistent_db_returns_empty(self, tmp_path):
        data = read_perf_data(str(tmp_path / "nope.db"))
        assert data == {}

    def test_empty_prefill_or_gen(self, tmp_path):
        db = str(tmp_path / "test.db")
        upsert_perf_results(db, "4", [], [(0, 50.0)])

        data = read_perf_data(db, "4")
        assert data["4"]["prefill"] == []
        assert len(data["4"]["generation"]) == 1


class TestAvailableBpws:
    def test_returns_sorted_bpws(self, tmp_path):
        db = str(tmp_path / "test.db")
        upsert_perf_results(db, "6", [(256, 800.0)], [(0, 45.0)])
        upsert_perf_results(db, "2", [(256, 1200.0)], [(0, 55.0)])
        upsert_perf_results(db, "4", [(256, 1000.0)], [(0, 50.0)])
        upsert_perf_results(db, "bf16", [(256, 1500.0)], [(0, 60.0)])

        bpws = available_bpws(db)
        assert bpws == ["bf16", "2", "4", "6"]

    def test_empty_db_returns_empty(self, tmp_path):
        assert available_bpws(str(tmp_path / "nope.db")) == []


class TestDefaultPerfDbPath:
    def test_path_format(self):
        path = default_perf_db_path("/some/dir/MyModel")
        assert path.endswith("MyModelPerfData.db")
        assert "/some/dir/MyModel/" in path


# ---------------------------------------------------------------------------
# extract_perf_detail
# ---------------------------------------------------------------------------

class TestExtractPerfDetail:
    def test_full_output(self):
        output = (
            "Warmup  100%\n"
            "Prefill:\n"
            "Length     256:       1715.46 tokens/s\n"
            "Length     512:       2410.60 tokens/s\n"
            "Length    1024:       2839.02 tokens/s\n"
            "Length    2048:       3036.13 tokens/s\n"
            "Length    4096:       3047.00 tokens/s\n"
            "Length    8192:       2915.87 tokens/s\n"
            "Length   16384:       2679.28 tokens/s\n"
            "Length   32768:       2312.52 tokens/s\n"
            "\n"
            "Generation\n"
            "Context      0:         49.43 tokens/s\n"
            "Context    256:         45.97 tokens/s\n"
            "Context    512:         45.25 tokens/s\n"
            "Context   1024:         44.02 tokens/s\n"
            "Context   2048:         44.07 tokens/s\n"
            "Context   4096:         43.25 tokens/s\n"
            "Context   8192:         41.51 tokens/s\n"
            "Context  16384:         39.23 tokens/s\n"
            "Context  32512:         35.98 tokens/s\n"
        )
        detail = extract_perf_detail(output)
        assert len(detail["prefill"]) == 8
        assert len(detail["generation"]) == 9
        assert detail["prefill"][0] == (256, 1715.46)
        assert detail["prefill"][-1] == (32768, 2312.52)
        assert detail["generation"][0] == (0, 49.43)
        assert detail["generation"][-1] == (32512, 35.98)

    def test_with_ansi_codes(self):
        output = (
            "\x1b[33;1mPrefill:\x1b[0m\n"
            "Length     256: \x1b[32;1m  1715.46\x1b[0m tokens/s\n"
            "\x1b[33;1mGeneration\x1b[0m\n"
            "Context      0: \x1b[32;1m    49.43\x1b[0m tokens/s\n"
        )
        detail = extract_perf_detail(output)
        assert len(detail["prefill"]) == 1
        assert detail["prefill"][0] == (256, 1715.46)
        assert len(detail["generation"]) == 1
        assert detail["generation"][0] == (0, 49.43)

    def test_empty_output(self):
        detail = extract_perf_detail("")
        assert detail["prefill"] == []
        assert detail["generation"] == []

    def test_warmup_lines_excluded(self):
        # Warmup lines don't match the Length/Context patterns
        output = (
            "Warmup  50%\n"
            "Warmup  100%\n"
            "Prefill:\n"
            "Length     256:       1000.00 tokens/s\n"
        )
        detail = extract_perf_detail(output)
        assert len(detail["prefill"]) == 1
        assert detail["prefill"][0] == (256, 1000.0)
        assert detail["generation"] == []


# ---------------------------------------------------------------------------
# Integration: extract → upsert → read round-trip
# ---------------------------------------------------------------------------

class TestPerfRoundTrip:
    def test_extract_upsert_read(self, tmp_path):
        output = (
            "Prefill:\n"
            "Length     256:       1715.46 tokens/s\n"
            "Length    4096:       3047.00 tokens/s\n"
            "Generation\n"
            "Context      0:         49.43 tokens/s\n"
            "Context   4096:         43.25 tokens/s\n"
        )
        detail = extract_perf_detail(output)
        db = str(tmp_path / "perf.db")
        upsert_perf_results(db, "4", detail["prefill"], detail["generation"])

        data = read_perf_data(db, "4")
        assert len(data["4"]["prefill"]) == 2
        assert len(data["4"]["generation"]) == 2
        assert data["4"]["prefill"][0]["tokens_per_second"] == 1715.46
        assert data["4"]["generation"][1]["tokens_per_second"] == 43.25

        bpws = available_bpws(db)
        assert bpws == ["4"]


# ---------------------------------------------------------------------------
# Perf chart generation
# ---------------------------------------------------------------------------

class TestMakePerfPlot:
    @pytest.fixture(autouse=True)
    def _skip_no_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_generates_svg(self, tmp_path):
        from ezexl3.graph_svg import make_perf_plot

        out = str(tmp_path / "perf.svg")
        make_perf_plot(
            [256, 512, 1024], [1715.0, 2410.0, 2839.0],
            [0, 256, 512], [49.0, 46.0, 45.0],
            title="Test Model — 4 BPW", outfile=out,
        )
        assert os.path.exists(out)
        content = open(out).read()
        assert "<svg" in content
        assert "Test Model" in content

    def test_prefill_only(self, tmp_path):
        from ezexl3.graph_svg import make_perf_plot

        out = str(tmp_path / "perf.svg")
        make_perf_plot(
            [256, 512], [1715.0, 2410.0],
            [], [],
            title="Prefill Only", outfile=out,
        )
        assert os.path.exists(out)

    def test_generation_only(self, tmp_path):
        from ezexl3.graph_svg import make_perf_plot

        out = str(tmp_path / "perf.svg")
        make_perf_plot(
            [], [],
            [0, 256], [49.0, 46.0],
            title="Gen Only", outfile=out,
        )
        assert os.path.exists(out)


class TestGeneratePerfSvg:
    @pytest.fixture(autouse=True)
    def _skip_no_matplotlib(self):
        pytest.importorskip("matplotlib")

    def test_round_trip_db_to_svg(self, tmp_path):
        from ezexl3.graph_svg import generate_perf_svg

        db = str(tmp_path / "perf.db")
        upsert_perf_results(db, "4",
                            [(256, 1715.0), (512, 2410.0)],
                            [(0, 49.0), (256, 46.0)])

        out = str(tmp_path / "chart.svg")
        generate_perf_svg(db, "4", out, "Test Model — 4 BPW")
        assert os.path.exists(out)
        content = open(out).read()
        assert "<svg" in content

    def test_missing_bpw_raises(self, tmp_path):
        from ezexl3.graph_svg import generate_perf_svg

        db = str(tmp_path / "perf.db")
        upsert_perf_results(db, "4", [(256, 1715.0)], [(0, 49.0)])

        with pytest.raises(ValueError, match="No perf data"):
            generate_perf_svg(db, "8", str(tmp_path / "x.svg"), "title")
