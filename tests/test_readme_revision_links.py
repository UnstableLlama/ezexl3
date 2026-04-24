import tempfile
import unittest
from pathlib import Path

from ezexl3.readme import run_readme


class ReadmeRevisionLinkTests(unittest.TestCase):
    def test_revision_links_target_hf_tree_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "UnstableLlama-SERA-8b-exl3"
            model_dir.mkdir()

            measured_csv = model_dir / "UnstableLlama-SERA-8b-exl3Measured.csv"
            measured_csv.write_text(
                "weights,KL Div,PPL r-100,GiB\n"
                "4,0.1234,7.89,5.43\n"
                "bf16,0,6.54,10.0\n"
            )

            run_readme(str(model_dir), template_name="basic", interactive=False)

            readme = (model_dir / "README.md").read_text()
            self.assertIn('tree/4.00bpw"', readme)
            self.assertIn('href="https://huggingface.co/UnstableLlama/SERA-8b-exl3"', readme)


class ReadmePerBpwMirrorTests(unittest.TestCase):
    """run_readme should overwrite README.md in every BPW subdir by default."""

    def _setup_model_dir(self, tmp: str) -> Path:
        model_dir = Path(tmp) / "Foo-Model-exl3"
        model_dir.mkdir()
        (model_dir / "Foo-Model-exl3Measured.csv").write_text(
            "weights,KL Div,PPL r-100,GiB\n"
            "2,0.5,12.3,2.0\n"
            "4,0.1,7.8,4.0\n"
            "bf16,0,6.5,10.0\n"
        )
        # Two BPW subdirs that look like quantize output
        (model_dir / "2").mkdir()
        (model_dir / "4").mkdir()
        # An unrelated dir that should be ignored
        (model_dir / "w-2").mkdir()
        return model_dir

    def test_mirrors_into_each_bpw_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._setup_model_dir(tmp)

            run_readme(str(model_dir), template_name="basic", interactive=False)

            root = (model_dir / "README.md").read_text()
            self.assertTrue(root)
            for bpw in ("2", "4"):
                copy_path = model_dir / bpw / "README.md"
                self.assertTrue(copy_path.exists(),
                                f"missing per-BPW README in {bpw}/")
                self.assertEqual(copy_path.read_text(), root)
            # Working dirs (w-*) must NOT receive a README.
            self.assertFalse((model_dir / "w-2" / "README.md").exists())

    def test_overwrites_existing_per_bpw_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._setup_model_dir(tmp)
            stale = model_dir / "4" / "README.md"
            stale.write_text("STALE — should be overwritten")

            run_readme(str(model_dir), template_name="basic", interactive=False)

            self.assertNotIn("STALE", stale.read_text())

    def test_write_per_bpw_false_skips_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._setup_model_dir(tmp)

            run_readme(str(model_dir), template_name="basic",
                       interactive=False, write_per_bpw=False)

            self.assertTrue((model_dir / "README.md").exists())
            self.assertFalse((model_dir / "2" / "README.md").exists())
            self.assertFalse((model_dir / "4" / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
