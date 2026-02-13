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
            self.assertIn(
                'href="https://huggingface.co/UnstableLlama/SERA-8b-exl3/tree/4.00bpw"',
                readme,
            )
            self.assertIn(
                'href="https://huggingface.co/UnstableLlama/SERA-8b-exl3/tree/bf16"',
                readme,
            )


if __name__ == "__main__":
    unittest.main()
