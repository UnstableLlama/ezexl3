import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ezexl3.readme import run_readme


class ReadmeRevisionLinkTests(unittest.TestCase):
    def test_revision_links_use_quantized_repo_owner_and_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "allenai-SERA-8B"
            model_dir.mkdir()

            measured_csv = model_dir / "allenai-SERA-8BMeasured.csv"
            measured_csv.write_text(
                "weights,KL Div,PPL r-100,GiB\n"
                "2,0.1200,7.10,5.43\n"
                "3,0.1234,7.89,6.54\n"
            )

            fake_meta = {
                "AUTHOR": "allenai",
                "MODEL": "SERA-8B",
                "REPOLINK": "https://huggingface.co/allenai/SERA-8B",
                "USER": "UnstableLlama",
                "QUANT_METHOD": "exl3",
                "QUANT_TOOL": "exllamav3",
            }

            with patch("ezexl3.readme.prompt_metadata", return_value=fake_meta):
                run_readme(str(model_dir), template_name="basic", interactive=False)

            readme = (model_dir / "README.md").read_text()
            self.assertIn(
                'href="https://huggingface.co/UnstableLlama/SERA-8B-exl3/tree/2.00bpw"',
                readme,
            )
            self.assertIn(
                'href="https://huggingface.co/UnstableLlama/SERA-8B-exl3/tree/3.00bpw"',
                readme,
            )
            self.assertNotIn(
                'href="https://huggingface.co/allenai/SERA-8B/tree/3.00bpw"',
                readme,
            )

    def test_does_not_duplicate_quant_method_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "UnstableLlama-SERA-8B-exl3"
            model_dir.mkdir()

            measured_csv = model_dir / "UnstableLlama-SERA-8B-exl3Measured.csv"
            measured_csv.write_text(
                "weights,KL Div,PPL r-100,GiB\n"
                "4,0.1234,7.89,5.43\n"
                "bf16,0,6.54,10.0\n"
            )

            fake_meta = {
                "AUTHOR": "UnstableLlama",
                "MODEL": "SERA-8B-exl3",
                "REPOLINK": "https://huggingface.co/UnstableLlama/SERA-8B-exl3",
                "USER": "UnstableLlama",
                "QUANT_METHOD": "exl3",
                "QUANT_TOOL": "exllamav3",
            }

            with patch("ezexl3.readme.prompt_metadata", return_value=fake_meta):
                run_readme(str(model_dir), template_name="basic", interactive=False)

            readme = (model_dir / "README.md").read_text()
            self.assertIn(
                'href="https://huggingface.co/UnstableLlama/SERA-8B-exl3/tree/4.00bpw"',
                readme,
            )
            self.assertIn(
                'href="https://huggingface.co/UnstableLlama/SERA-8B-exl3/tree/bf16"',
                readme,
            )
            self.assertNotIn("SERA-8B-exl3-exl3/tree", readme)


if __name__ == "__main__":
    unittest.main()
