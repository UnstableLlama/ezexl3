import unittest
from pathlib import Path


TEMPLATES = [
    "ezexl3/templates/basicTemplateREADME.md",
    "ezexl3/templates/fireTemplateREADME.md",
    "ezexl3/templates/greenTemplateREADME.md",
    "ezexl3/templates/punkTemplateREADME.md",
]


class TemplateGeometryContractTests(unittest.TestCase):
    def test_repo_data_includes_charts_above_table(self):
        for template_path in TEMPLATES:
            content = Path(template_path).read_text()
            charts_idx = content.find("{{QBENCH_CHARTS}}")
            table_idx = content.find('<div class="table-wrapper">')
            self.assertNotEqual(charts_idx, -1, template_path)
            self.assertNotEqual(table_idx, -1, template_path)
            self.assertLess(charts_idx, table_idx, template_path)

    def test_templates_no_longer_reference_the_retired_kl_ppl_svg(self):
        for template_path in TEMPLATES:
            content = Path(template_path).read_text()
            self.assertNotIn("{{GRAPH_FILE}}", content, template_path)

    def test_non_basic_templates_use_shared_repo_data_geometry_css(self):
        required_css = [
            ".repo-data-panel",
            ".repo-data-body",
            "--edge-gap: 8px;",
            ".repo-graph",
            "width: min(1440px, calc(100% - (var(--edge-gap) * 2)));",
            "max-width: calc(100% - (var(--edge-gap) * 2));",
        ]
        for template_path in TEMPLATES[1:]:
            content = Path(template_path).read_text()
            for token in required_css:
                self.assertIn(token, content, f"{template_path} missing {token}")

    def test_table_headers_keep_expected_order(self):
        # KL divergence and perplexity moved to the qbench charts, so the
        # table is down to the per-BPW repo link and its size. Standard
        # templates use uppercase headers; punk uses bracketed style.
        header_variants = {
            "ezexl3/templates/punkTemplateREADME.md": ["[Revision]", "[GiB]"],
        }
        default_headers = ["REVISION", "GiB"]
        for template_path in TEMPLATES:
            content = Path(template_path).read_text()
            headers = header_variants.get(template_path, default_headers)
            positions = [content.find(f"<th>{header}</th>") for header in headers]
            self.assertTrue(all(pos != -1 for pos in positions), template_path)
            self.assertEqual(positions, sorted(positions), template_path)


if __name__ == "__main__":
    unittest.main()
