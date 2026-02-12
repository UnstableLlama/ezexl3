import unittest
from pathlib import Path


TEMPLATES = [
    "ezexl3/templates/basicTemplateREADME.md",
    "ezexl3/templates/fireTemplateREADME.md",
    "ezexl3/templates/greenTemplateREADME.md",
    "ezexl3/templates/forestTemplateREADME.md",
]


class TemplateGeometryContractTests(unittest.TestCase):
    def test_repo_data_includes_graph_above_table(self):
        graph_line = '<img class="repo-graph" src="{{GRAPH_FILE}}" alt="Quantization graph">'
        for template_path in TEMPLATES:
            content = Path(template_path).read_text()
            graph_idx = content.find(graph_line)
            table_idx = content.find('<div class="table-wrapper">')
            self.assertNotEqual(graph_idx, -1, template_path)
            self.assertNotEqual(table_idx, -1, template_path)
            self.assertLess(graph_idx, table_idx, template_path)

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
        expected_headers = ["REVISION", "GiB", "KL DIV", "PPL"]
        for template_path in TEMPLATES:
            content = Path(template_path).read_text()
            positions = [content.find(f"<th>{header}</th>") for header in expected_headers]
            self.assertTrue(all(pos != -1 for pos in positions), template_path)
            self.assertEqual(positions, sorted(positions), template_path)


if __name__ == "__main__":
    unittest.main()
