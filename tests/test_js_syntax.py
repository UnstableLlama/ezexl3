"""Ensure all shipped JavaScript files parse without syntax errors.

Uses Node.js ``--check`` (``-c``) flag which parses but does not execute.
This catches stray braces, missing commas, unterminated strings, and
other issues that would silently break the entire dashboard UI.
"""

import shutil
import subprocess
import unittest
from pathlib import Path

JS_DIRS = [
    Path("ezexl3/ui/static/js"),
    Path("ezexl3/chat/static/js"),
]


@unittest.skipUnless(shutil.which("node"), "Node.js not available")
class JsSyntaxTests(unittest.TestCase):

    def test_all_js_files_parse(self):
        errors = []
        for js_dir in JS_DIRS:
            if not js_dir.is_dir():
                continue
            for js_file in sorted(js_dir.glob("*.js")):
                result = subprocess.run(
                    ["node", "-c", str(js_file)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    errors.append(f"{js_file}: {result.stderr.strip()}")

        if errors:
            self.fail(
                f"{len(errors)} JS file(s) have syntax errors:\n"
                + "\n".join(errors)
            )


if __name__ == "__main__":
    unittest.main()
