"""Text files are read and written as UTF-8 on every platform.

Without an explicit encoding, Python uses the platform's default: UTF-8 on
Linux and macOS, but usually cp1252 on Windows. There, writing a suite report
that contains a character such as "↑" fails, and a config file is read
differently than on Linux. The tests run on Linux, where the default happens
to be right, so this test checks the source code itself: every place that
opens a file in text mode must say ``encoding="utf-8"``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import scene_recon

SOURCE_DIR = Path(scene_recon.__file__).parent


def _lines_opening_text_without_encoding(tree: ast.AST) -> list[int]:
    """Lines of ``open(...)``, ``.read_text(...)`` and ``.write_text(...)`` calls without ``encoding=``."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        function = node.func
        if isinstance(function, ast.Name) and function.id == "open":
            mode = node.args[1] if len(node.args) > 1 else None
            mode = next((keyword.value for keyword in node.keywords if keyword.arg == "mode"), mode)
            binary = isinstance(mode, ast.Constant) and "b" in str(mode.value)
            if not binary and "encoding" not in keywords:
                lines.append(node.lineno)
        elif isinstance(function, ast.Attribute) and function.attr in ("read_text", "write_text"):
            if "encoding" not in keywords:
                lines.append(node.lineno)
    return lines


def test_every_text_file_in_the_source_is_opened_as_utf8():
    problems = []
    for path in sorted(SOURCE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        problems += [f"{path.relative_to(SOURCE_DIR)}:{line}" for line in _lines_opening_text_without_encoding(tree)]
    assert not problems, 'Pass encoding="utf-8" when opening these text files: ' + ", ".join(problems)


def test_the_check_finds_a_text_file_opened_without_encoding():
    code = """
open(path)
open(path, "w")
path.read_text()
path.write_text(text)
open(path, "rb")
open(path, encoding="utf-8")
path.read_text(encoding="utf-8")
"""
    assert _lines_opening_text_without_encoding(ast.parse(code)) == [2, 3, 4, 5]
