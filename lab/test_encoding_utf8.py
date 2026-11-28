import ast
import pathlib
import unittest


class ExplicitUtf8EncodingTest(unittest.TestCase):
    def test_all_lab_text_io_specifies_encoding(self):
        lab_dir = pathlib.Path(__file__).resolve().parent
        missing = []
        for path in sorted(lab_dir.rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute) and node.func.attr in ("read_text", "write_text"):
                    if not any(k.arg == "encoding" for k in node.keywords):
                        rel = path.relative_to(lab_dir.parent)
                        missing.append(f"{rel}:{node.lineno}:{node.func.attr}")
                elif isinstance(node.func, ast.Name) and node.func.id == "open":
                    mode = next((k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), None)
                    if mode is None and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        mode = node.args[1].value
                    if isinstance(mode, str) and "b" in mode:
                        continue
                    if not any(k.arg == "encoding" for k in node.keywords):
                        rel = path.relative_to(lab_dir.parent)
                        missing.append(f"{rel}:{node.lineno}:open")
        self.assertEqual(missing, [], f"Missing explicit encoding= in: {missing}")


if __name__ == "__main__":
    unittest.main()
