import subprocess
import sys
import unittest


class CLIStartupTests(unittest.TestCase):
    def test_help_does_not_import_heavy_dependencies(self) -> None:
        for module in (
            "data_pipline.cli",
            "experiments.exp1_L_M_table.cli",
            "experiments.exp2_PFH.cli",
            "experiments.exp3_L_M_diff.cli",
        ):
            with self.subTest(module=module):
                code = f"""
import sys
from {module} import main

try:
    main(["-h"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
print(sorted(m for m in ("torch", "transformers", "matplotlib") if m in sys.modules))
"""
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(result.stdout.splitlines()[-1], "[]")

class SampleStoreCLIHelpTests(unittest.TestCase):
    def test_nested_help_is_available_without_loading_models(self):
        for args in (["sample-store", "-h"], ["sample-store", "produce", "-h"], ["sample-store", "status", "-h"]):
            code = """
import sys
from data_pipline.cli import main
try:
    main(ARGS)
except SystemExit as error:
    assert error.code == 0
assert not any(m in sys.modules for m in ('torch', 'transformers', 'matplotlib'))
""".replace("ARGS", repr(args))
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
