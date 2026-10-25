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
            "experiments.exp4_neural_svd.cli",
            "experiments.exp5_L_M_neural_svd.cli",
            "experiments.exp6_stream_gpt2_mlp.cli",
            "experiments.exp7_stream_transformer.cli",
            "experiments.exp8_stream_token_mlp.cli",
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


if __name__ == "__main__":
    unittest.main()
