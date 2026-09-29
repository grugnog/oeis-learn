"""Offline recipe checks; not a substitute for building/running both images."""
from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DevImages(unittest.TestCase):
    def test_manifest_inputs_exist(self):
        config = json.loads((ROOT / "gpu-project.json").read_text())
        self.assertEqual(config["target"], "gpu")
        for name in config["inputs"]:
            self.assertTrue((ROOT / name).exists(), name)
        self.assertIn("crates/oeis_wasm_evaluator", config["inputs"])

    def test_default_is_smolvm(self):
        recipe = (ROOT / "Dockerfile").read_text()
        stages = [line for line in recipe.splitlines() if line.startswith("FROM ")]
        self.assertTrue(stages[-1].endswith("AS smolvm"))
        self.assertIn("AS gpu", stages[0])
        self.assertEqual(recipe.count("ENV PYTHONPATH=/workspace/src"), 2)

    def test_native_and_python_policy(self):
        setup = (ROOT / "docker/dev/install.sh").read_text()
        native = (ROOT / "docker/dev/build-native.sh").read_text()
        self.assertIn("uv python install 3.12.12", setup)
        self.assertIn("--prune torch", setup)
        self.assertIn("--require-hashes --no-deps", setup)
        self.assertIn("cmp /tmp/oeis-torch-before /tmp/oeis-torch-after", setup)
        self.assertIn("maturin build --release --locked", native)


if __name__ == "__main__":
    unittest.main()
