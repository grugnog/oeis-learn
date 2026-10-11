"""Discovery only: fixture bytes are never presented as admitted/trained evidence."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.gpu_progress_settings import discover


class SettingsDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def discover(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return discover(self.root)

    def test_missing_inputs_remain_null_and_small_cohort_is_explicit(self):
        settings = self.discover()
        for name in ("source_manifest", "archive", "holdouts", "legacy_cases"):
            self.assertIsNone(settings[name])
        config = Path(settings["cohort_config"]).read_text()
        self.assertIn("profile: prefix20_total100_affine_v2", config)
        self.assertIn("dev_count: 2\nfinal_count: 2", config)
        self.assertEqual(settings["final_seconds"], 3600)

    def test_pairs_archive_with_exact_holdout_identity_and_handles_ambiguity(self):
        profile = dict(
            version="expanded-holdouts/v1",
            seed=12,
            structures=[],
            compositions=[],
            parameter_bits=128,
            prefixes=[],
        )
        payload = json.dumps(
            profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
        identity = "sha256:" + hashlib.sha256(payload).hexdigest()
        holdouts = self.write("inputs/holdouts.json", profile)
        self.write("archive/index.json", dict(version="program-archive/v1", holdouts=identity))
        source = dict(version="indexed-source/v1", release="unit-fixture", records=[{}])
        manifest = self.write("inputs/source.json", source)
        settings = self.discover()
        self.assertEqual(settings["holdouts"], str(holdouts))
        self.assertEqual(settings["archive"], str(self.root / "archive"))
        self.assertEqual(settings["source_manifest"], str(manifest))
        self.assertIn("seed: 12", Path(settings["cohort_config"]).read_text())
        self.write("inputs/source-other.json", source)
        self.assertIsNone(self.discover()["source_manifest"])
        self.write("archive/index.json", dict(version="program-archive/v1", holdouts="wrong"))
        self.assertIsNone(self.discover()["archive"])

    def test_latest_numeric_checkpoint_requires_matching_run_config(self):
        self.write("runs/historical/config.yaml", {"model": {}})
        for epoch in (0, 999, 1000):
            self.write(f"runs/historical/model_epoch_{epoch:03d}.v2.pt", {})
        self.write("runs/orphan/model_epoch_9999.v2.pt", {})
        settings = self.discover()
        cases = json.loads(Path(settings["legacy_cases"]).read_text())
        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0]["checkpoint"].endswith("model_epoch_1000.v2.pt"))

    def test_excludes_test_history_and_symlink_inputs_and_output(self):
        source = dict(version="indexed-source/v1", records=[{}])
        self.write("tests/source.json", source)
        self.write("tmp-gpu-progress/history/source.json", source)
        outside = self.root / "hidden-source"
        outside.write_text(json.dumps(source))
        (self.root / "linked-source.json").symlink_to(outside)
        self.assertIsNone(self.discover()["source_manifest"])
        output = self.root / "tmp-gpu-inputs/settings.auto.json"
        output.unlink()
        output.symlink_to(outside)
        before = outside.read_bytes()
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.discover()
        self.assertEqual(outside.read_bytes(), before)

    def test_invalid_json_and_malformed_profiles_do_not_create_candidates(self):
        self.write("inputs/incomplete-holdouts.json", dict(version="expanded-holdouts/v1"))
        self.write("archive/index.json", dict(version="program-archive/v1", holdouts=[]))
        (self.root / "inputs/not-json.json").write_text('{"version": NaN}')
        settings = self.discover()
        self.assertIsNone(settings["archive"])
        self.assertIsNone(settings["holdouts"])

    def test_preserves_manual_settings_and_refuses_symlink_output_directory(self):
        manual = self.write("tmp-gpu-inputs/settings.json", {"manual": True})
        cohort = self.write("tmp-gpu-inputs/cohort.yaml", {"manual": True})
        before = manual.read_bytes(), cohort.read_bytes()
        self.discover()
        self.assertEqual((manual.read_bytes(), cohort.read_bytes()), before)
        with tempfile.TemporaryDirectory() as other:
            root = Path(other)
            (root / "tmp-gpu-inputs").symlink_to(
                self.root / "tmp-gpu-inputs", target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                discover(root)


if __name__ == "__main__":
    unittest.main()
