"""Launcher contract and process tests, runnable without Isaac Sim or pytest."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import stage1


class Stage1Tests(unittest.TestCase):
    def test_commands_match_official_cli_and_registration(self):
        registered = {row["task_id"] for row in stage1.inventory()}
        for action in stage1.TASKS:
            extra = ["--checkpoint", "model.pt"] if action == "pickplace-eval" else []
            args = stage1.parser().parse_args([action, *extra])
            cmd = stage1.build_command(args, "contract_test")
            tree = ast.parse(Path(cmd[1]).read_text(encoding="utf-8"))
            flags = {
                arg.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            }
            flags.update({"--headless", "--logger", "--checkpoint", "--run_name"})
            self.assertTrue({value for value in cmd[2:] if value.startswith("--")} <= flags, cmd)
            self.assertIn(cmd[cmd.index("--task") + 1], registered)
        train = stage1.build_command(stage1.parser().parse_args(["pickplace-train"]), "contract_test")
        self.assertIn("agent.save_interval=1", train)
        self.assertIn("tensorboard", train)
        self.assertNotIn("--resume", train)

    def test_versions_match_repository_lock(self):
        lock = tomllib.loads((stage1.ROOT / "uv.lock").read_text(encoding="utf-8"))
        versions = {p["name"]: p["version"].split("+")[0] for p in lock["package"] if "version" in p}
        for name, version in stage1.EXPECTED.items():
            self.assertEqual(versions[name], version, name)

    def test_lfs_pointer_is_not_an_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.pt"
            self.assertFalse(stage1.asset_ok(path))
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n", encoding="utf-8")
            self.assertFalse(stage1.asset_ok(path))
            path.write_bytes(b"PK\x03\x04checkpoint")
            self.assertTrue(stage1.asset_ok(path))

    def test_training_evidence_parses_real_log_shape(self):
        log = "\x1b[1m Learning iteration 0/2 \x1b[0m\nComputation: 321 steps/s\nLearning iteration 1/2\n"
        result = stage1.parse_training_log(log)
        self.assertEqual(result["iterations_logged"], [0, 1])
        self.assertEqual(result["throughput_env_steps_per_s"], [321.0])
        self.assertEqual(stage1.parse_training_log("ImportError")["iterations_logged"], [])

    def test_gpu_samples_preserve_device_and_skip_unavailable_values(self):
        output = io.StringIO()
        with patch.object(stage1, "capture", return_value={"stdout": "0, 1234, 90\n1, N/A, N/A\n"}):
            samples = stage1.sample_gpu(stage1.csv.writer(output), 2.5)
        self.assertEqual(samples, [{"gpu_index": 0, "memory_mib": 1234.0}])
        self.assertIn("2.5,0,1234.0,90.0", output.getvalue())

    def test_process_exit_and_timeout_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage1, "sample_gpu", return_value=[]):
            folder = Path(temporary)
            passed = stage1.run_logged([sys.executable, "-c", "print('completed')"], folder, 10)
            self.assertEqual(passed["status"], "passed")
            self.assertIn("completed", (folder / "stdout.log").read_text())
            failed = stage1.run_logged([sys.executable, "-c", "raise SystemExit(7)"], folder, 10)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["returncode"], 7)
            timeout = stage1.run_logged([sys.executable, "-c", "import time; time.sleep(60)"], folder, 0.01)
            self.assertEqual(timeout["status"], "timeout")
            self.assertIsNotNone(timeout["returncode"])

    def test_training_artifacts_are_scoped_to_this_run(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(stage1, "ROOT", Path(temporary)):
            folder = stage1.ROOT / "logs/rsl_rl" / stage1.EXPERIMENT / "timestamp_run123"
            folder.mkdir(parents=True)
            self.assertIsNone(stage1.training_artifacts("run123")["checkpoint"])
            for name in ("model_0.pt", "model_2.pt", "model_10.pt"):
                (folder / name).write_bytes(b"checkpoint")
            params = folder / "params"
            params.mkdir()
            for name in ("env.yaml", "agent.yaml"):
                (params / name).write_text("resolved: true\n")
            result = stage1.training_artifacts("run123")
            self.assertTrue(result["configs_complete"])
            self.assertTrue(result["checkpoint"].endswith("model_10.pt"))
            self.assertIsNone(stage1.training_artifacts("another_run")["checkpoint"])

    def test_dry_run_never_executes_preflight_or_simulator(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / "uv.lock").write_bytes(b"lock")
            with (
                patch.object(stage1, "ROOT", root),
                patch.object(stage1, "capture", return_value={}),
                patch.object(stage1, "preflight", side_effect=AssertionError("preflight called")),
                patch.object(stage1, "run_logged", side_effect=AssertionError("simulator called")),
            ):
                self.assertEqual(stage1.main(["walk", "--dry-run"]), 0)
            manifest = json.loads(next(root.rglob("manifest.json")).read_text())
            self.assertEqual(manifest["status"], "dry_run")
            self.assertNotIn("checkpoint", manifest)

    def test_failed_preflight_never_launches_simulator(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / "uv.lock").write_bytes(b"lock")
            with (
                patch.object(stage1, "ROOT", root),
                patch.object(stage1, "capture", return_value={}),
                patch.object(stage1, "preflight", return_value={"ok": False, "checks": []}),
                patch.object(stage1, "run_logged", side_effect=AssertionError("simulator called")),
            ):
                self.assertEqual(stage1.main(["walk"]), 2)
            self.assertEqual(json.loads(next(root.rglob("manifest.json")).read_text())["status"], "preflight_failed")

    def test_missing_training_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / "uv.lock").write_bytes(b"lock")

            def fake_run(command, directory, timeout):
                (directory / "stdout.log").write_text("no training updates")
                return {"status": "passed", "iterations_logged": []}

            with (
                patch.object(stage1, "ROOT", root),
                patch.object(stage1, "capture", return_value={}),
                patch.object(stage1, "preflight", return_value={"ok": True, "checks": []}),
                patch.object(stage1, "run_logged", side_effect=fake_run),
            ):
                self.assertEqual(stage1.main(["pickplace-train"]), 2)
            self.assertEqual(json.loads(next(root.rglob("manifest.json")).read_text())["status"], "evidence_missing")

    def test_invalid_arguments_fail_before_execution(self):
        for argv in (
            ["pickplace-eval"],
            ["walk", "--num-envs", "0"],
            ["walk", "--steps", "-1"],
            ["pickplace-scene", "--checkpoint", "model.pt"],
            ["inventory", "--dry-run"],
            ["walk", "--timeout", "nan"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stage1.main(argv)


if __name__ == "__main__":
    unittest.main()
