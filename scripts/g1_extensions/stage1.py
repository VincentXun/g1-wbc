#!/usr/bin/env python3
"""Reproduce official G1 tasks without importing Isaac Sim in the launcher.

Run with the repository's Linux uv environment; inventory and --dry-run also
work with a standalone Python on the development machine.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = "agile/data/policy/"
WALK = POLICY + "velocity_g1/unitree_g1_velocity_history_state_dict.pt"
HEIGHT = POLICY + "velocity_height_g1/unitree_g1_velocity_height_history_state_dict.pt"
LOWER = POLICY + "velocity_height_g1/unitree_g1_velocity_height_teacher_torchscript.pt"
MOTION = "agile/rl_env/assets/motion_data/object_pick_and_place_retarget_motion_g1_3finger_hands.yaml"
EXPERIMENT = "g1_extensions_stage1_pick_place"
TASKS = {
    "walk": ("Velocity-G1-History-v0", "eval.py", WALK, 2),
    "height": ("Velocity-Height-G1-History-v0", "eval.py", HEIGHT, 2),
    "pickplace-scene": ("PickPlace-G1-v0", "play.py", None, 2),
    "pickplace-train": ("PickPlace-G1-v0", "train.py", None, 32),
    "pickplace-eval": ("PickPlace-G1-v0", "eval.py", None, 2),
}
EXPECTED = {"isaaclab": "3.0.0b2", "isaacsim": "6.0.0.1", "torch": "2.11.0", "rsl-rl-lib": "5.4.1"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def capture(command: list[str], timeout: float = 15) -> dict:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


def inventory() -> list[dict]:
    """Read literal Gym registrations, without importing task modules or the simulator."""
    rows = []
    for path in sorted((ROOT / "agile/rl_env/tasks").rglob("__init__.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
                for keyword in node.keywords:
                    if keyword.arg == "id" and isinstance(keyword.value, ast.Constant):
                        rows.append(
                            {
                                "task_id": keyword.value.value,
                                "registration": path.relative_to(ROOT).as_posix(),
                                "runtime_verified": False,
                            }
                        )
    return sorted(rows, key=lambda row: row["task_id"])


def asset_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        return not stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")


def preflight(required: list[str]) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: object) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    check(
        "linux_x86_64",
        platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"},
        platform.platform(),
    )
    check("python_3_12", sys.version_info[:2] == (3, 12), sys.version)
    for package, expected in EXPECTED.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        # Torch wheels carry a +cu128 suffix.
        check(package, version.split("+")[0] == expected, {"installed": version, "expected": expected})
    for relative in required:
        check(relative, asset_ok(ROOT / relative), "file must exist and must not be a Git LFS pointer")
    gpu = capture(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    check("nvidia_smi", gpu["returncode"] == 0 and bool(gpu["stdout"]), gpu)
    if platform.system() == "Linux" and sys.version_info[:2] == (3, 12):
        cuda = capture(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.__version__, torch.version.cuda); "
                "assert torch.version.cuda == '12.8', 'Expected CUDA 12.8 torch build'; "
                "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                "x=torch.ones(1,device='cuda'); print((x+x).item(),torch.cuda.get_device_name(0))",
            ],
            timeout=60,
        )
        check("cuda_tensor_operation", cuda["returncode"] == 0, cuda)
        patch = capture([sys.executable, "scripts/verify_rsl_rl.py"], timeout=60)
        check("rsl_rl_patch", patch["returncode"] == 0, patch)
    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "note": "USD/network availability is verified only by launching a task, not by this preflight.",
    }


def build_command(args: argparse.Namespace, run_id: str) -> list[str]:
    task, script, checkpoint, default_envs = TASKS[args.action]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "--task",
        task,
        "--num_envs",
        str(args.num_envs or default_envs),
    ]
    if not args.viewer:
        cmd.append("--headless")
    if script == "train.py":
        cmd += [
            "--max_iterations",
            str(args.iterations),
            "--seed",
            str(args.seed),
            "--logger",
            "tensorboard",
            "--run_name",
            run_id,
            f"agent.experiment_name={EXPERIMENT}",
            "agent.save_interval=1",
        ]
    else:
        cmd += ["--num_steps", str(args.steps)]
        if script == "eval.py":
            path = Path(args.checkpoint).resolve() if args.checkpoint else ROOT / checkpoint
            cmd += ["--checkpoint", str(path)]
            if args.action in {"walk", "height"}:
                fields = ["lin_vel_x", "lin_vel_y", "ang_vel_z"]
                if args.action == "height":
                    fields.append("base_height")
                cmd += ["--random_seed", str(args.seed), "--random_commands", *fields]
            if args.video:
                cmd += ["--video", "--video_length", str(min(args.steps, 500))]
    return cmd


def parse_training_log(log: str) -> dict:
    clean = ANSI.sub("", log)
    return {
        "iterations_logged": sorted({int(value) for value in re.findall(r"Learning iteration\s+(\d+)\s*/", clean)}),
        "throughput_env_steps_per_s": [float(value) for value in re.findall(r"([\d.]+)\s+steps/s", clean)],
    }


def sample_gpu(writer, elapsed: float) -> list[dict]:
    result = capture(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], 5
    )
    samples = []
    for row in csv.reader(result["stdout"].splitlines()):
        try:
            index, memory, utilization = [float(value.strip()) for value in row]
        except (ValueError, TypeError):
            continue
        writer.writerow([round(elapsed, 3), int(index), memory, utilization])
        samples.append({"gpu_index": int(index), "memory_mib": memory})
    return samples


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def run_logged(command: list[str], directory: Path, timeout: float) -> dict:
    start = time.monotonic()
    samples = []
    status = "failed"
    process = None
    with (
        (directory / "stdout.log").open("w", encoding="utf-8") as log,
        (directory / "gpu.csv").open("w", encoding="utf-8", newline="") as gpu_log,
    ):
        writer = csv.writer(gpu_log)
        writer.writerow(["elapsed_s", "gpu_index", "whole_gpu_memory_used_mib", "gpu_utilization_percent"])
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            # play.py does not insert the checkout root, unlike train.py/eval.py.
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix"
            )
            next_notice = 30.0
            while process.poll() is None:
                elapsed = time.monotonic() - start
                samples.extend(sample_gpu(writer, elapsed))
                gpu_log.flush()
                if elapsed >= timeout:
                    status = "timeout"
                    break
                if elapsed >= next_notice:
                    print(f"Running: {elapsed:.0f}s; log: {directory / 'stdout.log'}", flush=True)
                    next_notice += 30
                time.sleep(1)
            else:
                status = "passed" if process.returncode == 0 else "failed"
        except KeyboardInterrupt:
            status = "interrupted"
        except OSError as exc:
            log.write(str(exc))
        finally:
            if process is not None:
                stop_process(process)
    peaks = {}
    for row in samples:
        key = str(row["gpu_index"])
        peaks[key] = max(peaks.get(key, 0), row["memory_mib"])
    output = (directory / "stdout.log").read_text(encoding="utf-8", errors="replace")
    return {
        "status": status,
        "returncode": process.returncode if process else None,
        "wall_seconds_including_startup": round(time.monotonic() - start, 3),
        "sampled_whole_gpu_peak_mib": peaks,
        "gpu_sample_count": len(samples),
        **parse_training_log(output),
    }


def training_artifacts(run_id: str) -> dict:
    base = ROOT / "logs/rsl_rl" / EXPERIMENT
    directories = sorted(base.glob(f"*_{run_id}"))
    checkpoints = [path for folder in directories for path in folder.glob("model_*.pt") if asset_ok(path)]
    checkpoints.sort(key=lambda path: int(path.stem.split("_")[-1]))
    configs = [folder / "params" / name for folder in directories for name in ("env.yaml", "agent.yaml")]
    return {
        "checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "configs": [str(path) for path in configs if path.is_file()],
        "configs_complete": len(directories) == 1 and len(configs) == 2 and all(p.is_file() for p in configs),
    }


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=["inventory", "preflight", *TASKS])
    cli.add_argument("--dry-run", action="store_true", help="Print commands only; never count as runtime acceptance.")
    cli.add_argument("--num-envs", type=int)
    cli.add_argument("--steps", type=int, default=2000, help="Bounded rollout length (50 control steps/s).")
    cli.add_argument("--iterations", type=int, default=2, help="PickPlace PPO updates; smoke test, not convergence.")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--timeout", type=float, default=1800, help="Wall timeout in seconds, including startup.")
    cli.add_argument("--checkpoint", help="Required for pickplace-eval; explicit caller-owned checkpoint.")
    cli.add_argument("--viewer", action="store_true", help="Show Isaac Sim GUI on a Linux graphical session.")
    cli.add_argument("--video", action="store_true", help="Record eval video (requires rendering resources).")
    return cli


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if (
        not math.isfinite(args.timeout)
        or min(args.steps, args.iterations, args.timeout, args.num_envs if args.num_envs is not None else 1) <= 0
    ):
        cli.error("steps, iterations, timeout and num-envs must be positive")
    if args.dry_run and args.action not in TASKS:
        cli.error("--dry-run is only supported for task actions; inventory never launches a simulator")
    if args.action == "pickplace-eval" and not args.checkpoint:
        cli.error("pickplace-eval requires --checkpoint from a completed training run")
    if args.checkpoint and (args.action not in TASKS or TASKS[args.action][1] != "eval.py"):
        cli.error("--checkpoint is only supported for evaluation actions")
    if args.checkpoint:
        args.checkpoint = str(Path(args.checkpoint).resolve())
    if args.video and (args.action not in TASKS or TASKS[args.action][1] != "eval.py"):
        cli.error("--video is only supported for evaluation actions")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = ROOT / "logs/g1_extensions/stage1" / f"{run_id}_{args.action}"
    directory.mkdir(parents=True)
    manifest = {
        "action": args.action,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "git": capture(["git", "rev-parse", "HEAD"]),
        "working_tree": capture(["git", "status", "--short"]),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "python": sys.executable,
        "options": vars(args),
        "status": "prepared",
    }
    write_json(directory / "manifest.json", manifest)
    print(f"Records: {directory}", flush=True)
    if args.action == "inventory":
        rows = inventory()
        write_json(directory / "tasks.json", rows)
        print("\n".join(row["task_id"] for row in rows))
        manifest["status"] = "inventory_only"
    elif args.dry_run and args.action != "preflight":
        manifest.update(status="dry_run", command=build_command(args, run_id))
        print(shlex.join(manifest["command"]))
    else:
        required = [WALK, HEIGHT, LOWER, MOTION] if args.action == "preflight" else []
        if args.action in {"walk", "height"}:
            required = [args.checkpoint or TASKS[args.action][2]]
        elif args.action.startswith("pickplace"):
            required = [LOWER, MOTION] + ([args.checkpoint] if args.checkpoint else [])
        report = preflight(required)
        write_json(directory / "preflight.json", report)
        for check in report["checks"]:
            print(f"{'OK' if check['ok'] else 'FAIL'}: {check['name']}")
        if not report["ok"]:
            manifest["status"] = "preflight_failed"
        elif args.action == "preflight":
            manifest["status"] = "preflight_passed"
        else:
            command = build_command(args, run_id)
            manifest.update(command=command, status="running")
            write_json(directory / "manifest.json", manifest)
            print(shlex.join(command), flush=True)
            manifest.update(run_logged(command, directory, args.timeout))
            if args.action == "pickplace-train":
                artifacts = training_artifacts(run_id)
                manifest.update(artifacts)
                if manifest["status"] == "passed" and (
                    not artifacts["checkpoint"]
                    or not artifacts["configs_complete"]
                    or len(manifest["iterations_logged"]) < args.iterations
                ):
                    manifest.update(
                        status="evidence_missing", reason="Expected updates, checkpoint and resolved configs"
                    )
            print((directory / "stdout.log").read_text(encoding="utf-8", errors="replace")[-3000:])
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_json(directory / "manifest.json", manifest)
    print(f"Status: {manifest['status']}; manifest: {directory / 'manifest.json'}")
    if manifest.get("checkpoint"):
        print(
            "Next: uv run --frozen scripts/g1_extensions/stage1.py pickplace-eval --checkpoint "
            + shlex.quote(manifest["checkpoint"])
        )
    return 0 if manifest["status"] in {"passed", "preflight_passed", "inventory_only", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
