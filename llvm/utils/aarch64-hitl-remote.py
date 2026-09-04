#!/usr/bin/env python3
"""Cache AArch64 artifacts over SSH and run a benchmark on a remote board.

The remote benchmark is pinned to one CPU and must print the standard HITL
JSON result.  Network and process-launch time are deliberately outside the
benchmark's own measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ssh(host: str, options: list[str], command: list[str], **kwargs):
    remote = shlex.join(command)
    check = kwargs.pop("check", True)
    return subprocess.run(
        ["ssh", *options, host, remote],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
        **kwargs,
    )


def cache_artifact(
    host: str, options: list[str], remote_dir: PurePosixPath, local: Path
) -> tuple[PurePosixPath, str]:
    digest = sha256(local)
    suffix = local.suffix if local.suffix else ".artifact"
    remote = remote_dir / f"{digest}{suffix}"
    probe = ssh(host, options, ["test", "-f", str(remote)], check=False)
    if probe.returncode == 0:
        return remote, digest

    temporary = remote_dir / f".{digest}.tmp"
    upload_command = (
        f"umask 077; mkdir -p {shlex.quote(str(remote_dir))}; "
        f"cat > {shlex.quote(str(temporary))} && "
        f"test $(sha256sum {shlex.quote(str(temporary))} | cut -d' ' -f1) = "
        f"{shlex.quote(digest)} && mv {shlex.quote(str(temporary))} "
        f"{shlex.quote(str(remote))}"
    )
    with local.open("rb") as stream:
        subprocess.run(
            ["ssh", *options, host, upload_command],
            stdin=stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    return remote, digest


def telemetry(host: str, options: list[str]) -> dict[str, Any]:
    script = (
        "printf 'frequency_khz='; "
        "cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null || echo 0; "
        "printf 'temperature_millic='; "
        "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0; "
        "printf 'throttled='; vcgencmd get_throttled 2>/dev/null | sed 's/.*=//' || echo 0"
    )
    result = ssh(host, options, ["sh", "-c", script])
    values: dict[str, Any] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        try:
            values[key] = int(value, 0)
        except ValueError:
            values[key] = value
    throttled = values.get("throttled", 0)
    values["currently_throttled"] = bool(
        isinstance(throttled, int) and throttled & 0xFFFF
    )
    return values


def benchmark_record(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            break
    else:
        raise RuntimeError("remote benchmark did not print a JSON object")
    for key in ("baseline_runtime_us", "candidate_runtime_us", "correct"):
        if key not in value:
            raise RuntimeError(f"remote benchmark JSON is missing {key!r}")
    for key in ("baseline_runtime_us", "candidate_runtime_us"):
        number = float(value[key])
        if not math.isfinite(number) or number <= 0:
            raise RuntimeError(f"remote benchmark {key} must be positive and finite")
        value[key] = number
    if not isinstance(value["correct"], bool):
        raise RuntimeError("remote benchmark correct must be a JSON boolean")
    return value


def samples_are_stable(record: dict[str, Any], max_ratio: float) -> bool:
    for key in ("baseline_samples_us", "candidate_samples_us"):
        samples = record.get(key)
        if samples is None:
            continue
        if not isinstance(samples, list) or not samples:
            raise RuntimeError(f"remote benchmark {key} must be a non-empty array")
        values = [float(sample) for sample in samples]
        if any(not math.isfinite(sample) or sample <= 0 for sample in values):
            raise RuntimeError(
                f"remote benchmark {key} values must be positive and finite"
            )
        if max(values) / min(values) > max_ratio:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--remote-dir", default="/tmp/llvm-aarch64-hitl")
    parser.add_argument("--cpu", type=int, default=3)
    parser.add_argument(
        "--max-sample-ratio", type=float, default=1.10,
        help="mark measurements unstable when max/min sample time exceeds this",
    )
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--remote-command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if args.cpu < 0:
        raise SystemExit("--cpu must be non-negative")
    if not math.isfinite(args.max_sample_ratio) or args.max_sample_ratio < 1:
        raise SystemExit("--max-sample-ratio must be finite and at least 1")
    if not args.remote_command:
        raise SystemExit("--remote-command requires a command")
    remote_dir = PurePosixPath(args.remote_dir)
    if not remote_dir.is_absolute() or ".." in remote_dir.parts:
        raise SystemExit("--remote-dir must be an absolute normalized path")
    options = [item for value in args.ssh_option for item in ("-o", value)]
    options.extend(["-o", "BatchMode=yes"])

    try:
        baseline, baseline_hash = cache_artifact(
            args.host, options, remote_dir, args.baseline.resolve()
        )
        candidate, candidate_hash = cache_artifact(
            args.host, options, remote_dir, args.candidate.resolve()
        )
        before = telemetry(args.host, options)
        replacements = {
            "remote_baseline": str(baseline),
            "remote_candidate": str(candidate),
            "baseline_sha256": baseline_hash,
            "candidate_sha256": candidate_hash,
            "remote_dir": str(remote_dir),
        }
        command = []
        for argument in args.remote_command:
            for key, replacement in replacements.items():
                argument = argument.replace("{" + key + "}", replacement)
            command.append(argument)
        result = ssh(
            args.host, options, ["taskset", "-c", str(args.cpu), *command]
        )
        after = telemetry(args.host, options)
        record = benchmark_record(result.stdout)
        record.update(
            remote_host=args.host,
            remote_cpu=args.cpu,
            baseline_sha256=baseline_hash,
            candidate_sha256=candidate_hash,
            telemetry_before=before,
            telemetry_after=after,
            stable=not before["currently_throttled"]
            and not after["currently_throttled"]
            and samples_are_stable(record, args.max_sample_ratio),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
