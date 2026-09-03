#!/usr/bin/env python3
"""Run serial hardware-in-the-loop SA over AMDGPU pre-RA schedules.

LLVM generates and verifies legal complete-schedule proposals.  A caller-owned
benchmark command measures each compiled candidate against the frozen founder
and prints one JSON object containing:

  {"baseline_runtime_us": 10.0, "candidate_runtime_us": 9.5,
   "correct": true}

The controller uses candidate_runtime_us / baseline_runtime_us as its energy,
performs Metropolis acceptance, and retains the complete compiler trajectory.
The benchmark command is passed without a shell and may use the placeholders
{baseline}, {candidate}, {function}, {region}, and {output_dir}.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(seed: int, evaluation: int, attempt: int) -> int:
    text = f"{seed}\0{evaluation}\0{attempt}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def parse_depths(text: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from error
    if not depths or any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError("depths must be positive")
    return depths


def metropolis_probability(parent: float, child: float, temperature: float) -> float:
    if parent <= 0 or child <= 0:
        raise ValueError("runtime ratios must be positive")
    if child <= parent:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-((child / parent) - 1.0) / temperature)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    wall_seconds: float


def run_command(command: list[str], log: Path, timeout: int) -> CommandResult:
    started = time.monotonic()
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=os.environ.copy(),
    )
    elapsed = time.monotonic() - started
    log.write_text(
        "$ " + shlex.join(command) + "\n"
        + f"exit: {result.returncode}\nwall_seconds: {elapsed:.6f}\n"
        + "--- stdout ---\n" + result.stdout
        + "\n--- stderr ---\n" + result.stderr
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log}"
        )
    return CommandResult(result.stdout, result.stderr, elapsed)


def trajectory_region_size(path: Path, region: int) -> int:
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if (
            record.get("kind") == "state"
            and record.get("role") == "founder"
            and record.get("region") == region
        ):
            order = record.get("order")
            if not isinstance(order, list):
                raise RuntimeError("founder trajectory has an invalid order")
            return len(order)
    raise RuntimeError(f"mutation region {region} was not emitted for the founder")


def benchmark_result(stdout: str) -> dict[str, Any]:
    record = None
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            record = value
            break
    if record is None:
        raise RuntimeError("benchmark did not print a JSON object")
    for key in ("baseline_runtime_us", "candidate_runtime_us", "correct"):
        if key not in record:
            raise RuntimeError(f"benchmark JSON is missing {key!r}")
    baseline = float(record["baseline_runtime_us"])
    candidate = float(record["candidate_runtime_us"])
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise RuntimeError("benchmark runtimes must be finite")
    if baseline <= 0 or candidate <= 0:
        raise RuntimeError("benchmark runtimes must be positive")
    if not isinstance(record["correct"], bool):
        raise RuntimeError("benchmark correct field must be a JSON boolean")
    record["baseline_runtime_us"] = baseline
    record["candidate_runtime_us"] = candidate
    return record


class Controller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output = args.output_dir.resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise RuntimeError(f"output directory is not empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = self.output / "search.jsonl"
        self.rng = random.Random(args.seed)
        self.started = time.monotonic()
        self.compiler_invocations = 0
        self.hardware_evaluations = 0
        self.compile_seconds = 0.0
        self.benchmark_seconds = 0.0

    def llc_command(
        self,
        directory: Path,
        schedule: Path,
        trajectory: Path | None,
        replay: Path | None = None,
        depth: int | None = None,
        mutation_seed: int | None = None,
    ) -> list[str]:
        command = [
            str(self.args.llc),
            str(self.args.input),
            "-o", str(directory / "kernel.o"),
            f"-mtriple={self.args.triple}",
            f"-mcpu={self.args.mcpu}",
            "-filetype=obj",
            "-verify-machineinstrs",
            f"-amdgpu-prera-training-function={self.args.function}",
            f"-amdgpu-prera-training-record-schedule={schedule}",
        ]
        command.extend(self.args.llc_arg)
        if trajectory is not None:
            command.append(f"-amdgpu-prera-training-record-trajectory={trajectory}")
        if replay is not None:
            command.append(f"-amdgpu-prera-training-replay-schedule={replay}")
        if depth is not None:
            command.extend(
                [
                    f"-amdgpu-prera-training-mutation-region={self.args.region}",
                    f"-amdgpu-prera-training-mutation-depth={depth}",
                    f"-amdgpu-prera-training-seed={mutation_seed}",
                ]
            )
        return command

    def invoke_llc(self, command: list[str], log: Path) -> None:
        result = run_command(command, log, self.args.timeout)
        self.compiler_invocations += 1
        self.compile_seconds += result.wall_seconds

    def link(self, directory: Path, stem: str = "kernel") -> Path:
        object_path = directory / f"{stem}.o"
        if self.args.artifact_kind == "object":
            return object_path
        artifact = directory / f"{stem}.hsaco"
        result = run_command(
            [str(self.args.ld_lld), "-shared", str(object_path), "-o", str(artifact)],
            directory / f"{stem}.link.log",
            self.args.timeout,
        )
        self.compile_seconds += result.wall_seconds
        return artifact

    def measure(self, baseline: Path, candidate: Path, directory: Path) -> dict[str, Any]:
        replacements = {
            "baseline": str(baseline),
            "candidate": str(candidate),
            "function": self.args.function,
            "region": str(self.args.region),
            "output_dir": str(directory),
        }
        command = [argument.format_map(replacements) for argument in self.args.benchmark_command]
        result = run_command(command, directory / "benchmark.log", self.args.timeout)
        self.hardware_evaluations += 1
        self.benchmark_seconds += result.wall_seconds
        return benchmark_result(result.stdout)

    def compile_founder(self) -> dict[str, Any]:
        directory = self.output / "founder"
        directory.mkdir()
        schedule = directory / "schedule.tsv"
        trajectory = directory / "compiler-trajectory.jsonl"
        self.invoke_llc(
            self.llc_command(directory, schedule, trajectory), directory / "compile.log"
        )
        region_size = trajectory_region_size(trajectory, self.args.region)
        if region_size < 2:
            raise RuntimeError(
                f"mutation region {self.args.region} has only {region_size} instruction"
            )
        artifact = self.link(directory)
        measurement = self.measure(artifact, artifact, directory)
        if not measurement["correct"]:
            raise RuntimeError("founder failed benchmark correctness")
        record = {
            "kind": "founder",
            "schedule": str(schedule),
            "schedule_sha256": sha256(schedule),
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
            "mutation_region_size": region_size,
            "runtime_ratio": 1.0,
            "benchmark": measurement,
        }
        append_jsonl(self.events, record)
        return record

    def compile_candidate(
        self, parent: dict[str, Any], evaluation: int, attempt: int, depth: int
    ) -> dict[str, Any]:
        identifier = f"e{evaluation:06d}-a{attempt:02d}-d{depth}"
        directory = self.output / "candidates" / identifier
        directory.mkdir(parents=True)
        schedule = directory / "schedule.tsv"
        trajectory = directory / "compiler-trajectory.jsonl"
        mutation_seed = derived_seed(self.args.seed, evaluation, attempt)
        self.invoke_llc(
            self.llc_command(
                directory,
                schedule,
                trajectory,
                Path(parent["schedule"]),
                depth,
                mutation_seed,
            ),
            directory / "compile.log",
        )

        verified = directory / "verified"
        verified.mkdir()
        verified_schedule = verified / "schedule.tsv"
        self.invoke_llc(
            self.llc_command(
                verified, verified_schedule, None, replay=schedule
            ),
            verified / "compile.log",
        )
        if schedule.read_bytes() != verified_schedule.read_bytes():
            raise RuntimeError(f"schedule replay mismatch for {identifier}")
        artifact = self.link(verified)
        return {
            "id": identifier,
            "directory": str(directory),
            "schedule": str(schedule),
            "schedule_sha256": sha256(schedule),
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
            "depth": depth,
            "mutation_seed": mutation_seed,
        }

    def summary(
        self,
        founder: dict[str, Any],
        current: dict[str, Any],
        best: dict[str, Any],
        evaluations: int,
        duplicates: int,
    ) -> dict[str, Any]:
        best_ratio = float(best["runtime_ratio"])
        return {
            "status": "complete" if evaluations == self.args.budget else "running",
            "budget": self.args.budget,
            "evaluations": evaluations,
            "duplicates": duplicates,
            "seed": self.args.seed,
            "depths": list(self.args.depths),
            "initial_temperature": self.args.temperature,
            "cooling": self.args.cooling,
            "temperature_floor": self.args.temperature_floor,
            "founder_schedule_sha256": founder["schedule_sha256"],
            "current_schedule_sha256": current["schedule_sha256"],
            "best_schedule_sha256": best["schedule_sha256"],
            "best_runtime_ratio": best_ratio,
            "best_speedup_pct": (1.0 / best_ratio - 1.0) * 100.0,
            "compiler_invocations": self.compiler_invocations,
            "hardware_evaluations": self.hardware_evaluations,
            "compile_seconds": self.compile_seconds,
            "benchmark_seconds": self.benchmark_seconds,
            "wall_seconds": time.monotonic() - self.started,
            "visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        }

    def run(self) -> dict[str, Any]:
        config = vars(self.args).copy()
        config.update(
            llc=str(self.args.llc),
            ld_lld=str(self.args.ld_lld),
            input=str(self.args.input),
            output_dir=str(self.output),
            depths=list(self.args.depths),
        )
        write_json(self.output / "config.json", config)

        founder = self.compile_founder()
        current = founder
        best = founder
        seen = {founder["schedule_sha256"]}
        evaluations = 0
        duplicates = 0
        temperature = self.args.temperature

        while evaluations < self.args.budget:
            depth = self.args.depths[evaluations % len(self.args.depths)]
            candidate = None
            last_error = None
            for attempt in range(self.args.max_proposal_attempts):
                try:
                    proposal = self.compile_candidate(
                        current, evaluations + 1, attempt, depth
                    )
                except RuntimeError as error:
                    last_error = error
                    append_jsonl(
                        self.events,
                        {
                            "kind": "compile_failure",
                            "evaluation": evaluations + 1,
                            "attempt": attempt,
                            "depth": depth,
                            "error": str(error),
                        },
                    )
                    continue
                if proposal["schedule_sha256"] in seen:
                    duplicates += 1
                    append_jsonl(
                        self.events,
                        {
                            "kind": "duplicate",
                            "evaluation": evaluations + 1,
                            **proposal,
                        },
                    )
                    continue
                candidate = proposal
                break
            if candidate is None:
                detail = f": {last_error}" if last_error else ""
                raise RuntimeError(
                    "could not generate a unique legal schedule proposal" + detail
                )

            seen.add(candidate["schedule_sha256"])
            measurement = self.measure(
                Path(founder["artifact"]), Path(candidate["artifact"]),
                Path(candidate["directory"]),
            )
            evaluations += 1
            ratio = measurement["candidate_runtime_us"] / measurement["baseline_runtime_us"]
            probability = metropolis_probability(
                float(current["runtime_ratio"]), ratio, temperature
            )
            draw = self.rng.random()
            accepted = bool(measurement["correct"] and draw < probability)
            candidate.update(
                runtime_ratio=ratio,
                benchmark=measurement,
                correct=measurement["correct"],
            )
            if measurement["correct"] and ratio < float(best["runtime_ratio"]):
                best = candidate
            previous = current
            if accepted:
                current = candidate
            event = {
                "kind": "proposal",
                "evaluation": evaluations,
                "temperature": temperature,
                "parent_schedule_sha256": previous["schedule_sha256"],
                "acceptance_probability": probability,
                "acceptance_draw": draw,
                "accepted": accepted,
                "best_schedule_sha256": best["schedule_sha256"],
                "best_runtime_ratio": best["runtime_ratio"],
                **candidate,
            }
            append_jsonl(self.events, event)
            temperature = max(self.args.temperature_floor, temperature * self.args.cooling)
            write_json(
                self.output / "summary.json",
                self.summary(founder, current, best, evaluations, duplicates),
            )

        shutil.copy2(best["schedule"], self.output / "best.schedule")
        shutil.copy2(best["artifact"], self.output / f"best{Path(best['artifact']).suffix}")
        shutil.copy2(current["schedule"], self.output / "current.schedule")
        summary = self.summary(founder, current, best, evaluations, duplicates)
        write_json(self.output / "summary.json", summary)
        return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--llc", type=Path, required=True)
    result.add_argument("--ld-lld", type=Path)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--function", required=True)
    result.add_argument("--region", type=int, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--budget", type=int, default=16)
    result.add_argument("--depths", type=parse_depths, default=parse_depths("1,2,4,8"))
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--temperature", type=float, default=0.01)
    result.add_argument("--cooling", type=float, default=0.985)
    result.add_argument("--temperature-floor", type=float, default=0.0005)
    result.add_argument("--max-proposal-attempts", type=int, default=16)
    result.add_argument("--timeout", type=int, default=1200)
    result.add_argument("--triple", default="amdgcn-amd-amdhsa")
    result.add_argument("--mcpu", default="gfx950")
    result.add_argument("--artifact-kind", choices=("hsaco", "object"), default="hsaco")
    result.add_argument(
        "--llc-arg", action="append", default=[],
        help="additional llc argument; use --llc-arg=-foo for option values",
    )
    result.add_argument(
        "--benchmark-command", nargs=argparse.REMAINDER, required=True,
        help="benchmark argv (must be last); supports {baseline} and {candidate}",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if args.region < 0:
        raise SystemExit("--region must be non-negative")
    if args.max_proposal_attempts <= 0:
        raise SystemExit("--max-proposal-attempts must be positive")
    if args.temperature < 0 or args.temperature_floor < 0:
        raise SystemExit("temperatures must be non-negative")
    if not 0 < args.cooling <= 1:
        raise SystemExit("--cooling must be in (0, 1]")
    if not args.benchmark_command:
        raise SystemExit("--benchmark-command requires a command")
    # Preserve tool symlink names.  In particular, LLD selects its driver from
    # argv[0], so resolving ld.lld to the generic `lld` binary changes behavior.
    args.llc = Path(os.path.abspath(args.llc))
    args.input = args.input.resolve()
    if args.ld_lld is None:
        args.ld_lld = args.llc.with_name("ld.lld")
    else:
        args.ld_lld = Path(os.path.abspath(args.ld_lld))
    try:
        summary = Controller(args).run()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
