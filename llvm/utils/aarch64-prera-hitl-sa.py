#!/usr/bin/env python3
"""Run hardware-in-the-loop SA over Cortex-A53 pre-RA schedules.

This target adapter reuses the reference serial SA controller.  LLVM compiles
and replay-verifies AArch64 objects locally; --benchmark-command may invoke
aarch64-hitl-remote.py to transfer and measure them on a remote board.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def load_controller_module():
    path = Path(__file__).with_name("amdgpu-prera-hitl-sa.py")
    spec = importlib.util.spec_from_file_location("machine_sched_hitl_sa", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared SA controller from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_controller_module()


class Controller(base.Controller):
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
            f"-aarch64-prera-training-function={self.args.function}",
            f"-aarch64-prera-training-record-schedule={schedule}",
        ]
        command.extend(self.args.llc_arg)
        if trajectory is not None:
            command.append(
                f"-aarch64-prera-training-record-trajectory={trajectory}"
            )
        if replay is not None:
            command.append(f"-aarch64-prera-training-replay-schedule={replay}")
        if depth is not None:
            command.extend(
                [
                    f"-aarch64-prera-training-mutation-region={self.args.region}",
                    f"-aarch64-prera-training-mutation-depth={depth}",
                    f"-aarch64-prera-training-seed={mutation_seed}",
                ]
            )
        return command

    def link(self, directory: Path, stem: str = "kernel") -> Path:
        return directory / f"{stem}.o"

    def measure(self, baseline: Path, candidate: Path, directory: Path):
        for attempt in range(self.args.measurement_retries + 1):
            measurement = super().measure(baseline, candidate, directory)
            if measurement.get("stable", True):
                return measurement
            base.append_jsonl(
                self.events,
                {
                    "kind": "unstable_measurement",
                    "attempt": attempt,
                    "baseline": str(baseline),
                    "candidate": str(candidate),
                    "benchmark": measurement,
                },
            )
        raise RuntimeError("remote board remained throttled during measurement")


def parser():
    result = base.parser()
    result.description = __doc__
    result.set_defaults(
        triple="aarch64-linux-gnu", mcpu="cortex-a53", artifact_kind="object"
    )
    result.add_argument(
        "--measurement-retries", type=int, default=2,
        help="retry measurements reported as thermally or otherwise unstable",
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
    if args.measurement_retries < 0:
        raise SystemExit("--measurement-retries must be non-negative")
    if args.temperature < 0 or args.temperature_floor < 0:
        raise SystemExit("temperatures must be non-negative")
    if not 0 < args.cooling <= 1:
        raise SystemExit("--cooling must be in (0, 1]")
    if not args.benchmark_command:
        raise SystemExit("--benchmark-command requires a command")
    args.llc = Path(os.path.abspath(args.llc))
    args.input = args.input.resolve()
    if args.ld_lld is None:
        args.ld_lld = args.llc.with_name("ld.lld")
    try:
        summary = Controller(args).run()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(base.json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
