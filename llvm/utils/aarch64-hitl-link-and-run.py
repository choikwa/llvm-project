#!/usr/bin/env python3
"""Link cached schedule objects as DSOs and invoke a native HITL runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def shared_object(obj: Path, cc: str) -> Path:
    result = obj.with_suffix(obj.suffix + ".so")
    if not result.exists() or result.stat().st_mtime_ns < obj.stat().st_mtime_ns:
        temporary = result.with_suffix(result.suffix + ".tmp")
        subprocess.run(
            [cc, "-shared", "-Wl,-z,defs", str(obj), "-o", str(temporary)],
            check=True,
        )
        temporary.replace(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()
    if args.iterations <= 0 or args.trials <= 0:
        raise SystemExit("iterations and trials must be positive")
    baseline = shared_object(args.baseline, args.cc)
    candidate = shared_object(args.candidate, args.cc)
    subprocess.run(
        [
            str(args.runner), str(baseline), str(candidate),
            str(args.iterations), str(args.trials),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
