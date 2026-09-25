"""Install into the invoking Python environment, optionally for Pascal GPUs."""

import argparse
import platform
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_TORCH = "torch==2.14.0+cu126"
LEGACY_INDEX = "https://download.pytorch.org/whl/cu126"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-cuda", action="store_true",
        help="install pinned CUDA 12.6 PyTorch for GTX 1080 / Pascal GPUs",
    )
    parser.add_argument("--notebook", action="store_true", help="include notebook dependencies")
    parser.add_argument("--test", action="store_true", help="include test dependencies")
    parser.add_argument("--dry-run", action="store_true", help="print commands without installing")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required")
    if args.legacy_cuda and (
        platform.system() not in ("Linux", "Windows")
        or platform.machine().lower() not in ("x86_64", "amd64")
    ):
        parser.error("--legacy-cuda supports Linux/Windows on x86_64")

    extras = [name for name in ("notebook", "test") if getattr(args, name)]
    package = str(ROOT) + ("[" + ",".join(extras) + "]" if extras else "")
    pip = [sys.executable, "-m", "pip", "install"]
    commands = []
    if args.legacy_cuda:
        commands.append(pip + [LEGACY_TORCH, "--index-url", LEGACY_INDEX])
    # Keep the exact CUDA build during project dependency resolution, too.
    commands.append(pip + ["-e", package] + ([LEGACY_TORCH] if args.legacy_cuda else []))
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
