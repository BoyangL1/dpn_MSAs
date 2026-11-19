#!/usr/bin/env python3
"""Run network statistics and generate the associated figures."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel MSA workers for the distance-decay estimator",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    commands = [
        ["01_prepare_network.py"],
        ["02A_plot_network_map.py"],
        ["02B_plot_strength_distributions.py"],
        ["02C_plot_network_statistics.py"],
        ["02D_plot_dependency_weights.py"],
        ["02E_plot_shared_colorbar.py"],
        ["03_distribution_similarity.py"],
        ["04_distance_decay.py", "--workers", str(args.workers)],
    ]
    for command in commands:
        print(f"\nRunning {command[0]}", flush=True)
        subprocess.run(
            [sys.executable, str(HERE / command[0]), *command[1:]], check=True, env=env
        )
    print(
        f'\nComplete: eight figures plus one shared colorbar in {HERE.parent / "figures"} '
        f"({time.perf_counter() - started:.1f} s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
