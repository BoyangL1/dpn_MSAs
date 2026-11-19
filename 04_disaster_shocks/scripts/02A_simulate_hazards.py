"""Simulate hazard shocks and equal-size random controls."""

from pathlib import Path
import argparse
from shock_lib.shock_paths import ROOT
from shock_lib.propagation import simulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    simulate(parser.parse_args().root)
