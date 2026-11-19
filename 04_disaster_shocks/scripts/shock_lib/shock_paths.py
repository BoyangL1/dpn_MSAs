"""Paths for external inputs and local analysis outputs."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    Path(os.environ.get("DPN_DATA_DIR", ROOT.parent / "data")).expanduser().resolve()
)
DATA_DIR = DATA_ROOT / "disaster_shocks"
TRANSFER_ROOT = ROOT.parent / "02_transferability"
