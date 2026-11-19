from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent / "02_transferability"
sys.path.insert(0, str(MAIN / "scripts"))
from lib.data_paths import DATA_DIR, ACS, POI, EDGES, CBSA, GEOGRAPHY

__all__ = [
    "ROOT",
    "MAIN",
    "INPUTS",
    "RESULTS",
    "PUBLISHED",
    "FIGURES",
    "CONFIG",
    "DATA_DIR",
    "ACS",
    "POI",
    "EDGES",
    "CBSA",
    "GEOGRAPHY",
]

INPUTS = ROOT / ".work/inputs"
RESULTS = ROOT / ".work/results"
PUBLISHED = ROOT / "results"
FIGURES = ROOT / "figures"
CONFIG = MAIN / "config"
