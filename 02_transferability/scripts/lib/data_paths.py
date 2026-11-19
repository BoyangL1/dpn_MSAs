"""External input locations."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    Path(os.environ.get("DPN_DATA_DIR", ROOT.parent / "data")).expanduser().resolve()
)
GEOGRAPHY = DATA_DIR / "geography"
CBSA = Path(
    os.environ.get(
        "DPN_CBSA_PATH", GEOGRAPHY / "cb_2023_us_cbsa_500k/cb_2023_us_cbsa_500k.shp"
    )
)
STATES = Path(
    os.environ.get(
        "DPN_STATES_PATH", GEOGRAPHY / "cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
    )
)
ACS = DATA_DIR / "acs_2021_cbg_us_model_vars.csv"
POI = DATA_DIR / "poi_us_msa376_cleaned.parquet"
EDGES = DATA_DIR / "us_dependency_network_20210101_20211231_new_nij_gt50_cleaned.csv"
