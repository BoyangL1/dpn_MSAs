"""Prepare hazard, geographic, and functional inputs from external source files."""

from pathlib import Path
import argparse
import os
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from shock_lib.network_weights import attach_node_visits, attach_original_weights
from shock_lib.scenarios import SCENARIOS
from shock_lib.shock_paths import DATA_DIR, DATA_ROOT, TRANSFER_ROOT

HAZARDS = ["IFLD", "HRCN", "ERQK", "WFIR", "TRND"]
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=TRANSFER_ROOT / ".work/build")
    parser.add_argument(
        "--transfer-results", type=Path, default=TRANSFER_ROOT / "results"
    )
    parser.add_argument("--transfer-work", type=Path, default=TRANSFER_ROOT / ".work")
    parser.add_argument("--geography-dir", type=Path, default=DATA_ROOT / "geography")
    parser.add_argument("--cbsa-file", type=Path)
    parser.add_argument("--states-file", type=Path)
    parser.add_argument("--tracts-file", type=Path)
    parser.add_argument(
        "--poi-file", type=Path, default=DATA_ROOT / "poi_us_msa376_cleaned.parquet"
    )
    parser.add_argument(
        "--network-file",
        type=Path,
        default=DATA_ROOT
        / "us_dependency_network_20210101_20211231_new_nij_gt50_cleaned.csv",
    )
    parser.add_argument(
        "--hazard-file", type=Path, default=DATA_ROOT / "hazard_scores.parquet"
    )
    args = parser.parse_args()
    cbsa_file = (
        args.cbsa_file
        or Path(
            os.environ.get(
                "DPN_CBSA_PATH",
                args.geography_dir / "cb_2023_us_cbsa_500k/cb_2023_us_cbsa_500k.shp",
            )
        ).expanduser()
    )
    states_file = (
        args.states_file
        or Path(
            os.environ.get(
                "DPN_STATES_PATH",
                args.geography_dir / "cb_2018_us_state_500k/cb_2018_us_state_500k.shp",
            )
        ).expanduser()
    )
    tracts_file = args.tracts_file or args.geography_dir / "nri_tracts.shp"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(
        args.graph_dir / "nodes.parquet",
        columns=[
            "idx",
            "PLACEKEY",
            "msa_geoid",
            "cbg",
            "lat",
            "lon",
            "ni_out",
            "ni_in",
        ],
    )
    nodes = nodes.sort_values("idx").reset_index(drop=True)
    nodes["msa_geoid"] = nodes.msa_geoid.astype(str).str.zfill(5)
    assert nodes.idx.is_unique and nodes.PLACEKEY.is_unique
    crosswalk = (
        pd.read_csv(args.geography_dir / "ct_tract_crosswalk.csv", dtype=str)
        .set_index("old_geoid")
        .new_geoid
    )
    nodes["tract"] = nodes.cbg.astype(str).str.zfill(12).str[:11]
    mask = nodes.tract.str.startswith("09")
    nodes.loc[mask, "tract"] = (
        nodes.loc[mask, "tract"].map(crosswalk).fillna(nodes.loc[mask, "tract"])
    )
    nodes = nodes.drop(columns="cbg")
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='16GB'")
    connection.register("active_nodes", nodes[["idx", "PLACEKEY"]])
    attributes = connection.execute(
        "\n        SELECT n.idx, p.LOCATION_NAME, p.CITY, p.NAICS_CODE\n        FROM active_nodes n JOIN read_parquet(?) p USING(PLACEKEY)\n        ORDER BY n.idx\n    ",
        [str(args.poi_file)],
    ).fetchdf()
    assert np.array_equal(attributes.idx, nodes.idx)
    nodes["LOCATION_NAME"] = attributes.LOCATION_NAME.fillna("Unknown POI")
    nodes["CITY"] = attributes.CITY.fillna("")
    edges = connection.execute(
        "\n        SELECT src_idx, tgt_idx, nij, msa_geoid\n        FROM read_parquet(?, file_row_number=true)\n        WHERE nij>0 ORDER BY file_row_number\n    ",
        [str(args.graph_dir / "edges.parquet")],
    ).fetchdf()
    connection.close()
    edges["msa_geoid"] = edges.msa_geoid.astype(str).str.zfill(5)
    assert not edges.duplicated(["src_idx", "tgt_idx"]).any()
    edges = attach_original_weights(nodes, edges, args.network_file)
    nodes = attach_node_visits(nodes, args.network_file)
    nodes.to_parquet(DATA_DIR / "nodes.parquet", index=False, compression="zstd")
    edges.to_parquet(DATA_DIR / "edges.parquet", index=False, compression="zstd")
    scores = pd.read_parquet(args.hazard_file, columns=["TRACTFIPS", *HAZARDS])
    scores["TRACTFIPS"] = scores.TRACTFIPS.astype(str).str.zfill(11)
    assert scores.TRACTFIPS.is_unique
    assert scores[HAZARDS].stack().between(0, 1).all()
    scores.to_parquet(
        DATA_DIR / "hazard_scores.parquet", index=False, compression="zstd"
    )
    metros = gpd.read_file(cbsa_file, columns=["GEOID", "NAME", "geometry"])
    metros["GEOID"] = metros.GEOID.astype(str).str.zfill(5)
    metros = (
        metros.loc[metros.GEOID.isin(nodes.msa_geoid)]
        .sort_values("GEOID")
        .reset_index(drop=True)
    )
    assert set(metros.GEOID) == set(nodes.msa_geoid)
    metros.to_parquet(
        DATA_DIR / "msa_boundaries.parquet", index=False, compression="zstd"
    )
    states = gpd.read_file(states_file)[["STATEFP", "STUSPS", "NAME", "geometry"]]
    states = states.loc[~states.STATEFP.isin(NON_CONUS)]
    states.to_parquet(DATA_DIR / "states.parquet", index=False, compression="zstd")
    tracts = gpd.read_file(
        tracts_file,
        columns=["TRACTFIPS", "STCOFIPS", "STATEABBRV", "geometry"],
    )
    tracts["TRACTFIPS"] = tracts.TRACTFIPS.astype(str).str.zfill(11)
    assert tracts.TRACTFIPS.is_unique
    tracts = tracts.loc[~tracts.TRACTFIPS.str[:2].isin(NON_CONUS)].reset_index(
        drop=True
    )
    local_metros = metros.to_crs(tracts.crs)
    points = gpd.GeoDataFrame(
        tracts[["TRACTFIPS"]],
        geometry=tracts.geometry.representative_point(),
        crs=tracts.crs,
    )
    assigned = gpd.sjoin(
        points, local_metros[["GEOID", "geometry"]], how="left", predicate="within"
    )
    assigned = assigned.loc[~assigned.index.duplicated()].sort_index()
    assert len(assigned) == len(tracts)
    tracts["msa_spatial"] = assigned.GEOID.to_numpy()
    touching = gpd.sjoin(
        tracts[["geometry"]],
        local_metros[["geometry"]],
        how="inner",
        predicate="intersects",
    ).index.unique()
    top = set(scores.loc[scores[HAZARDS].ge(0.99).any(axis=1), "TRACTFIPS"])
    tracts = tracts.loc[
        tracts.index.isin(touching)
        | tracts.TRACTFIPS.isin(nodes.tract)
        | tracts.TRACTFIPS.isin(top)
    ]
    tracts.to_parquet(DATA_DIR / "tracts.parquet", index=False, compression="zstd")
    performance = pd.read_csv(
        args.transfer_work / "within_msa_performance.csv", dtype={"geoid": str}
    )
    cohort = performance[["geoid", "name", "best_epoch", "retention_reason"]].rename(
        columns={"geoid": "msa_geoid", "name": "msa_name"}
    )
    cohort["included_in_coverage"] = cohort.best_epoch.gt(1)
    cohort.to_csv(DATA_DIR / "hazard_coverage_cohort.csv", index=False)
    members = pd.read_csv(
        args.transfer_results / "gnn_all_pairs/k_8/cluster_membership.csv",
        dtype={"geoid": str},
    )
    assert set(members.geoid) == set(
        cohort.loc[cohort.included_in_coverage, "msa_geoid"]
    )
    members.to_csv(DATA_DIR / "functional_cluster_membership_k8.csv", index=False)
    labels = nodes[["idx", "PLACEKEY"]].copy()
    labels["primary_naics"] = (
        attributes.NAICS_CODE.fillna("")
        .astype(str)
        .str.split("\\s*\\|\\s*", n=1, regex=True)
        .str[0]
    )
    labels["naics_code"] = ""
    for code, _ in sorted(SCENARIOS, key=lambda item: len(item[0])):
        labels.loc[labels.primary_naics.str.startswith(code), "naics_code"] = code
    labels.to_parquet(
        DATA_DIR / "functional_node_categories.parquet", index=False, compression="zstd"
    )
    print(
        f"Prepared {len(nodes):,} nodes, {len(edges):,} edges, and {len(members)} study MSAs."
    )


if __name__ == "__main__":
    main()
