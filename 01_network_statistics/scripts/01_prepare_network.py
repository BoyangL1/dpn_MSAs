#!/usr/bin/env python3
"""Prepare network tables and metropolitan statistics."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dpn_matplotlib")

import duckdb
import geopandas as gpd
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    Path(os.environ.get("DPN_DATA_DIR", ROOT.parent / "data")).expanduser().resolve()
)
CBSA = Path(
    os.environ.get(
        "DPN_CBSA_PATH",
        DATA_DIR / "geography/cb_2023_us_cbsa_500k/cb_2023_us_cbsa_500k.shp",
    )
).expanduser()


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def run(root=ROOT):
    root = Path(root).resolve()
    out = root / ".work"
    out.mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    (out / "ready.json").unlink(missing_ok=True)
    poi = DATA_DIR / "poi_us_msa376_cleaned.parquet"
    network = (
        DATA_DIR / "us_dependency_network_20210101_20211231_new_nij_gt50_cleaned.csv"
    )
    font = fm.findfont(fm.FontProperties(family="Arial"), fallback_to_default=False)
    assert fm.FontProperties(fname=font).get_name() == "Arial"
    log("Reading cleaned network and matching complete polygon IDs to cleaned POIs")
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='32GB'")
    con.execute("CREATE TABLE edges AS SELECT * FROM read_csv_auto(?)", [str(network)])
    con.execute(
        """CREATE TABLE node_ids AS
        SELECT row_number() OVER (ORDER BY node)-1 AS node_id, node FROM
        (SELECT SOURCE AS node FROM edges UNION SELECT TARGET AS node FROM edges)"""
    )
    nodes = con.execute(
        """SELECT n.node_id,n.node,p.PLACEKEY,p.LATITUDE,p.LONGITUDE
        FROM node_ids n LEFT JOIN read_parquet(?) p ON n.node=p.PLACEKEY
        ORDER BY n.node_id""",
        [str(poi)],
    ).fetchdf()
    n = con.execute("SELECT COUNT(*) FROM node_ids").fetchone()[0]
    assert (
        len(nodes) == n and nodes.PLACEKEY.notna().all()
    ), "IDs must match exactly once"
    assert nodes.node.is_unique
    assert np.isfinite(nodes[["LATITUDE", "LONGITUDE"]].to_numpy()).all()
    log(
        f"Spatial join for {n:,} active nodes using stored POI coordinates (within predicate)"
    )
    cbsa = gpd.read_file(CBSA)[["GEOID", "NAME", "LSAD", "geometry"]]
    metro = cbsa.loc[cbsa.LSAD.eq("M1")].copy()
    points = gpd.GeoDataFrame(
        {"node_id": nodes.node_id},
        geometry=gpd.points_from_xy(nodes.LONGITUDE, nodes.LATITUDE, crs="EPSG:4326"),
    ).to_crs(metro.crs)
    joined = gpd.sjoin(points, metro, how="left", predicate="within").sort_values(
        "node_id"
    )
    assert (
        len(joined) == n and joined.GEOID.notna().all()
    ), "Missing/overlapping MSA assignments"
    nodes["GEOID"] = joined.GEOID.to_numpy()
    nodes["NAME"] = joined.NAME.to_numpy()
    con.register("node_msa", nodes[["node_id", "node", "GEOID"]])
    edges = con.execute(
        """SELECT s.node_id AS src_idx,t.node_id AS tgt_idx,
        e.NIJ,e.NI_SOURCE,e.NJ_TARGET,e.WIJ,s.GEOID AS source_msa,t.GEOID AS target_msa,
        s.GEOID=t.GEOID AS is_intra_msa FROM edges e
        JOIN node_msa s ON e.SOURCE=s.node JOIN node_msa t ON e.TARGET=t.node
        ORDER BY e.rowid"""
    ).fetchdf()
    assert len(edges) == con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert (edges.NIJ > 50).all() and (edges.NI_SOURCE > 0).all()
    assert np.isfinite(edges.WIJ).all() and (edges.WIJ > 0).all()
    assert np.allclose(edges.WIJ, edges.NIJ / edges.NI_SOURCE, rtol=1e-14, atol=0)
    assert not (edges.src_idx == edges.tgt_idx).any()
    assert not edges.duplicated(["src_idx", "tgt_idx"]).any()
    src, tgt, w = (
        edges.src_idx.to_numpy(),
        edges.tgt_idx.to_numpy(),
        edges.WIJ.to_numpy(),
    )
    nodes["in_str"] = np.bincount(tgt, weights=w, minlength=n)
    nodes["out_str"] = np.bincount(src, weights=w, minlength=n)
    nodes["in_degree"] = np.bincount(tgt, minlength=n)
    nodes["out_degree"] = np.bincount(src, minlength=n)
    assert np.isclose(nodes.in_str.sum(), w.sum()) and np.isclose(
        nodes.out_str.sum(), w.sum()
    )
    assert nodes.in_degree.sum() == nodes.out_degree.sum() == len(edges)
    summary = nodes.groupby("GEOID").size().rename("n_nodes").to_frame()
    summary["n_edges"] = edges.loc[edges.is_intra_msa].groupby("source_msa").size()
    summary["n_edges"] = summary.n_edges.fillna(0).astype("int64")
    summary["mean_out_degree"] = summary.n_edges / summary.n_nodes
    summary = summary.reset_index().merge(
        metro[["GEOID", "NAME", "LSAD"]], on="GEOID", validate="one_to_one"
    )
    summary = summary[
        ["GEOID", "NAME", "LSAD", "n_nodes", "n_edges", "mean_out_degree"]
    ].sort_values("GEOID")
    from study_cohort import MSA_IDS

    expected_geoids = set(MSA_IDS)
    assert set(summary.GEOID) == expected_geoids, "Must reproduce the selected 376 MSAs"
    log("Writing fresh node, edge and MSA statistics")
    nodes.to_parquet(out / "network_nodes.parquet", index=False)
    edges.to_parquet(out / "network_edges.parquet", index=False)
    summary.to_csv(out / "msa_summary.csv", index=False)
    config = {
        "font_family": "Arial",
        "font_path": font,
        "norm_vmin": int(summary.n_nodes.min()),
        "norm_vmax": int(summary.n_nodes.max()),
        "node_count_vmin": int(summary.n_nodes.min()),
        "node_count_vmax": int(summary.n_nodes.max()),
        "colormap": "viridis",
        "colormap_sampling_min": 0.03,
        "colormap_sampling_max": 0.88,
        "normalization": "LogNorm",
        "color_variable": "Active national-network nodes in each MSA",
        "cbsa_shapefile": str(CBSA),
        "boundary_predicate": "within",
    }
    (out / "plot_config.json").write_text(json.dumps(config, indent=2) + "\n")
    log("Fingerprinting inputs for provenance")
    manifest = {
        "inputs": {"poi": fingerprint(poi), "network": fingerprint(network)},
        "n_msa": len(summary),
        "n_nodes": n,
        "n_edges": len(edges),
        "n_intra_msa_edges": int(edges.is_intra_msa.sum()),
        "n_cross_msa_edges": int((~edges.is_intra_msa).sum()),
        "total_wij": float(w.sum()),
        "coordinate_method": "Exact full PLACEKEY join; stored polygon latitude/longitude",
        "msa_assignment": f"Fresh point-within-MSA join against {CBSA}; LSAD M1",
        "node_population": "Union of SOURCE and TARGET in the cleaned national network",
        "strength_population": "All retained network edges including cross-MSA links",
        "basic_edge_and_ccdf_population": "Retained edges whose endpoints belong to the same MSA",
        "validation": {
            "all_endpoints_match_poi": True,
            "no_duplicate_full_keys": True,
            "all_coordinates_finite": True,
            "all_nodes_in_selected_376_msa": True,
            "no_self_loops_or_duplicate_directed_edges": True,
            "wij_equals_nij_divided_by_ni_source": True,
            "degree_and_strength_conservation": True,
        },
    }
    (out / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (out / "ready.json").write_text(
        json.dumps({"complete": True, "n_nodes": n, "n_edges": len(edges)}) + "\n"
    )
    log(
        f"Ready: {len(summary)} MSAs, {n:,} nodes, {len(edges):,} edges; "
        f"{int(edges.is_intra_msa.sum()):,} within-MSA edges"
    )
    con.close()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
