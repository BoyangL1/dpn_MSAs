#!/usr/bin/env python3
"""Build node features, distance-matched negative edges, and unordered-pair splits."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from .pair_split import SPLIT, split_pairs_per_msa

ROOT = Path(__file__).resolve().parents[2]
from .data_paths import GEOGRAPHY, CBSA, ACS, POI, EDGES

SEED, K_NEG, KNN_K, EARTH_KM = 0, 5, 512, 6371.0088
DIST_BINS_KM = np.array(
    [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0, 50.0, 100.0, np.inf]
)
ACS_COLS = [
    "population",
    "households",
    "median_household_income",
    "per_capita_income",
    "median_home_value",
    "median_gross_rent",
    "unemployment_rate",
    "car_commute_share",
    "public_transit_share",
    "no_vehicle_share",
    "renter_share",
    "bachelor_plus_share",
    "young_adult_share_18_34",
    "senior_share_65plus",
    "white_share",
    "black_share",
    "asian_share",
    "hispanic_share",
]


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spatial_covariates(poi, reference):
    points = gpd.GeoDataFrame(
        {"idx": poi.idx.to_numpy()},
        geometry=gpd.points_from_xy(poi.lon, poi.lat, crs="EPSG:4326"),
    )
    metro = gpd.read_file(CBSA)[["GEOID", "NAME", "LSAD", "geometry"]]
    metro = metro.loc[metro.LSAD.eq("M1")].copy()
    join = gpd.sjoin(
        points.to_crs(metro.crs), metro, how="left", predicate="within"
    ).sort_values("idx")
    assert (
        len(join) == len(poi) and join.GEOID.notna().all()
    ), "Every node needs exactly one MSA"
    poi["msa_geoid"] = join.GEOID.to_numpy()
    boundaries = sorted(reference.glob("tl_2021_bg/*/*.shp"))
    log(f"Loading {len(boundaries)} original TIGER block-group shapefiles")
    frames = []
    for path in boundaries:
        frame = gpd.read_file(
            path, columns=["GEOID", "STATEFP", "COUNTYFP", "geometry"]
        )
        frames.append(frame[["GEOID", "STATEFP", "COUNTYFP", "geometry"]])
    bg = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs
    ).to_crs("EPSG:4326")
    log(f"Spatial join to {len(bg):,} block groups")
    join = (
        gpd.sjoin(points, bg, how="left", predicate="within")
        .drop_duplicates("idx", keep="first")
        .sort_values("idx")
    )
    assert len(join) == len(poi)
    poi["cbg"] = join.GEOID.fillna("").to_numpy()
    state = join.STATEFP.fillna("").to_numpy()
    county = join.COUNTYFP.fillna("").to_numpy()
    poi["county_fips"] = [
        s + c if s and c else "UNKNOWN" for s, c in zip(state, county)
    ]
    poi["county_id"], county_labels = pd.factorize(poi.county_fips)
    poi["county_id"] = poi.county_id.astype(np.int32)
    log(f'Unmatched CBG nodes: {(poi.county_fips == "UNKNOWN").sum():,}')
    return poi, metro[["GEOID", "NAME"]], len(county_labels), len(boundaries)


def sample_negatives(
    node_msa, lat, lon, src_pos, tgt_pos, bin_lo, bin_hi, pos_keys_sorted, n_nodes, rng
):
    """Original 512-NN distance-bin draws; all RNG calls retain source order."""
    chunks_src, chunks_tgt = [], []
    pos_msa = node_msa[src_pos]
    order = np.argsort(pos_msa, kind="stable")
    sorted_msa = pos_msa[order]
    bounds = np.flatnonzero(np.r_[True, sorted_msa[1:] != sorted_msa[:-1], True])
    for i, (left, right) in enumerate(zip(bounds[:-1], bounds[1:])):
        rows = order[left:right]
        pool = np.flatnonzero(node_msa == sorted_msa[left])
        k = min(KNN_K, len(pool))
        if len(pool) < 2:
            continue
        coords = np.deg2rad(np.column_stack([lat[pool], lon[pool]]))
        distances, neighbors = BallTree(coords, metric="haversine").query(coords, k=k)
        distances *= EARTH_KM
        lookup = np.full(n_nodes, -1, dtype=np.int64)
        lookup[pool] = np.arange(len(pool))
        local_src = lookup[src_pos[rows]]
        row_distances = distances[local_src]
        rank_lo = (row_distances < bin_lo[rows, None]).sum(axis=1)
        rank_hi = (row_distances < bin_hi[rows, None]).sum(axis=1)
        count = np.maximum(rank_hi - rank_lo, 0)
        del row_distances
        uniforms = rng.random((len(rows), K_NEG))
        rank = (rank_lo[:, None] + uniforms * np.maximum(count, 1)[:, None]).astype(
            np.int64
        )
        rank = np.minimum(rank, k - 1)
        local_negative = neighbors[local_src[:, None], rank]
        empty = count == 0
        if empty.any():
            local_negative[empty] = rng.integers(
                0, len(pool), size=(int(empty.sum()), K_NEG)
            )
        negative = pool[local_negative]
        source = src_pos[rows, None]
        keys = source * np.int64(n_nodes) + negative
        positions = np.minimum(
            np.searchsorted(pos_keys_sorted, keys.ravel()), len(pos_keys_sorted) - 1
        )
        observed = (pos_keys_sorted[positions] == keys.ravel()).reshape(keys.shape)
        bad = (negative == source) | (negative == tgt_pos[rows, None]) | observed
        keep = ~bad.ravel()
        chunks_src.append(np.repeat(src_pos[rows], K_NEG)[keep])
        chunks_tgt.append(negative.ravel()[keep])
        if (i + 1) % 25 == 0 or i + 1 == len(bounds) - 1:
            log(
                f"Negative draws: {i + 1}/{len(bounds) - 1} MSAs; {sum(map(len, chunks_src)):,} candidates"
            )
    return np.concatenate(chunks_src), np.concatenate(chunks_tgt)


def audit_splits(pairs, n_nodes, node_counts, names):
    rows = []
    for msa, frame in pairs.groupby("msa_geoid", sort=True):
        src, tgt = frame.src_idx.to_numpy(np.int64), frame.tgt_idx.to_numpy(np.int64)
        keys = np.minimum(src, tgt) * np.int64(n_nodes) + np.maximum(src, tgt)
        labels = frame["split"].to_numpy()
        check = pd.DataFrame({"pair": keys, "split": labels})
        grouped = check.groupby("pair").split
        unique_count = grouped.nunique()
        assert unique_count.eq(1).all()
        directed = pd.Index(src * np.int64(n_nodes) + tgt)
        assert directed.is_unique
        reverse_idx = directed.get_indexer(tgt * np.int64(n_nodes) + src)
        has_reverse = reverse_idx >= 0
        reverse_train = has_reverse & (labels[np.maximum(reverse_idx, 0)] == "train")
        positive = frame.nij.to_numpy() > 0
        reverse_positive_train = reverse_train & positive[np.maximum(reverse_idx, 0)]
        row = {
            "GEOID": msa,
            "NAME": names[msa],
            "n_nodes": int(node_counts[msa]),
            "n_positive_edges": int(positive.sum()),
            "n_negative_edges": int((~positive).sum()),
            "n_candidate_edges": len(frame),
            "n_unordered_pairs": len(unique_count),
            "n_reciprocal_unordered_pairs": int((grouped.size() == 2).sum()),
            "cross_split_unordered_pairs": int((unique_count > 1).sum()),
        }
        expected = {
            "train": int(SPLIT[0] * len(unique_count)),
            "val": int(SPLIT[1] * len(unique_count)),
        }
        expected["test"] = len(unique_count) - expected["train"] - expected["val"]
        for label, count in expected.items():
            mask = labels == label
            assert len(np.unique(keys[mask])) == count
            row[label + "_rows"] = int(mask.sum())
            row[label + "_pairs"] = count
            row[label + "_positive_rows"] = int((mask & positive).sum())
            row[label + "_reverse_in_train_rows"] = int((mask & reverse_train).sum())
            row[label + "_reverse_positive_in_train_rows"] = int(
                (mask & reverse_positive_train).sum()
            )
            row[label + "_positive_reverse_positive_in_train_rows"] = int(
                (mask & positive & reverse_positive_train).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    build, artifacts = root / ".work/build", root / ".work"
    build.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    (build / "meta.json").unlink(missing_ok=True)
    started = time.time()
    reference = GEOGRAPHY
    poi_path = POI
    edge_path = EDGES
    log("Loading cleaned network and exact full-ID POI matches")
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='32GB'")
    con.execute(
        "CREATE TABLE observed AS SELECT * FROM read_csv_auto(?)", [str(edge_path)]
    )
    con.execute(
        "CREATE TABLE active AS SELECT SOURCE AS node FROM observed UNION SELECT TARGET AS node FROM observed"
    )
    poi = con.execute(
        """SELECT p.PLACEKEY,p.LATITUDE AS lat,p.LONGITUDE AS lon,p.WKT_AREA_SQ_METERS AS area_sqm,
                        p.TOP_CATEGORY_2DIGIT,p.NAICS_CODE FROM read_parquet(?, file_row_number=true) p
                        JOIN active a ON p.PLACEKEY=a.node ORDER BY p.file_row_number""",
        [str(poi_path)],
    ).fetchdf()
    n_nodes = con.execute("SELECT COUNT(*) FROM active").fetchone()[0]
    assert len(poi) == n_nodes and poi.PLACEKEY.is_unique
    assert np.isfinite(poi[["lat", "lon"]].to_numpy()).all()
    poi["idx"] = np.arange(n_nodes, dtype=np.int64)
    poi["log_area"] = np.log1p(poi.area_sqm.fillna(0).astype(np.float32))
    assert np.isfinite(poi.log_area).all()
    con.register("node_ids", poi[["PLACEKEY", "idx"]])
    observed = con.execute(
        """SELECT s.idx AS src_idx,t.idx AS tgt_idx,e.NIJ AS nij
                             FROM observed e JOIN node_ids s ON e.SOURCE=s.PLACEKEY
                             JOIN node_ids t ON e.TARGET=t.PLACEKEY ORDER BY e.rowid"""
    ).fetchdf()
    assert len(observed) == con.execute("SELECT COUNT(*) FROM observed").fetchone()[0]
    con.close()
    assert not observed.duplicated(["src_idx", "tgt_idx"]).any()
    assert (observed.src_idx != observed.tgt_idx).all() and (observed.nij > 50).all()
    poi, msa_names, n_counties, n_bg_states = spatial_covariates(poi, reference)
    cat_base = (
        poi.TOP_CATEGORY_2DIGIT.fillna("UNKNOWN")
        .str.split(r"\s*\|\s*", n=1, regex=True)
        .str[0]
    )
    poi["top_cat_id"], category_labels = pd.factorize(cat_base)
    poi["top_cat_id"] = poi.top_cat_id.astype(np.int32)
    acs_path = ACS
    acs = pd.read_csv(acs_path, dtype={"GEOID": str})[["GEOID"] + ACS_COLS].rename(
        columns={"GEOID": "cbg"}
    )
    assert acs.cbg.is_unique
    acs[ACS_COLS] = acs[ACS_COLS].apply(pd.to_numeric, errors="coerce")
    acs_fill = acs[ACS_COLS].median(numeric_only=True)
    acs[ACS_COLS] = acs[ACS_COLS].fillna(acs_fill)
    means, stds = acs[ACS_COLS].mean(), acs[ACS_COLS].std().replace(0, 1.0)
    acs[ACS_COLS] = (acs[ACS_COLS] - means) / stds
    poi = (
        poi.merge(acs, on="cbg", how="left", validate="many_to_one")
        .sort_values("idx")
        .reset_index(drop=True)
    )
    poi[ACS_COLS] = poi[ACS_COLS].fillna(0.0).astype(np.float32)
    assert (
        np.array_equal(poi.idx, np.arange(n_nodes))
        and np.isfinite(poi[ACS_COLS]).all().all()
    )
    src_all, tgt_all, nij_all = (
        observed.src_idx.to_numpy(np.int64),
        observed.tgt_idx.to_numpy(np.int64),
        observed.nij.to_numpy(float),
    )
    poi["ni_out"] = np.bincount(src_all, weights=nij_all, minlength=n_nodes)
    poi["ni_in"] = np.bincount(tgt_all, weights=nij_all, minlength=n_nodes)
    assert np.isclose(poi.ni_out.sum(), nij_all.sum()) and np.isclose(
        poi.ni_in.sum(), nij_all.sum()
    )
    msa = poi.msa_geoid.to_numpy(str)
    lat, lon = poi.lat.to_numpy(float), poi.lon.to_numpy(float)
    intra = msa[src_all] == msa[tgt_all]
    src_pos, tgt_pos, nij_pos = src_all[intra], tgt_all[intra], nij_all[intra]
    positive_keys = np.sort(src_all * np.int64(n_nodes) + tgt_all)
    distances = haversine_km(lat[src_pos], lon[src_pos], lat[tgt_pos], lon[tgt_pos])
    bins = np.clip(np.digitize(distances, DIST_BINS_KM) - 1, 0, len(DIST_BINS_KM) - 2)
    log(
        f"{n_nodes:,} active nodes; {len(src_pos):,} within-MSA positives; {(~intra).sum():,} cross-MSA positives"
    )
    negative_src, negative_tgt = sample_negatives(
        msa,
        lat,
        lon,
        src_pos,
        tgt_pos,
        DIST_BINS_KM[bins],
        DIST_BINS_KM[bins + 1],
        positive_keys,
        n_nodes,
        np.random.default_rng(SEED),
    )
    negative = (
        pd.DataFrame({"src_idx": negative_src, "tgt_idx": negative_tgt})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    negative["nij"], negative["neg_kind"] = 0.0, "distbin"
    pairs = pd.concat(
        [
            pd.DataFrame(
                {
                    "src_idx": src_pos,
                    "tgt_idx": tgt_pos,
                    "nij": nij_pos,
                    "neg_kind": "pos",
                }
            ),
            negative,
        ],
        ignore_index=True,
    )
    del negative_src, negative_tgt, negative, observed
    s, t = pairs.src_idx.to_numpy(np.int64), pairs.tgt_idx.to_numpy(np.int64)
    pairs["msa_geoid"] = msa[s]
    assert (msa[s] == msa[t]).all() and (s != t).all()
    assert not pairs.duplicated(["src_idx", "tgt_idx"]).any()
    pairs["log1p_nij"] = np.log1p(pairs.nij).astype(np.float32)
    pairs["log_distance_km"] = np.log1p(
        haversine_km(lat[s], lon[s], lat[t], lon[t])
    ).astype(np.float32)
    for output, column in [
        ("same_cbg", "cbg"),
        ("same_county", "county_id"),
        ("same_top_cat", "top_cat_id"),
    ]:
        values = poi[column].to_numpy()
        pairs[output] = (values[s] == values[t]).astype(np.int8)
    log(
        f"{len(pairs):,} candidate rows; splitting unordered POI pairs with seed0"
    )
    pairs["split"], pair_counts = split_pairs_per_msa(
        pairs, n_nodes, np.random.default_rng(SEED)
    )
    node_counts = poi.groupby("msa_geoid").size().to_dict()
    names = msa_names.set_index("GEOID").NAME.to_dict()
    audit = audit_splits(pairs, n_nodes, node_counts, names)
    assert len(audit) == 376
    keep = [
        "PLACEKEY",
        "idx",
        "lat",
        "lon",
        "msa_geoid",
        "cbg",
        "county_fips",
        "county_id",
        "top_cat_id",
        "log_area",
        "ni_out",
        "ni_in",
        "TOP_CATEGORY_2DIGIT",
        "NAICS_CODE",
    ] + ACS_COLS
    poi[keep].to_parquet(build / "nodes.parquet", index=False)
    pairs = pairs[
        [
            "src_idx",
            "tgt_idx",
            "msa_geoid",
            "nij",
            "log1p_nij",
            "log_distance_km",
            "same_cbg",
            "same_county",
            "same_top_cat",
            "neg_kind",
            "split",
        ]
    ]
    pairs.to_parquet(build / "edges.parquet", index=False)
    audit.to_csv(artifacts / "network_msa_summary.csv", index=False)
    audit[["GEOID", "NAME", "n_nodes", "n_positive_edges"]].to_csv(
        build / "msa_summary.csv", index=False
    )
    train_y = pairs.loc[pairs["split"].eq("train"), "log1p_nij"]
    meta = {
        "n_nodes": n_nodes,
        "n_msas_with_edges": len(audit),
        "n_pairs_total": len(pairs),
        "n_positive": len(src_pos),
        "n_zero": int(pairs.nij.eq(0).sum()),
        "n_observed_edges": len(nij_all),
        "n_cross_msa_edges": int((~intra).sum()),
        "pos_rate": len(src_pos) / len(pairs),
        "n_counties": n_counties,
        "n_top_cats": len(category_labels),
        "bge_dim": 0,
        "acs_cols": ACS_COLS,
        "neg_per_pos": K_NEG,
        "neg_strategy": "per_msa_distance_bin_knn",
        "acs_preprocessing": {
            "feature_order": ACS_COLS,
            "missing_raw_value_fill": acs_fill.to_dict(),
            "mean": means.to_dict(),
            "std_ddof_1": stds.to_dict(),
            "unmatched_cbg_standardized_value": 0.0,
        },
        "knn_k": KNN_K,
        "dist_bins_km": [float(x) if np.isfinite(x) else None for x in DIST_BINS_KM],
        "split_seed": SEED,
        "split_strategy": "per_msa_unordered_pair",
        "split_ratios": dict(zip(("train", "val", "test"), SPLIT)),
        "split_row_counts": pair_counts,
        "split_counts": pairs["split"].value_counts().to_dict(),
        "cross_split_unordered_pairs": int(audit.cross_split_unordered_pairs.sum()),
        "split_rng": "Fresh default_rng(0), independent of negative-sampling RNG",
        "neg_kind_counts": pairs.neg_kind.value_counts().to_dict(),
        "log1p_nij_train_mean": float(train_y.mean()),
        "log1p_nij_train_std": float(train_y.std()),
        "n_unmatched_cbg_nodes": int(poi.county_fips.eq("UNKNOWN").sum()),
        "n_bg_state_files": n_bg_states,
        "node_order": "Original row order of cleaned POI file; complete PLACEKEY identifiers",
        "feature_note": "ni_out/ni_in sum NIJ over all cleaned positives including test edges; original global preprocessing retained",
        "category_note": "First retained TOP_CATEGORY_2DIGIT token for same_top_cat; name/NAICS/text embeddings not GNN inputs",
        "static_data_note": "ACS/TIGER inputs supplied externally; CBG joins and features calculated from source data",
        "acs_csv": str(acs_path),
        "geography_dir": str(reference),
        "inputs": {
            str(poi_path): sha256(poi_path),
            str(edge_path): sha256(edge_path),
            str(acs_path): sha256(acs_path),
        },
        "build_seconds": time.time() - started,
    }
    pd.DataFrame(
        [
            {
                "n_nodes": n_nodes,
                "n_msa": len(audit),
                "n_observed_edges": len(nij_all),
                "n_intra_msa_positive_edges": len(src_pos),
                "n_cross_msa_edges": int((~intra).sum()),
                "n_sampled_negative_edges": meta["n_zero"],
                "n_candidate_edges": len(pairs),
                "cross_split_unordered_pairs": int(
                    audit.cross_split_unordered_pairs.sum()
                ),
                "n_unmatched_cbg_nodes": meta["n_unmatched_cbg_nodes"],
            }
        ]
    ).to_csv(artifacts / "network_build_summary.csv", index=False)
    (build / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(
        f"Build complete: {len(audit)} MSAs, {n_nodes:,} nodes, {len(pairs):,} candidate edges; unordered-pair split checks passed"
    )


if __name__ == "__main__":
    main()
