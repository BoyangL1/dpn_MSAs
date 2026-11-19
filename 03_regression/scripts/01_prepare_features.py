"""Compute metropolitan socioeconomic, network, and urban-form predictors."""

from regression_lib.analysis_paths import (
    ROOT,
    MAIN,
    INPUTS,
    RESULTS,
    FIGURES,
    ACS,
    POI,
    EDGES,
    GEOGRAPHY,
)
import json, argparse, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import duckdb
import geopandas as gpd
import pyarrow.parquet as pq
import scipy.sparse as sp
from pyproj import Transformer

from lib import build_network as b

W = ROOT / ".work"
I = INPUTS
R = GEOGRAPHY
SCENARIOS = {
    "scn_Grocery___convenience_stores": "445",
    "scn_Gas_stations": "447",
    "scn_Transit__bus___taxi_services": "485",
    "scn_Shopping_malls___centers": "531120",
    "scn_Schools___colleges": "611",
    "scn_Parks___nature_areas": "712",
    "scn_Gyms___recreation_centers": "713",
    "scn_Hotels___motels": "721",
    "scn_Restaurants___bars": "722",
}


def dump(x, p):
    p.write_text(
        json.dumps(
            x, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        )
        + "\n"
    )


def ids():
    return pd.read_csv(
        I / "node_based_msa_features.csv", dtype={"msa_geoid": str}
    ).msa_geoid.tolist()


def demographic():
    nd = pd.read_parquet(MAIN / ".work/build/nodes.parquet")

    agg = nd.groupby("msa_geoid")
    out = agg[b.ACS_COLS].mean().add_prefix("acs_")
    out["n_nodes"] = agg.size()
    code = nd.NAICS_CODE.fillna("").astype(str).str.split("|").str[0].str.strip()
    for c, prefix in SCENARIOS.items():
        out[c] = code.str.startswith(prefix).groupby(nd.msa_geoid).mean()
    out.to_csv(I / "node_based_msa_features.csv")
    valid = nd.county_fips.str.fullmatch(r"\d{5}", na=False)
    cross = (
        nd.loc[valid]
        .groupby("county_fips")
        .msa_geoid.agg(lambda s: s.value_counts().index[0])
    )
    cross.to_csv(I / "county_to_msa_crosswalk.csv")
    acs = pd.read_csv(ACS, dtype={"GEOID": str})
    acs["msa_geoid"] = acs.GEOID.str[:5].map(cross)
    acs = acs.dropna(subset=["msa_geoid"])
    wt = pd.to_numeric(acs.population, errors="coerce").clip(lower=0).fillna(0)
    raw = pd.DataFrame(index=out.index)
    for c in b.ACS_COLS:
        values = pd.to_numeric(acs[c], errors="coerce")
        good = values.notna() & wt.gt(0)
        raw[c] = (
            values.groupby(acs.msa_geoid).sum(min_count=1)
            if c in ["population", "households"]
            else (values[good] * wt[good]).groupby(acs.msa_geoid[good]).sum()
            / wt[good].groupby(acs.msa_geoid[good]).sum()
        )
    metro = gpd.read_file(b.CBSA).set_index("GEOID")
    raw["pop_density"] = raw.population / (pd.to_numeric(metro.ALAND) / 1e6)
    raw.to_csv(I / "demographic_raw_msa_features.csv")
    print(
        "Saved node ACS, scenario shares, and population-weighted heatmap ACS",
        flush=True,
    )


def urban():
    cache = W / "urban_chunks"
    cache.mkdir(exist_ok=True)
    metro = gpd.read_file(b.CBSA)
    metro = metro[metro.LSAD.eq("M1")].to_crs(4326)[["GEOID", "geometry"]]
    area = metro.to_crs(5070).set_index("GEOID").area / 1e6
    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    pp = POI
    con = duckdb.connect()
    total, unique = con.execute(
        "select count(*),count(distinct PLACEKEY) from read_parquet(?)", [str(pp)]
    ).fetchone()
    con.close()
    assert total == unique
    for j, batch in enumerate(
        pq.ParquetFile(pp).iter_batches(
            batch_size=250000, columns=["LATITUDE", "LONGITUDE", "TOP_CATEGORY_2DIGIT"]
        )
    ):
        done = cache / f"{j:03d}.json"
        if done.exists():
            continue
        d = batch.to_pandas()
        d = d.dropna(subset=["LATITUDE", "LONGITUDE"])
        d["idx"] = np.arange(len(d))
        pt = gpd.GeoDataFrame(
            d, geometry=gpd.points_from_xy(d.LONGITUDE, d.LATITUDE), crs=4326
        )
        q = (
            gpd.sjoin(pt, metro, how="left", predicate="within")
            .drop_duplicates("idx")
            .dropna(subset=["GEOID"])
        )
        x, y = tr.transform(q.LONGITUDE.to_numpy(), q.LATITUDE.to_numpy())
        q["ix"] = np.floor(x / 1000).astype(int)
        q["iy"] = np.floor(y / 1000).astype(int)
        q["sector"] = (
            q.TOP_CATEGORY_2DIGIT.astype("string").str.split("|").str[0].str.strip()
        )
        q.groupby(["GEOID", "ix", "iy"]).size().rename("n").reset_index().to_parquet(
            cache / f"{j:03d}_grid.parquet", index=False
        )
        q.dropna(subset=["sector"]).groupby(["GEOID", "sector"]).size().rename(
            "n"
        ).reset_index().to_parquet(cache / f"{j:03d}_sector.parquet", index=False)
        dump({"input": len(d), "matched": len(q)}, done)
        print(f"Urban form batch {j+1}: {len(q):,} POIs", flush=True)
    grids = (
        pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("*_grid.parquet"))])
        .groupby(["GEOID", "ix", "iy"], as_index=False)
        .n.sum()
    )
    sectors = (
        pd.concat([pd.read_parquet(p) for p in sorted(cache.glob("*_sector.parquet"))])
        .groupby(["GEOID", "sector"], as_index=False)
        .n.sum()
    )
    grids.to_parquet(I / "poi_1km_grid_counts.parquet", index=False)
    sectors.to_csv(I / "all_poi_sector_counts.csv", index=False)
    from regression_lib.feature_metrics import metrics_for_msa as poly
    rows = []
    for g, grid in grids.groupby("GEOID"):
        p = poly(grid, int(grid.n.sum()), 0.01)
        s = sectors[sectors.GEOID.eq(g)].n
        frac = s / s.sum()
        rows.append(
            dict(
                msa_geoid=g,
                uf_poi_density=s.sum() / area[g],
                uf_poi_entropy=float(-(frac * np.log(frac)).sum()),
                uf_poly_K=p["n_centers"],
                uf_poly_D=p["largest_center_share"],
                **p,
            )
        )
    pd.DataFrame(rows).to_csv(I / "urban_form_raw_msa_features.csv", index=False)
    print("Urban form complete", flush=True)


_NODE = {}
_EDGE = {}


def worker(g):
    nd = _NODE[g]
    ed = _EDGE[g]
    n = len(nd)
    from regression_lib import feature_metrics as pr

    topo = pr.describe_graph
    gini = pr.gini
    idx = nd.idx.to_numpy()
    s = np.searchsorted(idx, ed.src_idx.to_numpy())
    t = np.searchsorted(idx, ed.tgt_idx.to_numpy())
    nij = ed.nij.to_numpy(float)
    assert np.array_equal(idx[s], ed.src_idx) and np.array_equal(idx[t], ed.tgt_idx)
    result = topo(
        ed.src_idx.to_numpy(), ed.tgt_idx.to_numpy(), nij, n, [("main", 0.25, 1)]
    )[0]
    denom = nd.ni_out.to_numpy()
    positive = denom > 0
    assert positive[s].all()
    w = sp.csr_matrix((nij / denom[s], (s, t)), shape=(n, n))
    incoming = np.asarray(w.sum(axis=0)).ravel()
    row = np.asarray(w.sum(axis=1)).ravel()
    p, it, resid = pr.pagerank(w)
    visits = nd.ni_in.to_numpy()
    k = max(1, int(round(0.01 * n)))
    top = np.argsort(-incoming, kind="stable")[:k]
    lat, lon = nd.lat.to_numpy(), nd.lon.to_numpy()
    a, bb = np.triu_indices(k, k=1)
    distance = (
        float(
            b.haversine_km(lat[top[a]], lon[top[a]], lat[top[bb]], lon[top[bb]]).mean()
        )
        if k > 1
        else 0.0
    )
    result.update(
        msa_geoid=g,
        net_pagerank_gini=gini(p),
        pagerank_residual=resid,
        pagerank_iterations=it,
        top1_share_visits=float(np.sort(visits)[-k:].sum() / visits.sum()),
        incoming_strength_gini=gini(incoming),
        edge_weight_gini=gini(nij),
        w_row_sum_mean=float(row.mean()),
        log_top1_pair_distance=float(np.log1p(distance)),
        top1_k=k,
        legacy_candidate_outgoing_gini=gini(row),
    )
    dump(result, W / "network_cache" / f"{g}.json")
    return g


def network(workers):
    cache = W / "network_cache"
    cache.mkdir(exist_ok=True)
    selected = ids()
    pending = [g for g in selected if not (cache / f"{g}.json").exists()]
    if pending:
        nd = pd.read_parquet(MAIN / ".work/build/nodes.parquet")
        ed = pd.read_parquet(
            MAIN / ".work/build/edges.parquet", filters=[("nij", ">", 0)]
        )
        _NODE.update({g: d for g, d in nd.groupby("msa_geoid") if g in pending})
        _EDGE.update({g: d for g, d in ed.groupby("msa_geoid") if g in pending})
        print(f"Computing network metrics for {len(pending)} MSAs", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("fork")
        ) as pool:
            futures = [pool.submit(worker, g) for g in pending]
            for j, f in enumerate(as_completed(futures), 1):
                f.result()
                if j % 10 == 0 or j == len(pending):
                    print(f"Network completed {j}/{len(pending)}", flush=True)
    result = pd.DataFrame(
        [json.loads((cache / f"{g}.json").read_text()) for g in selected]
    )

    result = result.rename(
        columns={"incoming_strength_gini": "diagnostic_reconstructed_W_column_gini"}
    )
    strength_path = I / "raw_w_strength_features.csv"
    if strength_path.exists():
        result = result.merge(
            pd.read_csv(strength_path, dtype={"msa_geoid": str}),
            on="msa_geoid",
            validate="one_to_one",
        )
    result.to_csv(I / "network_msa_features.csv", index=False)


def strength():
    from regression_lib.feature_metrics import gini

    nd = pd.read_parquet(
        MAIN / ".work/build/nodes.parquet", columns=["PLACEKEY", "idx", "msa_geoid"]
    )
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.register("nd", nd)
    ep = EDGES

    con.execute(
        "CREATE TABLE observed AS SELECT SOURCE,TARGET,WIJ FROM read_csv_auto(?)",
        [str(ep)],
    )
    con.execute(
        "CREATE TABLE source_joined AS SELECT s.idx AS src,s.msa_geoid,e.TARGET,e.WIJ FROM observed e JOIN nd s ON e.SOURCE=s.PLACEKEY"
    )
    print("Raw WIJ source join complete", flush=True)
    con.execute(
        "CREATE TABLE mapped AS SELECT e.src,t.idx AS tgt,e.msa_geoid,t.msa_geoid AS target_msa,e.WIJ AS wij FROM source_joined e JOIN nd t ON e.TARGET=t.PLACEKEY"
    )
    con.execute(
        "CREATE TABLE within_edges AS SELECT src,tgt,msa_geoid,wij FROM mapped WHERE msa_geoid=target_msa"
    )
    values = con.execute(
        """WITH active AS (SELECT msa_geoid,src AS idx FROM within_edges UNION SELECT msa_geoid,tgt AS idx FROM within_edges), outs AS (SELECT src,sum(wij) AS row_sum FROM within_edges GROUP BY src) SELECT a.msa_geoid,a.idx,coalesce(o.row_sum,0) AS row_sum FROM active a LEFT JOIN outs o ON a.idx=o.src ORDER BY a.idx"""
    ).fetchdf()
    con.close()
    results = []
    for g, q in values.groupby("msa_geoid"):
        results.append(
            dict(
                msa_geoid=g,
                incoming_strength_gini=gini(q.row_sum),
                raw_w_within_node_count=len(q),
                raw_w_row_sum_mean=float(q.row_sum.mean()),
            )
        )
    pd.DataFrame(results).to_csv(I / "raw_w_strength_features.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "demographic", "urban", "strength", "network"],
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for path in [W, I, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    stages = (
        ["demographic", "urban", "strength", "network"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        if stage == "network":
            network(args.workers)
        else:
            globals()[stage]()


if __name__ == "__main__":
    main()
