"""Leontief shocks using original WIJ, with an explicit stability prerequisite.

W[source,target] is the original NIJ / NI_SOURCE. Neither W nor beta is
renormalized to obtain stability. Non-subcritical cities have no reported
convergent Leontief cascade or amplification ratio.
"""

from pathlib import Path
import json
import time
import zlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .spectral_checks import assess_criticality
from .shock_paths import ROOT, DATA_DIR

WEIGHT_DEFINITION = (
    "Original WIJ = NIJ / NI_SOURCE; no row normalization or spectral rescaling"
)
NON_CONUS = {"02", "15", "60", "66", "69", "72", "78"}
CONTROL_METRICS = (
    "amplification_ratio_unw",
    "amplification_ratio",
    "cascade_loss_share",
)


def _activity(primary, fallback):
    values = np.asarray(primary, float)
    values = np.where(np.isfinite(values), values, np.asarray(fallback, float))
    return np.clip(np.where(np.isfinite(values), values, 0.0), 0.0, None)


class NonSubcriticalError(ValueError):
    """The unmodified matrix has no verified convergent cascade at this beta."""

    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__(
            f"Original WIJ: beta*rho(W)={metadata['beta_rho']:.8g}; status={metadata['criticality_status']}. No rescaling applied."
        )


def build_weights(nodes, edges):
    """Return the original directed WIJ matrix in the supplied node order."""
    assert len(nodes) and nodes.idx.is_unique
    assert {"wij", "ni_source"} <= set(
        edges
    ), "Input edges require original WIJ and visitation counts."
    index = pd.Index(nodes.idx)
    src = index.get_indexer(edges.src_idx)
    tgt = index.get_indexer(edges.tgt_idx)
    keep = (src >= 0) & (tgt >= 0)
    nij = edges.nij.to_numpy(float)[keep]
    visits = edges.ni_source.to_numpy(float)[keep]
    wij = edges.wij.to_numpy(float)[keep]
    assert np.isfinite(nij).all() and (nij > 0).all()
    assert np.isfinite(visits).all() and (visits > 0).all()
    assert np.isfinite(wij).all() and (wij >= 0).all() and (wij <= 1 + 1e-12).all()
    np.testing.assert_allclose(wij, nij / visits, rtol=1e-12, atol=1e-14)
    assert not edges.loc[keep].duplicated(["src_idx", "tgt_idx"]).any()
    W = sp.coo_matrix(
        (wij, (src[keep], tgt[keep])), shape=(len(nodes), len(nodes))
    ).tocsr()
    W.eliminate_zeros()
    return (W, src[keep], tgt[keep], nij, int((~keep).sum()))


def build_cascade(nodes, edges, beta):
    """Preserve node order; return W, A, solve/solve_T, local edges and metadata."""
    local = nodes.reset_index(drop=True).copy()
    assert len(local) and local.idx.is_unique
    local["local"] = np.arange(len(local), dtype=np.int64)
    n = len(local)
    visits = local.ni_source.to_numpy(float)
    assert np.isfinite(visits).all() and (visits > 0).all()
    W, src, tgt, nij, dropped = build_weights(local, edges)
    criticality = assess_criticality(W, beta)
    metadata = dict(
        n_pois=n,
        n_edges=int(W.nnz),
        total_visits=float(visits.sum()),
        beta=float(beta),
        orientation="W",
        rescaled=False,
        dropped_outside_edges=dropped,
        weight_definition=WEIGHT_DEFINITION,
        **criticality,
    )
    if not criticality["subcritical"]:
        raise NonSubcriticalError(metadata)
    A = (beta * W).tocsr()
    lu = sla.splu(sp.eye(n, format="csc") - A.tocsc()) if A.nnz else None

    def solve(f):
        f = np.asarray(f, float)
        return f.copy() if lu is None else lu.solve(f)

    def solve_T(b):
        b = np.asarray(b, float)
        return b.copy() if lu is None else lu.solve(b, trans="T")

    return dict(
        nodes=local,
        n=n,
        visits=visits,
        W=W,
        A=A,
        solve=solve,
        solve_T=solve_T,
        src_local=src,
        tgt_local=tgt,
        nij=nij,
        metadata=metadata,
    )


def run_shock(cascade, selected, shock_value=-1.0):
    """Return (original core shock metrics, full node-response vector)."""
    n, visits = (cascade["n"], cascade["visits"])
    selected = np.asarray(selected, dtype=np.int64)
    assert len(np.unique(selected)) == len(selected)
    assert ((selected >= 0) & (selected < n)).all() and shock_value < 0
    f = np.zeros(n)
    f[selected] = shock_value
    v = cascade["solve"](f)
    assert np.isfinite(v).all() and v.max(initial=0.0) < 1e-08
    absolute = np.abs(v)
    direct_u = float(np.abs(f).sum())
    direct_w = float(np.abs(f) @ visits)
    cascade_w = float(absolute @ visits)
    total = float(visits.sum())
    mask = np.ones(n, dtype=bool)
    mask[selected] = False
    survivor = absolute[mask]
    if len(survivor) and survivor.sum() > 0:
        hhi = float(np.sum((survivor / survivor.sum()) ** 2))
        affected = 1 / hhi / len(survivor)
        reach = float(np.mean(survivor >= 0.01))
    else:
        hhi, affected, reach = (np.nan, np.nan, 0.0)
    share = lambda value: float(value / total) if total > 0 else np.nan
    metrics = dict(
        direct_loss_share=share(direct_w),
        cascade_loss_share=share(cascade_w),
        capped_loss_share=share(np.minimum(absolute, 1.0) @ visits),
        indirect_loss_share=share(np.abs(v - f) @ visits),
        amplification_ratio=cascade_w / direct_w if direct_w > 0 else np.nan,
        amplification_ratio_unw=(
            float(absolute.sum() / direct_u) if direct_u > 0 else np.nan
        ),
        loss_hhi=hhi,
        eff_affected_frac=affected,
        reach_frac=reach,
    )
    return (metrics, v)


def seed_rng(geoid, hazard, draw):
    key = zlib.crc32(f"{geoid}|rand_{hazard}|{draw}".encode()) & 2147483647
    return np.random.default_rng(key)


def select_tracts(local, scores, hazard, fraction=0.1):
    """Use the historical value_counts order and stable ties, including round()."""
    tracts = local.tract.astype(str).to_numpy()
    counts = pd.Series(tracts).value_counts()
    risk = scores[hazard].reindex(counts.index).astype(float)
    risk = risk.where(np.isfinite(risk), 0.0)
    target = max(1, int(round(fraction * len(counts))))
    chosen = risk[risk > 0].sort_values(ascending=False, kind="stable").index[:target]
    selected = np.flatnonzero(np.isin(tracts, chosen))
    return (selected, chosen, risk, counts, target)


def screen_msas(nodes, scores, tract_attributes, names, hazards, threshold):
    """POI-majority MSA first; polygon representative-point fallback second."""
    by_tract = nodes.groupby("tract", sort=True).agg(
        n_pois=("idx", "size"), msa_poi=("msa_geoid", lambda x: x.mode().iat[0])
    )
    rows = []
    attributes = tract_attributes.set_index("TRACTFIPS")
    assert attributes.index.is_unique and scores.index.is_unique
    for hazard in hazards:
        top = scores.loc[scores[hazard].ge(threshold), [hazard]].rename(
            columns={hazard: "risk_percentile"}
        )
        top = top.loc[~top.index.str[:2].isin(NON_CONUS)].copy()
        assert set(top.index) <= set(
            attributes.index
        ), "A national top-risk tract has no supplied geometry/assignment"
        top = top.join(by_tract).join(attributes[["STATEABBRV", "msa_spatial"]])
        top["msa_geoid"] = top.msa_poi.fillna(top.msa_spatial)
        top.loc[~top.msa_geoid.isin(names), "msa_geoid"] = np.nan
        top["assignment"] = np.where(
            top.msa_poi.notna(),
            "poi",
            np.where(top.msa_geoid.notna(), "spatial", "outside"),
        )
        top["hazard"] = hazard
        top["msa_name"] = top.msa_geoid.map(names)
        top["n_pois"] = top.n_pois.fillna(0).astype(int)
        top["poi_spatial_disagree"] = (
            top.msa_poi.notna()
            & top.msa_spatial.notna()
            & top.msa_poi.ne(top.msa_spatial)
        )
        rows.append(top.reset_index())
    return pd.concat(rows, ignore_index=True)


def simulate(root=ROOT):
    root = Path(root).resolve()
    data, out = (DATA_DIR, root / "results")
    out.mkdir(exist_ok=True)
    config = json.loads((root / "config.json").read_text())
    assert config["orientation"] == "W" and config["shock_value"] < 0
    hazards = config["hazards"]
    nodes = pd.read_parquet(data / "nodes.parquet")
    nodes["msa_geoid"] = nodes.msa_geoid.astype(str).str.zfill(5)
    nodes["tract"] = nodes.tract.astype(str).str.zfill(11)
    edges = pd.read_parquet(data / "edges.parquet")
    edges["msa_geoid"] = edges.msa_geoid.astype(str).str.zfill(5)
    boundaries = pd.read_parquet(
        data / "msa_boundaries.parquet", columns=["GEOID", "NAME"]
    )
    names = dict(zip(boundaries.GEOID.astype(str).str.zfill(5), boundaries.NAME))
    assert len(names) == config["base_msa_count"] and nodes.idx.is_unique
    assert set(nodes.msa_geoid) == set(names)
    scores = pd.read_parquet(data / "hazard_scores.parquet")
    scores["TRACTFIPS"] = scores.TRACTFIPS.astype(str).str.zfill(11)
    scores = scores.set_index("TRACTFIPS")
    attrs = pd.read_parquet(
        data / "tracts.parquet", columns=["TRACTFIPS", "STATEABBRV", "msa_spatial"]
    )
    attrs["TRACTFIPS"] = attrs.TRACTFIPS.astype(str).str.zfill(11)
    attrs["msa_spatial"] = attrs.msa_spatial.map(
        lambda v: str(v).zfill(5) if pd.notna(v) else np.nan
    )
    screened = screen_msas(
        nodes, scores, attrs, names, hazards, config["national_percentile_threshold"]
    )
    cases = (
        screened.dropna(subset=["msa_geoid"]).groupby(["msa_geoid", "hazard"]).size()
    )
    ids = sorted(set(cases.index.get_level_values("msa_geoid")))
    node_groups = nodes.groupby("msa_geoid", sort=False).indices
    edge_groups = edges.groupby("msa_geoid", sort=False).indices
    rows, controls = ([], [])
    started = time.time()
    for count, geoid in enumerate(ids, 1):
        local = nodes.iloc[node_groups[geoid]]
        local_edges = edges.iloc[edge_groups.get(geoid, np.array([], dtype=int))]
        try:
            cascade = build_cascade(local, local_edges, config["beta"])
            meta = cascade["metadata"]
        except NonSubcriticalError as exc:
            cascade, meta = (None, exc.metadata)
        assert meta["dropped_outside_edges"] == 0
        local = local.reset_index(drop=True)
        n, visits = (len(local), local.ni_source.to_numpy(float))
        if cascade is not None:
            influence = cascade["solve_T"](np.ones(n))
            weighted = cascade["solve_T"](visits)
            assert np.isfinite(influence).all() and (influence >= 1 - 1e-09).all()
            indeg = np.asarray(cascade["A"].sum(axis=0)).ravel()
        xy = np.column_stack(
            [
                np.radians(local.lon) * 6371.0 * np.cos(np.radians(local.lat.mean())),
                np.radians(local.lat) * 6371.0,
            ]
        )
        for hazard in hazards:
            if (geoid, hazard) not in cases.index:
                continue
            selected, chosen, risk, tract_counts, target = select_tracts(
                local, scores, hazard, config["within_msa_tract_fraction"]
            )
            k = len(selected)
            row = dict(
                msa_geoid=geoid,
                msa_name=names[geoid],
                hazard=hazard,
                hazard_name=hazards[hazard],
                n_pois=n,
                n_tracts=len(tract_counts),
                n_tracts_national_top1=int(cases.loc[geoid, hazard]),
                k_tracts_target=target,
                k_tracts_removed=len(chosen),
                k_short=int(len(chosen) < target),
                k_pois_removed=k,
                frac_pois_removed=k / n,
                n_rand=config["random_draws"] if cascade is not None else 0,
                beta=config["beta"],
                beta_rho=meta["beta_rho"],
                criticality_status=meta["criticality_status"],
                min_s_removed=float(risk[chosen].min()) if len(chosen) else np.nan,
                n_removed_in_national_top1=int(
                    risk[chosen].ge(config["national_percentile_threshold"]).sum()
                ),
            )
            if k == 0 or k >= n:
                row["n_rand"] = 0
                rows.append(dict(**row, status="undefined_removal_budget"))
                continue
            if cascade is None:
                rows.append(dict(**row, status="non_subcritical"))
                continue
            observed, response = run_shock(cascade, selected, config["shock_value"])
            row.update(observed)
            f = np.zeros(n)
            f[selected] = config["shock_value"]
            residual = float(np.max(np.abs(response - cascade["A"] @ response - f)))
            assert residual < 1e-08
            assert np.isclose(
                observed["amplification_ratio_unw"],
                influence[selected].mean(),
                rtol=1e-10,
                atol=1e-10,
            )
            row.update(
                removed_indeg_ratio=(
                    float(indeg[selected].mean() / indeg.mean())
                    if indeg.mean() > 0
                    else np.nan
                ),
                removed_visits_share=(
                    float(visits[selected].sum() / visits.sum())
                    if visits.sum() > 0
                    else np.nan
                ),
                removed_radius_km=float(
                    np.sqrt(((xy[selected] - xy[selected].mean(0)) ** 2).sum(1)).max()
                ),
            )
            per_case = []
            for draw in range(config["random_draws"]):
                sel = seed_rng(geoid, hazard, draw).choice(n, k, replace=False)
                direct = visits[sel].sum()
                values = dict(
                    amplification_ratio_unw=float(influence[sel].mean()),
                    amplification_ratio=(
                        float(weighted[sel].sum() / direct) if direct > 0 else np.nan
                    ),
                    cascade_loss_share=(
                        float(
                            abs(config["shock_value"])
                            * weighted[sel].sum()
                            / visits.sum()
                        )
                        if visits.sum() > 0
                        else np.nan
                    ),
                )
                rec = dict(
                    msa_geoid=geoid,
                    msa_name=names[geoid],
                    hazard=hazard,
                    seed=draw,
                    k_pois_removed=k,
                    **values,
                )
                controls.append(rec)
                per_case.append(rec)
            draw_frame = pd.DataFrame(per_case)
            for metric in CONTROL_METRICS:
                row[metric + "_rand"] = float(draw_frame[metric].mean())
                row[metric + "_rand_sd"] = float(draw_frame[metric].std(ddof=1))
            row["ar_premium_vs_rand"] = (
                row["amplification_ratio_unw"] / row["amplification_ratio_unw_rand"]
            )
            row["ar_w_premium_vs_rand"] = (
                row["amplification_ratio"] / row["amplification_ratio_rand"]
            )
            row["above_random_mean"] = row["ar_premium_vs_rand"] > 1
            row["exact_random_ar_mean"] = float(influence.mean())
            row["exact_random_ar_sd"] = float(
                np.sqrt((1 - k / n) * np.var(influence, ddof=1) / k)
            )
            row["random_ar_mean_relative_error_pct"] = 100 * (
                row["amplification_ratio_unw_rand"] / row["exact_random_ar_mean"] - 1
            )
            rows.append(dict(**row, status="ok"))
        if count == 1 or count % 10 == 0 or count == len(ids):
            print(
                f"{count}/{len(ids)} screened MSAs; {len(rows)} hazard cases; {len(controls)} controls; {time.time() - started:.1f}s",
                flush=True,
            )
    result = pd.DataFrame(rows)
    draws = pd.DataFrame(
        controls,
        columns=[
            "msa_geoid",
            "msa_name",
            "hazard",
            "seed",
            "k_pois_removed",
            *CONTROL_METRICS,
        ],
    )
    for field in [
        "direct_loss_share",
        "cascade_loss_share",
        "capped_loss_share",
        "indirect_loss_share",
        "amplification_ratio",
        "amplification_ratio_unw",
        "loss_hhi",
        "eff_affected_frac",
        "reach_frac",
        "ar_premium_vs_rand",
        "ar_w_premium_vs_rand",
        "above_random_mean",
        "removed_indeg_ratio",
        "removed_visits_share",
        "removed_radius_km",
        "exact_random_ar_mean",
        "exact_random_ar_sd",
        "random_ar_mean_relative_error_pct",
        *[
            metric + suffix
            for metric in CONTROL_METRICS
            for suffix in ["_rand", "_rand_sd"]
        ],
    ]:
        if field not in result:
            result[field] = np.nan
    assert not result.duplicated(["msa_geoid", "hazard"]).any()
    valid = result.status.eq("ok")
    assert len(draws) == int(valid.sum()) * config["random_draws"]
    assert (
        draws.groupby(["msa_geoid", "hazard"]).size().eq(config["random_draws"]).all()
    )
    assert (result.loc[valid, "amplification_ratio_unw"] >= 1 - 1e-09).all()
    result.to_csv(out / "shock_results.csv", index=False)
    return result
