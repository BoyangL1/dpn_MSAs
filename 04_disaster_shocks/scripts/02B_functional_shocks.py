"""Simulate functional shocks and estimate cluster-by-sector interactions."""

from pathlib import Path
import argparse
import json
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shock_lib.propagation import NonSubcriticalError, build_cascade, run_shock
from shock_lib.scenarios import SCENARIOS
from shock_lib.shock_paths import ROOT, DATA_DIR

STEM = "heatmap_interaction_ar_flat_18cm"
GROUPS = list(range(1, 9))
RESPONSE_METRICS = [
    "direct_loss_share",
    "cascade_loss_share",
    "capped_loss_share",
    "indirect_loss_share",
    "amplification_ratio",
    "amplification_ratio_unw",
    "loss_hhi",
    "eff_affected_frac",
    "reach_frac",
]


def write_json(path, record):
    Path(path).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def simulate(root, members):
    out = root / "results/functional_shocks"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    config = json.loads((root / "config.json").read_text())
    beta = float(config["beta"])
    shock_value = float(config["shock_value"])
    assert config["orientation"] == "W" and shock_value < 0
    nodes = pd.read_parquet(DATA_DIR / "nodes.parquet")
    edges = pd.read_parquet(DATA_DIR / "edges.parquet")
    labels = pd.read_parquet(DATA_DIR / "functional_node_categories.parquet")
    assert labels.idx.is_unique and nodes.idx.is_unique
    nodes = nodes.merge(
        labels[["idx", "PLACEKEY", "naics_code"]],
        on=["idx", "PLACEKEY"],
        how="left",
        validate="one_to_one",
    )
    assert nodes.naics_code.notna().all()
    nodes = nodes.loc[nodes.msa_geoid.isin(members.geoid)].copy()
    edges = edges.loc[edges.msa_geoid.isin(members.geoid)].copy()
    ng = nodes.groupby("msa_geoid", sort=False).indices
    eg = edges.groupby("msa_geoid", sort=False).indices
    membership = members.set_index("geoid")
    rows = []
    max_residual, max_identity = (0.0, 0.0)
    for i, geoid in enumerate(sorted(members.geoid), 1):
        local = nodes.iloc[ng[geoid]]
        local_edges = edges.iloc[eg.get(geoid, np.array([], dtype=int))]
        assert len(local) == int(membership.loc[geoid, "n_nodes"])
        try:
            cascade = build_cascade(local, local_edges, beta=beta)
            city_metadata = cascade["metadata"]
        except NonSubcriticalError as error:
            cascade = None
            city_metadata = error.metadata
        assert city_metadata["dropped_outside_edges"] == 0
        n = len(local)
        node_codes = local.naics_code.to_numpy()
        if cascade is not None:
            influence = cascade["solve_T"](np.ones(n))
            assert np.isfinite(influence).all() and (influence >= 1 - 1e-09).all()
        identity = dict(
            msa_geoid=geoid,
            msa_name=membership.loc[geoid, "name"],
            cluster=int(membership.loc[geoid, "cluster"]),
        )
        for code, sector in SCENARIOS:
            selected = np.flatnonzero(node_codes == code)
            k = len(selected)
            eligible = 0 < k < n
            record = dict(
                **identity,
                naics_code=code,
                sector=sector,
                n_pois=n,
                k_removed=k,
                cat_share=k / n,
                eligible=eligible,
                criticality_status=city_metadata["criticality_status"],
                beta=beta,
                beta_rho=city_metadata["beta_rho"],
            )
            if not eligible or cascade is None:
                reason = (
                    ("absent_category" if k == 0 else "all_nodes")
                    if not eligible
                    else city_metadata["criticality_status"]
                )
                rows.append(
                    dict(
                        **record,
                        status="ineligible" if not eligible else "non_subcritical",
                        exclusion_reason=reason,
                        **{key: np.nan for key in RESPONSE_METRICS},
                    )
                )
                continue
            metrics, response = run_shock(cascade, selected, shock_value=shock_value)
            f = np.zeros(n)
            f[selected] = shock_value
            residual = float(np.max(np.abs(response - cascade["A"] @ response - f)))
            error = abs(
                metrics["amplification_ratio_unw"] - float(influence[selected].mean())
            )
            assert residual < 1e-08
            np.testing.assert_allclose(
                metrics["amplification_ratio_unw"],
                influence[selected].mean(),
                rtol=1e-10,
                atol=1e-10,
            )
            assert metrics["amplification_ratio_unw"] >= 1 - 1e-09
            max_residual = max(max_residual, residual)
            max_identity = max(max_identity, error)
            rows.append(dict(**record, status="ok", exclusion_reason="", **metrics))
        if i == 1 or i % 20 == 0 or i == len(members):
            print(
                f"{i}/{len(members)} MSAs; {len(rows)} functional shocks; {time.time() - start:.1f}s",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    assert not raw.duplicated(["msa_geoid", "naics_code"]).any()
    valid = raw.status.eq("ok")
    assert len(raw) == len(members) * len(SCENARIOS)
    assert np.isfinite(raw.loc[valid, "amplification_ratio_unw"]).all()
    assert raw.loc[valid, "amplification_ratio_unw"].ge(1 - 1e-09).all()
    assert raw.loc[~valid, RESPONSE_METRICS].isna().all().all()
    raw["log_ar"] = np.nan
    raw.loc[valid, "log_ar"] = np.log(raw.loc[valid, "amplification_ratio_unw"])
    raw.to_csv(out / "category_shock_raw.csv", index=False)
    return raw


def calculate_interaction(raw, members, out):
    order = [label for _, label in SCENARIOS]
    assert not raw.duplicated(["msa_geoid", "sector"]).any()
    assert set(raw.msa_geoid) == set(members.geoid)
    valid = (
        raw.status.eq("ok")
        & np.isfinite(raw.amplification_ratio_unw)
        & raw.amplification_ratio_unw.gt(0)
    )
    assert valid.equals(
        raw.status.eq("ok")
    ), "An apparently valid scenario has invalid AR"
    np.testing.assert_allclose(
        raw.loc[valid, "log_ar"],
        np.log(raw.loc[valid, "amplification_ratio_unw"]),
        atol=1e-14,
        rtol=0,
    )
    assert raw.loc[~valid, "log_ar"].isna().all()
    pivot = raw.loc[valid].pivot(index="msa_geoid", columns="sector", values="log_ar")
    pivot = pivot.reindex(index=members.geoid, columns=order)
    pivot.index.name = "msa_geoid"
    pivot = pivot.loc[np.isfinite(pivot).all(axis=1)]
    member_index = members.set_index("geoid")
    labels = member_index.loc[pivot.index, "cluster"].to_numpy()
    values = pivot.to_numpy(float)
    counts = np.array([np.sum(labels == c) for c in GROUPS], dtype=int)
    present = counts > 0
    estimable = len(pivot) > 0 and int(present.sum()) >= 2
    if len(pivot):
        residual = (
            values
            - values.mean(axis=1, keepdims=True)
            - values.mean(axis=0, keepdims=True)
            + values.mean()
        )
        np.testing.assert_allclose(residual.mean(axis=0), 0, atol=1e-13)
        np.testing.assert_allclose(residual.mean(axis=1), 0, atol=1e-13)
        membership = np.stack([labels == c for c in GROUPS], axis=1).astype(float)
        means = np.divide(
            membership.T @ residual,
            counts[:, None],
            out=np.full((len(GROUPS), len(order)), np.nan),
            where=counts[:, None] > 0,
        )
        grouped = (
            pivot.assign(cluster=labels)
            .groupby("cluster")
            .mean()
            .reindex(index=GROUPS, columns=order)
        )
        independent = (
            grouped.sub(grouped.mean(axis=1), axis=0).sub(pivot.mean(axis=0), axis=1)
            + values.mean()
        )
        np.testing.assert_allclose(
            means, independent, rtol=0, atol=1e-13, equal_nan=True
        )
        np.testing.assert_allclose(
            counts[present] @ means[present] / counts.sum(), 0, atol=1e-13
        )
        group_error = float(np.nanmax(np.abs(means - independent.to_numpy())))
        row_error = float(np.abs(residual.mean(axis=1)).max())
        column_error = float(np.abs(residual.mean(axis=0)).max())
    else:
        residual = np.empty((0, len(order)))
        means = np.full((len(GROUPS), len(order)), np.nan)
        group_error = row_error = column_error = None
    cells = pd.DataFrame(means, index=pd.Index(GROUPS, name="cluster"), columns=order)
    ledger = members[["geoid", "name", "cluster"]].rename(
        columns={"geoid": "msa_geoid", "name": "msa_name"}
    )
    ledger["n_finite_scenarios"] = (
        ledger.msa_geoid.map(raw.loc[valid].groupby("msa_geoid").size())
        .fillna(0)
        .astype(int)
    )
    ledger["complete_case"] = ledger.msa_geoid.isin(pivot.index)
    ledger["included_in_heatmap"] = ledger.complete_case & estimable
    available = raw.loc[valid].groupby("msa_geoid").sector.agg(set).to_dict()
    ledger["missing_scenarios"] = ledger.msa_geoid.map(
        lambda g: "; ".join((s for s in order if s not in available.get(g, set())))
    )
    failure_reasons = (
        raw.loc[~valid]
        .groupby("msa_geoid")
        .exclusion_reason.agg(lambda x: "; ".join(sorted(set(x.dropna()))))
    )
    ledger["exclusion_reason"] = ledger.msa_geoid.map(failure_reasons).fillna("")
    ledger["criticality_status"] = ledger.msa_geoid.map(
        raw.groupby("msa_geoid").criticality_status.first()
    )
    ledger.to_csv(out / "complete_case_membership.csv", index=False)
    seed, n_perm = (20260923, 5000 if estimable else 0)
    rng = np.random.default_rng(seed)
    observed = (
        float((counts[present, None] * means[present] ** 2).sum())
        if estimable
        else None
    )
    null = np.empty(n_perm)
    if estimable:
        for b in range(n_perm):
            perm = rng.permutation(labels)
            gp = np.stack(
                [perm == c for c, keep in zip(GROUPS, present) if keep], axis=1
            ).astype(float)
            mp = gp.T @ residual / counts[present, None]
            null[b] = (counts[present, None] * mp**2).sum()
    stats = dict(
        status="computed" if estimable else "not_estimable",
        cohort_msas=len(members),
        complete_case_msas=len(pivot),
        excluded_incomplete_msas=len(members) - len(pivot),
        non_subcritical_msas=int(
            raw.loc[raw.status.eq("non_subcritical"), "msa_geoid"].nunique()
        ),
        missing_clusters=[c for c, keep in zip(GROUPS, present) if not keep],
        n_per_regime={str(c): int(n) for c, n in zip(GROUPS, counts)},
        T_obs=observed,
        n_permutations=n_perm,
        permutation_seed=seed,
        p_perm=(
            float((1 + (null >= observed).sum()) / (n_perm + 1)) if estimable else None
        ),
        null_mean=float(null.mean()) if estimable else None,
        null_q95=float(np.quantile(null, 0.95)) if estimable else None,
        independent_grouped_recalculation_max_error=group_error,
        max_row_centering_error=row_error,
        max_column_centering_error=column_error,
    )
    write_json(out / "interaction_statistics.json", stats)
    return (cells, stats)


def plot(root, cells, stats):
    path = root / "figures" / f"{STEM}.png"
    if stats["status"] != "computed":
        if path.exists():
            path.unlink()
        return None
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "text.color": "#20262C",
            "axes.labelcolor": "#20262C",
            "xtick.color": "#30363B",
            "ytick.color": "#30363B",
            "axes.edgecolor": "#50575D",
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 10.4,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "savefig.bbox": None,
        }
    )
    width, height, dpi = (18.0, 5.0, 600)
    fig = plt.figure(figsize=(width / 2.54, height / 2.54), dpi=150)
    ax = fig.add_axes([5.0 / width, 0.55 / height, 10.7 / width, 3.9 / height])
    cax = fig.add_axes([16.1 / width, 0.55 / height, 0.28 / width, 3.9 / height])
    pct = (np.expm1(cells.to_numpy(float)) * 100).T
    ordered = np.sort(np.abs(pct[np.isfinite(pct)]))[::-1]
    top = ordered[1] if ordered[0] > 1.5 * ordered[1] else ordered[0]
    limit = max(10.0, float(np.ceil(top / 10) * 10))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#D9D9D9")
    im = ax.imshow(
        np.ma.masked_invalid(np.clip(pct, -limit, limit)),
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(GROUPS)), [f"C{c}" for c in GROUPS])
    ax.set_yticks(range(len(SCENARIOS)), [s for _, s in SCENARIOS])
    ax.tick_params(which="major", length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, pct.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, pct.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Amplification ratio (AR)", fontsize=10.4, pad=4)
    cb = fig.colorbar(im, cax=cax, ticks=[-limit, -limit / 2, 0, limit / 2, limit])
    cb.set_label("Deviation from additivity (%)", fontsize=10.4, labelpad=7)
    cb.ax.tick_params(labelsize=9.5, width=0.65, length=2.5, pad=3)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "results/functional_shocks"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        raw = pd.read_csv(
            out / "category_shock_raw.csv", dtype={"msa_geoid": str, "naics_code": str}
        )
        members = pd.read_csv(
            out / "complete_case_membership.csv", dtype={"msa_geoid": str}
        )[["msa_geoid", "msa_name", "cluster"]].rename(
            columns={"msa_geoid": "geoid", "msa_name": "name"}
        )
    else:
        members = pd.read_csv(
            DATA_DIR / "functional_cluster_membership_k8.csv", dtype={"geoid": str}
        )
        raw = simulate(root, members)
    cells, stats = calculate_interaction(raw, members, out)
    path = plot(root, cells, stats)
    print(
        json.dumps(
            dict(figure=str(path) if path is not None else None, **stats), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
