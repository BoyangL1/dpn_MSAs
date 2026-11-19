"""Estimate building-value-weighted impacts of hazard and matched random shocks."""

from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
SHOCK = ROOT.parent / "04_disaster_shocks"
DATA_ROOT = Path(os.environ.get("DPN_DATA_DIR", ROOT.parent / "data")).expanduser()
DATA = DATA_ROOT / "disaster_shocks"
sys.path.insert(0, str(SHOCK / "scripts"))

import numpy as np
import pandas as pd
import pyogrio
from shock_lib import propagation as model
from economic_lib.aggregation import TractAggregation, summarize_tracts


def building_values(source):
    frame = pyogrio.read_dataframe(
        source, columns=["TRACTFIPS", "BUILDVALUE"], read_geometry=False
    )
    frame["TRACTFIPS"] = frame.TRACTFIPS.astype(str).str.zfill(11)
    if not frame.TRACTFIPS.is_unique or frame.BUILDVALUE.lt(0).any():
        raise ValueError(
            "FEMA tract IDs must be unique and building values nonnegative."
        )
    return frame.set_index("TRACTFIPS").BUILDVALUE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA, help="Prepared disaster-shock inputs."
    )
    parser.add_argument(
        "--fema-source",
        type=Path,
        default=DATA_ROOT / "geography/nri_tracts.shp",
        help="FEMA census-tract shapefile containing TRACTFIPS and BUILDVALUE.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser()
    config = json.loads((SHOCK / "config.json").read_text())
    beta = float(config["beta"])
    if config["orientation"] != "W" or config["shock_value"] != -1:
        raise ValueError("This analysis requires W-oriented negative unit shocks.")
    n_random = int(config["random_draws"])
    cohort = pd.read_csv(
        data_dir / "hazard_coverage_cohort.csv", dtype={"msa_geoid": str}
    )
    included = cohort.included_in_coverage.astype(str).str.lower().eq("true")
    study_ids = set(cohort.loc[included, "msa_geoid"])
    baseline = pd.read_csv(
        SHOCK / "results/shock_results.csv", dtype={"msa_geoid": str}
    )
    baseline = baseline.loc[baseline.msa_geoid.isin(study_ids)].copy()
    if not baseline.status.eq("ok").all():
        raise ValueError("Economic impacts require valid subcritical hazard cases.")
    if baseline.duplicated(["msa_geoid", "hazard"]).any():
        raise ValueError("Hazard cases must be unique within each MSA.")
    buildings = building_values(args.fema_source)
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes["msa_geoid"] = nodes.msa_geoid.astype(str).str.zfill(5)
    nodes["tract"] = nodes.tract.astype(str).str.zfill(11)
    edges["msa_geoid"] = edges.msa_geoid.astype(str).str.zfill(5)
    scores = pd.read_parquet(data_dir / "hazard_scores.parquet")
    scores["TRACTFIPS"] = scores.TRACTFIPS.astype(str).str.zfill(11)
    scores = scores.set_index("TRACTFIPS")
    node_groups = nodes.groupby("msa_geoid", sort=False).indices
    edge_groups = edges.groupby("msa_geoid", sort=False).indices
    summaries = []
    city_ids = sorted(baseline.msa_geoid.unique())
    for city_number, geoid in enumerate(city_ids, 1):
        local_nodes = nodes.iloc[node_groups[geoid]]
        local_edges = edges.iloc[edge_groups.get(geoid, np.array([], dtype=int))]
        cascade = model.build_cascade(local_nodes, local_edges, beta)
        local, visits, n = cascade["nodes"], cascade["visits"], cascade["n"]
        if not cascade["metadata"]["subcritical"]:
            raise ValueError(f"Network {geoid} is not subcritical at beta={beta}.")
        aggregation = TractAggregation(local, visits, buildings)
        tract_audit = aggregation.table
        for old in baseline.loc[baseline.msa_geoid.eq(geoid)].itertuples(index=False):
            hazard = old.hazard
            selected, chosen, _, _, _ = model.select_tracts(
                local, scores, hazard, config["within_msa_tract_fraction"]
            )
            case_rows = []
            for realization in range(n_random + 1):
                kind, seed = (
                    ("hazard", -1) if realization == 0 else ("random", realization - 1)
                )
                selected_now = (
                    selected
                    if seed == -1
                    else model.seed_rng(geoid, hazard, seed).choice(
                        n, len(selected), replace=False
                    )
                )
                _, response = model.run_shock(cascade, selected_now, -1.0)
                initial = np.zeros(n)
                initial[selected_now] = 1.0
                tracts = aggregation.aggregate(initial, -response)
                case_rows.append(summarize_tracts(tracts, kind))
            case = pd.DataFrame(case_rows)
            random = case.iloc[1:]
            h = case_rows[0]
            strict_mean = lambda col: float(np.mean(random[col].to_numpy()))
            mean_ar = strict_mean("dollar_amplification_ratio")
            mean_known_ar = strict_mean("known_coverage_dollar_amplification_ratio")
            summary = dict(
                msa_geoid=geoid,
                msa_name=old.msa_name,
                hazard=hazard,
                hazard_name=old.hazard_name,
                beta=beta,
                number_shocked_tracts=len(chosen),
                number_shocked_pois=len(selected),
                n_random=n_random,
                initial_building_value_loss_dollars=h[
                    "initial_building_value_loss_dollars"
                ],
                propagated_dollar_impact=h["propagated_dollar_impact"],
                hazard_dollar_amplification_ratio=h["dollar_amplification_ratio"],
                mean_random_initial_loss_dollars=strict_mean(
                    "initial_building_value_loss_dollars"
                ),
                mean_random_propagated_impact_dollars=strict_mean(
                    "propagated_dollar_impact"
                ),
                mean_random_dollar_amplification_ratio=mean_ar,
                hazard_random_dollar_amplification_ratio=(
                    h["dollar_amplification_ratio"] / mean_ar if mean_ar > 0 else np.nan
                ),
                n_random_defined_dollar_ar=int(
                    random.dollar_amplification_ratio.notna().sum()
                ),
                n_missing_BUILDVALUE_tracts=int(tract_audit.missing_BUILDVALUE.sum()),
                n_zero_BUILDVALUE_tracts=int(tract_audit.zero_BUILDVALUE.sum()),
                n_zero_visit_tracts=int(tract_audit.zero_total_visits.sum()),
                all_dollar_totals_complete=bool(case.dollar_totals_complete.all()),
                known_coverage_hazard_initial_loss_dollars=h[
                    "known_coverage_initial_loss_dollars"
                ],
                known_coverage_hazard_propagated_impact_dollars=h[
                    "known_coverage_propagated_impact_dollars"
                ],
                known_coverage_hazard_dollar_ar=h[
                    "known_coverage_dollar_amplification_ratio"
                ],
                known_coverage_mean_random_initial_loss_dollars=strict_mean(
                    "known_coverage_initial_loss_dollars"
                ),
                known_coverage_mean_random_propagated_impact_dollars=strict_mean(
                    "known_coverage_propagated_impact_dollars"
                ),
                known_coverage_mean_random_dollar_ar=mean_known_ar,
                known_coverage_hazard_random_dollar_ratio=(
                    h["known_coverage_dollar_amplification_ratio"] / mean_known_ar
                    if mean_known_ar > 0
                    else np.nan
                ),
                original_hazard_ar_unweighted=old.amplification_ratio_unw,
                original_random_mean_ar_unweighted=old.amplification_ratio_unw_rand,
            )
            denom = summary["mean_random_propagated_impact_dollars"]
            summary["hazard_random_propagated_loss_ratio"] = (
                summary["propagated_dollar_impact"] / denom if denom > 0 else np.nan
            )
            summaries.append(summary)
        print(
            f"{city_number}/{len(city_ids)} MSAs; {len(summaries)}/{len(baseline)} hazard cases",
            flush=True,
        )
    summary = pd.DataFrame(summaries)
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "msa_hazard_summary.csv", index=False)


if __name__ == "__main__":
    main()
