"""Fit directed-pair OLS models with clustered and dyadic covariance."""

from regression_lib.analysis_paths import (
    MAIN,
    INPUTS,
    RESULTS,
    PUBLISHED,
    FIGURES,
    CONFIG,
    DATA_DIR,
    CBSA,
)
import json, itertools, platform
import numpy as np, pandas as pd
import geopandas as gpd
import scipy, statsmodels
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests
from regression_lib import ols_helpers as h

I = INPUTS
OUT = RESULTS
BLOCKS = json.loads((CONFIG / "original_feature_blocks.json").read_text())
NAMES = list(BLOCKS)
SPECS = {
    m: [NAMES[j] for j in ix]
    for m, ix in {
        "M0": [0],
        "M1": [1],
        "M2": [2],
        "M3": [0, 1],
        "M4": [0, 2],
        "M5": [1, 2],
        "M6": [0, 1, 2],
    }.items()
}


def dump(obj, path):
    path.write_text(
        json.dumps(
            obj, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)
        )
        + "\n"
    )


def read(p, index="msa_geoid"):
    return pd.read_csv(p, dtype={index: str}).set_index(index)


def assemble():
    T = pd.read_csv(
        MAIN / "results/gnn_all_pairs/transfer_pearson_analysis.csv", index_col=0
    )
    T.index = T.index.astype(str)
    T.columns = T.columns.astype(str)
    ids = T.index.tolist()
    T = T.loc[ids, ids]
    assert len(ids) >= 8 and np.isfinite(T).all().all()
    city = (
        read(I / "node_based_msa_features.csv")
        .loc[ids]
        .join(read(I / "network_msa_features.csv").drop(columns="n_nodes"))
        .join(read(I / "urban_form_raw_msa_features.csv"))
    )

    city["incoming_strength_gini"] = read(
        I / "raw_w_strength_features.csv"
    ).incoming_strength_gini
    rdpath = DATA_DIR / "transferability/msa_road_network_metrics.csv"
    roads = (
        pd.read_csv(rdpath, dtype={"GEOID": str})
        .set_index("GEOID")[["intersection_density", "orientation_entropy"]]
        .rename(columns=lambda c: "rd_" + c)
    )
    roads.to_csv(I / "road_metrics_static.csv")
    city = city.join(roads)
    city.index.name = "msa_geoid"
    city.to_csv(I / "regression_msa_features_before_transform.csv")
    dictionary = pd.read_csv(CONFIG / "original_variable_dictionary.csv")
    features = [
        c.removeprefix("diff_") for c in dictionary.term if c.startswith("diff_")
    ]
    city_t = city[features].copy()
    dominance = city_t.uf_poly_D.isna() & city_t.uf_poly_K.eq(0)
    city_t.loc[dominance, "uf_poly_D"] = 1.0
    logs = ["uf_poi_density", "uf_poly_K", "rd_intersection_density"]
    for c in logs:
        city_t[c] = np.log1p(city_t[c].clip(lower=0))
    fill = city_t.median()
    missing = city_t.isna()
    imputation = []
    for g, c in zip(*np.where(missing)):
        imputation.append(
            dict(
                msa_geoid=ids[g],
                feature=city_t.columns[c],
                fill_value=float(fill.iloc[c]),
                reason="missing transformed feature; selected-cohort median",
            )
        )
    for g in city.index[dominance]:
        imputation.append(
            dict(
                msa_geoid=g,
                feature="uf_poly_D",
                fill_value=1.0,
                reason="No detected spatial center; original monocentric limit convention",
            )
        )
    pd.DataFrame(
        imputation, columns=["msa_geoid", "feature", "fill_value", "reason"]
    ).to_csv(I / "feature_imputation_audit.csv", index=False)
    city_t = city_t.fillna(fill)
    assert np.isfinite(city_t).all().all()
    means, sd = city_t.mean(), city_t.std(ddof=0)
    assert (sd > 0).all()
    pd.DataFrame(
        {
            "mean": means,
            "sd_ddof0": sd,
            "log1p_before_z": [c in logs for c in features],
            "z_before_pair_difference": [not c.startswith("scn_") for c in features],
        }
    ).to_csv(I / "msa_feature_scaling.csv")
    z = (city_t - means) / sd

    scenario = [c for c in z if c.startswith("scn_")]
    z[scenario] = city_t[scenario]
    z.to_csv(I / "regression_msa_predictor_values.csv")
    city_t.to_csv(I / "regression_msa_features_transformed.csv")
    metro = gpd.read_file(CBSA).set_index("GEOID").loc[ids].to_crs(5070)
    xy = np.column_stack([metro.centroid.x, metro.centroid.y])
    n = len(ids)
    s, t = np.where(~np.eye(n, dtype=bool))
    pair = pd.DataFrame(
        {
            "source": np.array(ids)[s],
            "target": np.array(ids)[t],
            "r_ij": T.to_numpy()[s, t],
        }
    )
    logn = np.log1p(city.n_nodes.to_numpy())
    pair["log_N_diff"] = logn[s] - logn[t]
    pair["log_geo_dist"] = np.log1p(np.linalg.norm(xy[s] - xy[t], axis=1) / 1000)
    for c in features:
        val = z[c].to_numpy()
        pair["diff_" + c] = np.abs(val[s] - val[t])
    pair = pair[["source", "target", "r_ij"] + sum(BLOCKS.values(), [])]
    pair.to_parquet(INPUTS / "transfer_pair_dataset.parquet", index=False)
    for i, r in dictionary.iterrows():
        c = r.term
        f = c.removeprefix("diff_")
        transform = (
            "signed log1p(network POI count) source minus target"
            if c == "log_N_diff"
            else (
                "log1p(EPSG:5070 centroid distance km)"
                if c == "log_geo_dist"
                else (
                    "absolute difference of active-network-node scenario shares"
                    if f.startswith("scn_")
                    else "absolute difference of selected-cohort MSA z-scores, ddof=0"
                )
            )
        )
        dictionary.loc[i, "pair_transform"] = transform
        dictionary.loc[i, "source"] = (
            "inputs/regression_msa_features_before_transform.csv; inputs/msa_feature_scaling.csv"
        )
        dictionary.loc[i, "definition_note"] = (
            "National-median-imputed CBG covariate, national z-score ddof=1, float32, mean across active POI nodes; NOT MSA population total"
            if f.startswith("acs_")
            else (
                "Resolved original definition: Gini of supplied WIJ row sums over within-MSA positive-edge endpoints; incoming of propagation P=W_raw.T. Verified against all 346 legacy cities."
                if f == "incoming_strength_gini"
                else (
                    "Frozen original road metrics, same CBSA boundaries"
                    if f.startswith("rd_")
                    else ""
                )
            )
        )
    dictionary.loc[dictionary.term.eq("diff_incoming_strength_gini"), "label"] = (
        "Dependency-strength Gini"
    )
    dictionary.to_csv(OUT / "variable_dictionary.csv", index=False)
    dump(BLOCKS, OUT / "feature_blocks.json")
    dump(SPECS, OUT / "model_specifications.json")
    return pair, dictionary


def main():
    for path in [I, OUT, PUBLISHED, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)
    pairs, dictionary = assemble()
    raw = pairs[sum(BLOCKS.values(), [])]
    ids = sorted(pairs.source.unique())
    y = pairs.r_ij
    assert len(pairs) == len(ids) * (len(ids) - 1) and set(pairs.target) == set(ids)
    assert (
        not pairs.duplicated(["source", "target"]).any()
        and not pairs.source.eq(pairs.target).any()
    )
    scales = pd.DataFrame(
        {
            "term": raw.columns,
            "pair_mean": raw.mean().values,
            "pair_sd_ddof0": raw.std(ddof=0).values,
        }
    )
    scales.to_csv(OUT / "predictor_scaling.csv", index=False)
    lookup = dictionary.set_index("term")
    allcoef = []
    dyadic = []
    comparisons = []
    tests = []
    vifs = []
    fits = {}
    covchecks = []
    predictions = pairs[["source", "target", "r_ij"]].copy()
    for m, bs in SPECS.items():
        cols = [c for b in bs for c in BLOCKS[b]]
        fit, tab, X = h.fit(y, raw[cols], pairs.source, m)
        fits[m] = fit
        x = X.to_numpy()
        beta_check = np.linalg.solve(x.T @ x, x.T @ y.to_numpy())
        assert np.max(np.abs(beta_check - fit.params)) < 1e-10
        assert np.linalg.matrix_rank(x.T @ x) == len(cols) + 1
        tab["covariance"] = "source_MSA_clustered"
        for c in ["block", "component", "label"]:
            tab[c] = tab.term.map(lookup[c]).fillna("Intercept")
        mask = tab.term.ne("const")
        tab.loc[mask, "q_BH_within_model"] = multipletests(
            tab.loc[mask, "p_value"], method="fdr_bh"
        )[1]
        tab["coef_outcome_SD"] = tab.coef / y.std(ddof=0)
        allcoef.append(tab)
        cov = h.dyadic_covariance(X, fit.resid, pairs.source, pairs.target)
        assert (np.diag(cov) > 0).all(), f"Nonpositive variance in {m}"
        robust = tab.copy()
        robust["std_err"] = np.sqrt(np.diag(cov))
        robust["t"] = robust.coef / robust.std_err
        robust["p_value"] = 2 * norm.sf(robust.t.abs())
        robust["ci_low"] = robust.coef - norm.ppf(0.975) * robust.std_err
        robust["ci_high"] = robust.coef + norm.ppf(0.975) * robust.std_err
        robust["covariance"] = "dyadic_shared_endpoint_asymptotic"
        robust.loc[mask, "q_BH_within_model"] = multipletests(
            robust.loc[mask, "p_value"], method="fdr_bh"
        )[1]
        dyadic.append(robust)
        cvdir = OUT / "covariance_matrices"
        cvdir.mkdir(exist_ok=True)
        fit.cov_params().to_csv(cvdir / f"{m}_source_clustered.csv")
        pd.DataFrame(cov, index=X.columns, columns=X.columns).to_csv(
            cvdir / f"{m}_dyadic.csv"
        )

        scores = x * fit.resid.to_numpy()[:, None]
        ss = pd.DataFrame(scores).groupby(pairs.source.to_numpy()).sum().to_numpy()
        bread = np.linalg.inv(x.T @ x)
        correction = len(ids) / (len(ids) - 1) * (len(y) - 1) / (len(y) - x.shape[1])
        manual = bread @ ss.T @ ss @ bread * correction
        cov_error = float(np.max(np.abs(manual - fit.cov_params().to_numpy())))
        assert cov_error < 1e-10
        covchecks.append(
            dict(
                model=m,
                source_sandwich_max_error=cov_error,
                normal_equations_max_beta_error=float(
                    np.max(np.abs(beta_check - fit.params))
                ),
            )
        )
        vif = np.diag(np.linalg.inv(np.corrcoef(x[:, 1:], rowvar=False)))
        vifs.extend(
            dict(
                model=m,
                term=c,
                block=lookup.loc[c, "block"],
                VIF=float(v),
                tolerance=float(1 / v),
            )
            for c, v in zip(cols, vif)
        )
        comparisons.append(
            dict(
                model=m,
                predictor_blocks=" + ".join(bs),
                n_msa=len(ids),
                **{k: v for k, v in h.comparison(fit, m).items() if k != "model"},
                rmse_in_sample=float(np.sqrt(np.mean(fit.resid**2))),
                delta_r2_vs_M0=fit.rsquared - fits["M0"].rsquared,
                delta_adj_r2_vs_M0=fit.rsquared_adj - fits["M0"].rsquared_adj,
                contains_M0_controls=set(SPECS["M0"]).issubset(bs),
                max_VIF=float(vif.max()),
                dyadic_cov_min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
            )
        )
        for b in bs:
            ix = [list(X.columns).index(c) for c in BLOCKS[b]]
            contrast = np.eye(x.shape[1])[ix]
            test = fit.wald_test(contrast, scalar=True)
            cc = cov[np.ix_(ix, ix)]
            be = fit.params.iloc[ix].to_numpy()
            positive = bool(np.linalg.eigvalsh(cc).min() > 0)
            wald = float(be @ np.linalg.solve(cc, be)) if positive else np.nan
            tests.append(
                dict(
                    model=m,
                    block=b,
                    df=len(ix),
                    wald_source_clustered=float(test.statistic),
                    p_source_clustered=float(test.pvalue),
                    wald_dyadic=wald,
                    p_dyadic=float(chi2.sf(wald, len(ix))) if positive else np.nan,
                    dyadic_block_cov_positive_definite=positive,
                )
            )
        predictions[m + "_fitted"] = fit.fittedvalues
        predictions[m + "_residual"] = fit.resid
        print(
            f"{m}: n={int(fit.nobs)}, p={len(cols)}, R2={fit.rsquared:.6f}, maxVIF={vif.max():.2f}",
            flush=True,
        )
    shares, coalitions = h.shapley_blocks(y, raw, BLOCKS)
    assert len(coalitions) == 8 and np.isclose(
        shares.shapley_delta_adj_r2.sum(), fits["M6"].rsquared_adj, atol=1e-10
    )
    for m, bs in SPECS.items():
        q = coalitions[
            coalitions.included_groups.map(lambda s: set(s.split("+")) == set(bs))
        ].iloc[0]
        assert abs(q.R2 - fits[m].rsquared) < 1e-10
    increments = []
    for a, b in itertools.product(SPECS, SPECS):
        if set(SPECS[a]) < set(SPECS[b]):
            dr = fits[b].rsquared - fits[a].rsquared
            assert dr >= -1e-12
            increments.append(
                dict(
                    reduced_model=a,
                    expanded_model=b,
                    added_blocks=" + ".join(x for x in SPECS[b] if x not in SPECS[a]),
                    delta_r2=dr,
                    delta_adj_r2=fits[b].rsquared_adj - fits[a].rsquared_adj,
                    partial_r2=dr / (1 - fits[a].rsquared),
                )
            )
    tables = {
        "model_comparison": pd.DataFrame(comparisons),
        "regression_coefficients": pd.concat(allcoef, ignore_index=True),
        "regression_dyadic_robust": pd.concat(dyadic, ignore_index=True),
        "group_joint_tests": pd.DataFrame(tests),
        "vif_by_model": pd.DataFrame(vifs),
        "group_contribution_shapley": shares,
        "model_combination_r2": coalitions,
        "r2_incremental": pd.DataFrame(increments),
        "variable_dictionary": dictionary,
        "predictor_scaling": scales,
    }
    for name, table in tables.items():
        destination = (
            PUBLISHED
            if name in {"regression_coefficients", "regression_dyadic_robust"}
            else OUT
        )
        table.to_csv(destination / f"{name}.csv", index=False)
    predictions.to_parquet(
        INPUTS / "pair_fitted_values_and_residuals.parquet", index=False
    )

    full = (
        tables["regression_coefficients"]
        .query("model == 'M6' and term != 'const'")
        .assign(abs_coef=lambda x: x.coef.abs())
    )
    selection = pd.concat(
        [
            full[full.component.eq(c)].nlargest(n, "abs_coef")
            for c, n in [
                ("Distance & Size", 2),
                ("Socioeconomic", 8),
                ("POI composition", 3),
                ("Network structure", 5),
                ("Urban form", 4),
            ]
        ]
    )
    selection.to_csv(OUT / "top_contributors_full.csv", index=False)
    dump(
        dict(
            n_msa=len(ids),
            n_pairs=len(pairs),
            n_predictors=43,
            outcome="Raw directed transfer Pearson r, gnn_all_pairs; no Fisher transformation",
            cohort="Newly fitted legacy GNNs with best_epoch > 1; excludes both roles of newly flagged MSAs",
            depends_on_k=False,
            models=SPECS,
            main_covariance="source-MSA clustered sandwich with default finite-sample correction; normal 95% intervals",
            sensitivity_covariance="Shared-endpoint dyadic sandwich, no finite-sample correction or eigenvalue clipping",
            k_specific_regressions=False,
            classification_regression=False,
            versions={
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "statsmodels": statsmodels.__version__,
            },
        ),
        OUT / "metadata.json",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
