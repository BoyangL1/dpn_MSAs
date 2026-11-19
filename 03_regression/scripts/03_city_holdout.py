"""Evaluate regression generalization with five city-disjoint folds."""

from regression_lib.analysis_paths import INPUTS, RESULTS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

OUT = RESULTS


def main():
    pairs = pd.read_parquet(INPUTS / "transfer_pair_dataset.parquet")
    ids = sorted(pairs.source.unique())
    lookup = {g: i for i, g in enumerate(ids)}
    s = pairs.source.map(lookup).to_numpy()
    t = pairs.target.map(lookup).to_numpy()
    y = pairs.r_ij.to_numpy()
    dictionary = pd.read_csv(OUT / "variable_dictionary.csv")
    features = (
        dictionary.term[dictionary.term.str.startswith("diff_")]
        .str.removeprefix("diff_")
        .tolist()
    )
    city = (
        pd.read_csv(
            INPUTS / "regression_msa_features_before_transform.csv",
            dtype={"msa_geoid": str},
        )
        .set_index("msa_geoid")
        .loc[ids]
    )
    values = city[features].copy()
    values.loc[values.uf_poly_K.eq(0) & values.uf_poly_D.isna(), "uf_poly_D"] = 1.0
    for c in ["uf_poi_density", "uf_poly_K", "rd_intersection_density"]:
        values[c] = np.log1p(values[c].clip(lower=0))
    size = np.log1p(city.n_nodes.to_numpy())

    def prediction(train_cities, train_pairs):
        v = values.fillna(values.iloc[train_cities].median()).to_numpy()
        x = np.column_stack(
            [size[s] - size[t], pairs.log_geo_dist, np.abs(v[s] - v[t])]
        )
        assert np.isfinite(x).all() and x.shape[1] == 43
        mean = x[train_pairs].mean(0)
        sd = x[train_pairs].std(0)
        assert (sd > 0).all()
        z = (x - mean) / sd
        fit = LinearRegression().fit(z[train_pairs], y[train_pairs])
        assert fit.rank_ == 43
        return fit.predict(z)

    fold_ids = np.full(len(ids), -1)
    for fold, (_, test) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=20261111).split(ids)
    ):
        fold_ids[test] = fold
    pd.DataFrame({"msa_geoid": ids, "fold": fold_ids}).to_csv(
        OUT / "city_holdout_folds.csv", index=False
    )
    predictions, scores = [], []
    for fold in range(5):
        held = fold_ids == fold
        train = (~held[s]) & (~held[t])
        pred = prediction(np.flatnonzero(~held), train)
        for scenario, mask in {
            "unseen_source": held[s] & (~held[t]),
            "unseen_target": (~held[s]) & held[t],
            "both_cities_unseen": held[s] & held[t],
        }.items():
            q = pairs.loc[mask, ["source", "target", "r_ij"]].copy()
            q["prediction"], q["fold"], q["scenario"] = pred[mask], fold, scenario
            predictions.append(q)
            scores.append(
                dict(
                    fold=fold,
                    scenario=scenario,
                    n_train_pairs=int(train.sum()),
                    n_test_pairs=int(mask.sum()),
                    r2=r2_score(y[mask], pred[mask]),
                    rmse=float(np.sqrt(mean_squared_error(y[mask], pred[mask]))),
                )
            )
        print(f"City-disjoint validation fold {fold+1}/5 complete", flush=True)
    all_pred = pd.concat(predictions, ignore_index=True)
    all_pred.to_parquet(INPUTS / "city_holdout_predictions.parquet", index=False)
    pd.DataFrame(scores).to_csv(OUT / "city_holdout_fold_scores.csv", index=False)
    pooled = []
    for scenario, q in all_pred.groupby("scenario"):
        assert not q.duplicated(["source", "target"]).any()
        pooled.append(
            dict(
                model="original_abs_difference",
                scenario=scenario,
                n_pairs=len(q),
                r2=r2_score(q.r_ij, q.prediction),
                rmse=float(np.sqrt(mean_squared_error(q.r_ij, q.prediction))),
            )
        )
    pd.DataFrame(pooled).to_csv(OUT / "city_disjoint_validation.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
