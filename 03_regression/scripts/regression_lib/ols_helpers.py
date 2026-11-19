import itertools, math

import numpy as np

import pandas as pd

import statsmodels.api as sm


def zscore(frame):
    sd = frame.std(ddof=0)
    if (sd <= 0).any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite or constant predictor")
    return (frame - frame.mean()) / sd


def fit(y, raw, source, name):
    X = sm.add_constant(zscore(raw))
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": source.to_numpy()})
    interval = res.conf_int()
    tidy = pd.DataFrame(
        dict(
            model=name,
            term=X.columns,
            coef=res.params.to_numpy(),
            std_err=res.bse.to_numpy(),
            t=res.tvalues.to_numpy(),
            p_value=res.pvalues.to_numpy(),
            ci_low=interval.iloc[:, 0].to_numpy(),
            ci_high=interval.iloc[:, 1].to_numpy(),
            nobs=int(res.nobs),
            r2=res.rsquared,
            r2_adj=res.rsquared_adj,
        )
    )
    return res, tidy, X


def dyadic_covariance(X, residual, source, target):
    """Sandwich with all shared endpoints, including reciprocated directed pairs.

    Sum endpoint score outer products, subtract unordered-pair score outer
    products (otherwise each pair, including its reverse, is counted twice).
    Asymptotic estimator, no finite-sample adjustment or eigenvalue clipping.
    """
    X = np.asarray(X, float)
    ids = sorted(set(source) | set(target))
    lookup = {g: k for k, g in enumerate(ids)}
    s = np.array([lookup[g] for g in source])
    t = np.array([lookup[g] for g in target])
    if np.any(s == t):
        raise ValueError("Self pairs not allowed")
    score = X * np.asarray(residual)[:, None]
    endpoint = np.zeros((len(ids), X.shape[1]))
    np.add.at(endpoint, s, score)
    np.add.at(endpoint, t, score)
    pair_id = np.minimum(s, t) * len(ids) + np.maximum(s, t)
    _, codes = np.unique(pair_id, return_inverse=True)
    pair_score = np.zeros((codes.max() + 1, X.shape[1]))
    np.add.at(pair_score, codes, score)
    meat = endpoint.T @ endpoint - pair_score.T @ pair_score
    bread = np.linalg.inv(X.T @ X)
    return bread @ meat @ bread


def comparison(res, model):
    return dict(
        model=model,
        n_predictors=int(res.df_model),
        r2=res.rsquared,
        r2_adj=res.rsquared_adj,
        aic=res.aic,
        bic=res.bic,
        nobs=int(res.nobs),
    )


def shapley_blocks(y, raw, groups):
    X = zscore(raw).to_numpy()
    y = np.asarray(y) - np.mean(y)
    gram, xy, tss = X.T @ X, X.T @ y, y @ y
    names = list(raw.columns)
    util, rows = {}, []
    for n_groups in range(len(groups) + 1):
        for keys in itertools.combinations(groups, n_groups):
            cols = [c for g in keys for c in groups[g]]
            if cols:
                ix = [names.index(c) for c in cols]
                G, q = gram[np.ix_(ix, ix)], xy[ix]
                beta = np.linalg.solve(G, q)
                rank = np.linalg.matrix_rank(G)
                if rank != len(cols):
                    raise ValueError("Rank-deficient group coalition")
                r2 = float(q @ beta / tss)
                adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - rank - 1)
            else:
                rank, r2, adj = 0, 0.0, 0.0
            util[frozenset(keys)] = adj
            rows.append(
                dict(
                    included_groups="+".join(keys) or "none",
                    n_predictors=rank,
                    R2=r2,
                    adjusted_R2=adj,
                )
            )
    contributions = []
    for group in groups:
        others = [g for g in groups if g != group]
        value = 0.0
        for n in range(len(others) + 1):
            weight = (
                math.factorial(n)
                * math.factorial(len(groups) - n - 1)
                / math.factorial(len(groups))
            )
            for subset in itertools.combinations(others, n):
                before = frozenset(subset)
                value += weight * (util[before | {group}] - util[before])
        contributions.append(
            dict(
                group=group,
                shapley_delta_adj_r2=value,
                share_of_total_explained_variance=value / util[frozenset(groups)],
            )
        )
    result = pd.DataFrame(contributions).sort_values(
        "shapley_delta_adj_r2", ascending=False
    )
    assert np.isclose(result.shapley_delta_adj_r2.sum(), util[frozenset(groups)])
    return result, pd.DataFrame(rows)
