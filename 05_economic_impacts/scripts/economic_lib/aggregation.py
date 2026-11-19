"""Visit-weighted tract aggregation of building-value-scaled impacts."""

import numpy as np
import pandas as pd


def strict_sum(x):
    """Do not silently replace unknown building values / weights with zero."""
    return float(np.sum(np.asarray(x, dtype=float)))


def dollar_product(fraction, building):
    out = fraction * building
    # Exactly zero model impact contributes zero, even if valuation is unknown.
    out[fraction == 0] = 0.0
    out[building == 0] = 0.0
    return out


class TractAggregation:
    def __init__(self, local, visits, buildings):
        self.local = local
        self.visits = np.asarray(visits)
        self.codes, tract_ids = pd.factorize(
            local.tract, sort=False, use_na_sentinel=False
        )
        self.count = len(tract_ids)
        self.total = np.bincount(self.codes, weights=visits, minlength=self.count)
        self.n_pois = np.bincount(self.codes, minlength=self.count)
        self.building = buildings.reindex(tract_ids).to_numpy(float)
        self.weights = np.divide(
            visits,
            self.total[self.codes],
            out=np.zeros(len(visits)),
            where=self.total[self.codes] > 0,
        )
        sums = np.bincount(self.codes, weights=self.weights, minlength=self.count)
        assert np.isfinite(visits).all() and (visits >= 0).all()
        self.table = pd.DataFrame(
            {
                "tract": tract_ids,
                "n_pois": self.n_pois,
                "total_visits": self.total,
                "visit_weight_sum": sums,
                "BUILDVALUE": self.building,
            }
        )
        self.table["missing_BUILDVALUE"] = ~np.isfinite(self.building)
        self.table["zero_BUILDVALUE"] = self.building == 0
        self.table["zero_total_visits"] = self.total == 0
        self.table["invalid_tract_id"] = ~self.table.tract.astype(
            "string"
        ).str.fullmatch(r"\d{11}", na=False) | self.table.tract.eq("00000000000")
        assert not self.table.tract.duplicated().any()

    def aggregate(self, f, v):
        assert np.isfinite(f).all() and np.isfinite(v).all()
        assert np.isin(f, [0.0, 1.0]).all() and (v >= -1e-9).all()
        hit = f == 1
        counts = np.bincount(self.codes, weights=hit, minlength=self.count).astype(int)
        numerator = np.bincount(
            self.codes, weights=self.visits * v, minlength=self.count
        )
        initial_numerator = np.bincount(
            self.codes, weights=self.visits * f, minlength=self.count
        )
        propagated = np.divide(
            numerator, self.total, out=np.full(self.count, np.nan), where=self.total > 0
        )
        initial = np.divide(
            initial_numerator,
            self.total,
            out=np.full(self.count, np.nan),
            where=self.total > 0,
        )
        t = self.table.copy()
        t["n_shocked_pois"] = counts
        t["all_pois_shocked"] = counts == self.n_pois
        t["visit_weighted_initial_loss"] = initial
        t["visit_weighted_propagated_loss"] = propagated
        t["sum_visits_times_v"] = numerator
        t["initial_dollar_loss"] = dollar_product(initial, self.building)
        t["propagated_dollar_impact"] = dollar_product(propagated, self.building)
        return t


def summarize_tracts(t, kind):
    touched = t.n_shocked_pois > 0
    if kind == "hazard":
        assert t.loc[
            touched, "all_pois_shocked"
        ].all(), "Hazard tract not fully shocked"
        # The explicit hazard denominator is one BUILDVALUE per selected tract.
        t["initial_dollar_loss"] = np.where(touched, t.BUILDVALUE, 0.0)
    initial, prop = (
        strict_sum(t[c]) for c in ["initial_dollar_loss", "propagated_dollar_impact"]
    )
    initial_known, prop_known = (
        float(t[c].sum()) for c in ["initial_dollar_loss", "propagated_dollar_impact"]
    )
    return {
        "number_shocked_tracts": int(touched.sum()),
        "number_fully_shocked_tracts": int(t.all_pois_shocked.sum()),
        "initial_building_value_loss_dollars": initial,
        "propagated_dollar_impact": prop,
        "dollar_amplification_ratio": prop / initial if initial > 0 else np.nan,
        "known_coverage_initial_loss_dollars": initial_known,
        "known_coverage_propagated_impact_dollars": prop_known,
        "known_coverage_dollar_amplification_ratio": (
            prop_known / initial_known if initial_known > 0 else np.nan
        ),
        "n_unknown_initial_contributions": int(t.initial_dollar_loss.isna().sum()),
        "n_unknown_propagated_contributions": int(
            t.propagated_dollar_impact.isna().sum()
        ),
        "dollar_totals_complete": bool(np.isfinite(initial) and np.isfinite(prop)),
    }
