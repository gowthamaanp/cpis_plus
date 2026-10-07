"""Trajectory features added to the landmark table: deviation from the patient's own
baseline (dev_), acute-vs-subacute trend (trend_) and within-window variability
(var_). All use windows ending strictly before the landmark.

Output: data/cohort/landmarks_{h}h_trajectory.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C
from cpis_plus import data as D

BASELINE_H = 24          # first N hours of ventilation define the personal baseline


def log(msg):
    print(f"[trajectory] {msg}", flush=True)


def load_events(cohort):
    ce = pd.read_parquet(C.INTERIM / "chartevents_cohort.parquet")
    le = pd.read_parquet(C.INTERIM / "labevents_cohort.parquet")
    keys = cohort[["hadm_id", "stay_id", "vent_start", "vent_end"]]
    le = le.merge(keys, on="hadm_id", how="inner")
    m = ((le.charttime >= le.vent_start - pd.Timedelta(hours=24))
         & (le.charttime <= le.vent_end + pd.Timedelta(hours=24)))
    le = le[m].drop(columns=["vent_start", "vent_end"])
    cols = ["stay_id", "charttime", "feature", "valuenum"]
    return pd.concat([ce[cols], le[cols]], ignore_index=True).dropna(subset=["valuenum"])


def baselines(events, cohort):
    """Median of each feature over the first BASELINE_H hours of ventilation."""
    e = events.merge(cohort[["stay_id", "vent_start"]], on="stay_id", how="inner")
    h = (e.charttime - e.vent_start).dt.total_seconds() / 3600
    b = e[(h >= 0) & (h <= BASELINE_H)]
    out = (b.groupby(["stay_id", "feature"]).valuenum.median()
            .unstack("feature"))
    log(f"baselines computed for {len(out):,} episodes x {out.shape[1]} features")
    return out


def add_trajectory(ds, base):
    """Baseline deviations, cross-window trends and dispersion."""
    feats = sorted({c.rsplit("_mean_", 1)[0] for c in ds.columns if "_mean_" in c})
    new = {}
    n_dev = n_trend = 0
    for f in feats:
        m6, m24, m48 = (ds.get(f"{f}_mean_{h}h") for h in (6, 24, 48))
        mn24, mx24 = ds.get(f"{f}_min_24h"), ds.get(f"{f}_max_24h")

        if m6 is not None and m24 is not None:
            new[f"trend_{f}_acute"] = m6 - m24
            n_trend += 1
        if m24 is not None and m48 is not None:
            new[f"trend_{f}_sub"] = m24 - m48
        if mn24 is not None and mx24 is not None:
            new[f"var_{f}_range24"] = mx24 - mn24

        if f in base.columns:
            b = base[f].reindex(ds.stay_id).values
            if m6 is not None:
                new[f"dev_{f}_6h"] = m6.values - b
                n_dev += 1
            if m24 is not None:
                new[f"dev_{f}_24h"] = m24.values - b
            if mx24 is not None:
                new[f"dev_{f}_max24"] = mx24.values - b
    log(f"added {len(new)} trajectory features ({n_dev} baseline-referenced, "
        f"{n_trend} trend)")
    return pd.concat([ds, pd.DataFrame(new, index=ds.index)], axis=1)


def main():
    cohort = pd.read_parquet(C.COHORT / "cohort.parquet")
    events = load_events(cohort)
    base = baselines(events, cohort)

    for h in C.PREDICTION_HORIZONS:
        ds = D.load_landmarks(h)
        ds = add_trajectory(ds, base)
        # time since ventilation started is itself informative (VAP risk peaks
        # in the first week then declines)
        ds["vent_day"] = (ds.hrs_on_vent / 24).round()
        ds.to_parquet(C.COHORT / f"landmarks_{h}h_trajectory.parquet", index=False)
        log(f"-> landmarks_{h}h_trajectory.parquet {ds.shape}")


if __name__ == "__main__":
    main()
