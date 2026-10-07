"""Fold the extended clinical signals into the landmark table using the same
strictly-before-landmark windows, plus static comorbidities and cumulative
organ-support exposure.

Output: data/cohort/landmarks_{h}h_extended.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C

LOOKBACKS = (6, 24, 48)
BASELINE_H = 24


def log(msg):
    print(f"[merge] {msg}", flush=True)


def summarise(ev, lm):
    out = {}
    lm_key = lm[["stay_id", "t"]].reset_index()
    max_lb = max(LOOKBACKS)
    for feat, g in ev.groupby("feature"):
        g = g[["stay_id", "charttime", "valuenum"]].sort_values("charttime")
        m = lm_key.merge(g, on="stay_id", how="left")
        dt = (m.t - m.charttime).dt.total_seconds() / 3600
        m = m[(dt > 0) & (dt <= max_lb)]
        if m.empty:
            continue
        m = m.copy(); m["dt"] = dt[m.index]
        out[f"{feat}_last"] = m.sort_values("charttime").groupby("index").valuenum.last()
        for lb in LOOKBACKS:
            gp = m[m.dt <= lb].groupby("index").valuenum
            out[f"{feat}_mean_{lb}h"] = gp.mean()
            out[f"{feat}_max_{lb}h"] = gp.max()
            out[f"{feat}_n_{lb}h"] = gp.size()
    return pd.DataFrame(out, index=lm.index)


def cumulative(ev, lm):
    """Total exposure since the start of ventilation, strictly before T."""
    out = {}
    lm_key = lm[["stay_id", "t"]].reset_index()
    for feat in ("vasopressor", "sedative", "opioid", "paralytic", "urine_output",
                 "hemodialysis", "crrt", "prone", "bronchoscopy"):
        g = ev[ev.feature == feat][["stay_id", "charttime", "valuenum"]]
        if g.empty:
            continue
        m = lm_key.merge(g, on="stay_id", how="left")
        dt = (m.t - m.charttime).dt.total_seconds() / 3600
        sub = m[dt > 0]
        gp = sub.groupby("index").valuenum
        out[f"cum_{feat}_sum"] = gp.sum()
        out[f"cum_{feat}_n"] = gp.size()
    return pd.DataFrame(out, index=lm.index)


def main():
    cohort = pd.read_parquet(C.COHORT / "cohort.parquet")
    ev = pd.read_parquet(C.INTERIM / "extended_cohort.parquet")
    cm = pd.read_parquet(C.INTERIM / "comorbidities.parquet")
    log(f"extended events {len(ev):,}, comorbidity rows {len(cm):,}")

    for h in C.PREDICTION_HORIZONS:
        ds = pd.read_parquet(C.COHORT / f"landmarks_{h}h_trajectory.parquet")
        F = summarise(ev, ds)
        G = cumulative(ev, ds)
        log(f"[{h}h] added {F.shape[1]} windowed + {G.shape[1]} cumulative features")

        ds = pd.concat([ds, F, G], axis=1)
        ds = ds.merge(cm, on="hadm_id", how="left")

        # cumulative ventilator-days is the strongest known VAP risk factor and
        # is already present as hrs_on_vent; add its interaction with sedation
        if "cum_sedative_n" in ds:
            ds["sedation_per_vent_day"] = ds.cum_sedative_n / (ds.hrs_on_vent / 24).clip(lower=1)
        if "cum_urine_output_sum" in ds:
            ds["urine_per_hour"] = ds.cum_urine_output_sum / ds.hrs_on_vent.clip(lower=1)

        ds.to_parquet(C.COHORT / f"landmarks_{h}h_extended.parquet", index=False)
        log(f"-> landmarks_{h}h_extended.parquet {ds.shape}")


if __name__ == "__main__":
    main()
