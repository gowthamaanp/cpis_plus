"""SOFA severity composite (0-24) built from signals already in the landmark
table: PaO2/FiO2 -> respiratory, platelets -> coagulation, bilirubin -> liver,
MAP and vasopressor exposure -> cardiovascular, GCS -> CNS, creatinine -> renal.
Everything is computed from windows ending strictly before the landmark.

Output: data/cohort/landmarks_{h}h_severity.parquet
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C

warnings.filterwarnings("ignore")


def log(m):
    print(f"[severity] {m}", flush=True)


def add_pf_ratio(ds):
    """Explicit PaO2/FiO2 - the quantity CPIS actually scores."""
    new = {}
    pao2_src = {}
    for w in ("6h", "24h", "48h"):
        for agg in ("mean", "min", "max"):
            p = ds.get(f"pao2_{agg}_{w}")
            pl = ds.get(f"pao2_lab_{agg}_{w}")
            if p is None and pl is None:
                continue
            po = p.fillna(pl) if (p is not None and pl is not None) else (p if p is not None else pl)
            pao2_src[f"{agg}_{w}"] = po
    for w in ("6h", "24h", "48h"):
        f = ds.get(f"fio2_max_{w}")
        fl = ds.get(f"fio2_lab_max_{w}")
        fio = f.fillna(fl) if (f is not None and fl is not None) else (f if f is not None else fl)
        if fio is None:
            continue
        frac = np.where(fio > 1, fio / 100.0, fio)
        frac = pd.Series(frac, index=ds.index).replace(0, np.nan)
        for agg in ("mean", "min"):
            po = pao2_src.get(f"{agg}_{w}")
            if po is None:
                continue
            new[f"pf_ratio_{agg}_{w}"] = po / frac
    F = pd.DataFrame(new, index=ds.index)
    # derived: trend and own-baseline deviation of the ratio
    if "pf_ratio_min_6h" in F and "pf_ratio_min_24h" in F:
        F["pf_ratio_trend_acute"] = F.pf_ratio_min_6h - F.pf_ratio_min_24h
    if "pf_ratio_min_24h" in F and "pf_ratio_min_48h" in F:
        F["pf_ratio_trend_sub"] = F.pf_ratio_min_24h - F.pf_ratio_min_48h
    log(f"added {F.shape[1]} PaO2/FiO2 ratio features "
        f"(computable on {F.notna().any(axis=1).mean():.1%} of landmarks)")
    return F


def sofa_component(ds):
    """Standard SOFA points (0-4 per organ system) from what we already have."""
    def _get(name, agg, win="24h"):
        for n in (name, f"{name}_lab"):
            c = f"{n}_{agg}_{win}"
            if c in ds.columns:
                return ds[c]
        return pd.Series(np.nan, index=ds.index)

    resp = np.select(
        [ds.get("pf_ratio_min_24h", pd.Series(np.nan, index=ds.index)) < 100,
         ds.get("pf_ratio_min_24h", pd.Series(np.nan, index=ds.index)) < 200,
         ds.get("pf_ratio_min_24h", pd.Series(np.nan, index=ds.index)) < 300,
         ds.get("pf_ratio_min_24h", pd.Series(np.nan, index=ds.index)) < 400],
        [4, 3, 2, 1], default=0)

    plt = _get("platelet", "min")
    coag = np.select([plt < 20, plt < 50, plt < 100, plt < 150], [4, 3, 2, 1], default=0)
    coag = np.where(plt.isna(), np.nan, coag)

    bili = _get("bilirubin", "max")
    liver = np.select([bili >= 12, bili >= 6, bili >= 2, bili >= 1.2],
                      [4, 3, 2, 1], default=0)
    liver = np.where(bili.isna(), np.nan, liver)

    mp = _get("map", "min")
    vaso = ds.get("cum_vasopressor_sum", pd.Series(0, index=ds.index)).fillna(0)
    cardio = np.select([vaso > 0, mp < 70], [2, 1], default=0)
    cardio = np.where(mp.isna(), np.nan, cardio)

    gcs = ds[["gcs_motor_min_24h", "gcs_verbal_min_24h", "gcs_eye_min_24h"]].sum(
        axis=1, min_count=1) if "gcs_motor_min_24h" in ds.columns else pd.Series(np.nan, index=ds.index)
    cns = np.select([gcs < 6, gcs < 10, gcs < 13, gcs < 15], [4, 3, 2, 1], default=0)
    cns = np.where(gcs.isna(), np.nan, cns)

    cr = _get("creatinine", "max")
    uo = ds.get("cum_urine_output_sum", pd.Series(np.nan, index=ds.index))
    renal = np.select([cr >= 5, cr >= 3.5, cr >= 2, cr >= 1.2], [4, 3, 2, 1], default=0)
    renal = np.where(cr.isna(), np.nan, renal)

    F = pd.DataFrame({
        "sofa_resp": resp, "sofa_coag": coag, "sofa_liver": liver,
        "sofa_cardio": cardio, "sofa_cns": cns, "sofa_renal": renal,
    }, index=ds.index)
    n_obs = F.notna().sum(axis=1)
    F["sofa_total"] = F.fillna(0).sum(axis=1).where(n_obs >= 4)
    log(f"SOFA computable on {F.sofa_total.notna().mean():.1%} of landmarks, "
        f"mean {F.sofa_total.mean():.1f}")
    return F


def main():
    for h in C.PREDICTION_HORIZONS:
        ds = pd.read_parquet(C.COHORT / f"landmarks_{h}h_extended.parquet").reset_index(drop=True)
        PF = add_pf_ratio(ds)
        PF = PF[[c for c in PF.columns if c not in ds.columns]]
        ds = pd.concat([ds, PF], axis=1)
        ds = pd.concat([ds, sofa_component(ds)], axis=1)
        ds.to_parquet(C.COHORT / f"landmarks_{h}h_severity.parquet", index=False)
        log(f"landmarks_{h}h_severity.parquet {ds.shape}")


if __name__ == "__main__":
    main()
