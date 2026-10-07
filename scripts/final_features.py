"""Lock the 339-feature set: 18 clinical signals with their window summaries,
trend, deviation and variability terms plus static covariates ("Tier-A"; 332
columns) and the 7 SOFA columns. Writes the landmark table every evaluation
script reads.

Outputs: data/cohort/landmarks_{h}h_final.parquet, final_features_{h}h.json
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C
from cpis_plus import data as D

CORE15 = [
    "temp_c", "temp_f", "wbc", "sputum_color", "sputum_consistency", "sputum_amount",
    "fio2", "fio2_lab", "peep", "peak_insp_pressure", "tidal_volume",
    "resp_rate", "resp_rate_total", "spo2", "heart_rate",
    "gcs_motor", "gcs_verbal", "gcs_eye", "urine_output",
]
TIER_A_EXTRA = ["mean_airway_pressure", "minute_volume", "rass", "platelet", "creatinine"]
STATIC_A = ["age", "gender", "hrs_on_vent", "vent_day", "first_careunit", "admission_type"]


def tier_a_columns(ds):
    signals = CORE15 + TIER_A_EXTRA
    pat = re.compile(r"^(dev_|trend_|var_)?(" + "|".join(
        re.escape(s) for s in sorted(signals, key=len, reverse=True)) + r")(_|$)")
    cols = []
    for c in ds.columns:
        if c in D.ID_COLS or c in D.SUSPICION_COLS or c.startswith("leaky_"):
            continue
        if c.endswith("_n") or re.search(r"_n_\d+h$", c):
            continue
        if pat.match(c):
            cols.append(c)
    cols += [c for c in STATIC_A if c in ds.columns]
    return list(dict.fromkeys(cols))


def main():
    for h in C.PREDICTION_HORIZONS:
        ds = pd.read_parquet(C.COHORT / f"landmarks_{h}h_severity.parquet").reset_index(drop=True)
        cols = list(dict.fromkeys(tier_a_columns(ds) + [c for c in ds.columns if c.startswith("sofa_")]))
        keep = ["stay_id", "subject_id", "hadm_id", "t", "y", "vap_onset", "vap_any"] + cols
        out = ds[list(dict.fromkeys(keep))].copy()
        D.assert_no_single_feature_leak(D.encode(out, cols), out.y)
        out.to_parquet(C.COHORT / f"landmarks_{h}h_final.parquet", index=False)
        (C.COHORT / f"final_features_{h}h.json").write_text(
            json.dumps({"horizon": h, "features": cols, "n_features": len(cols)}, indent=2))
        print(f"{h}h: {len(cols)} features")


if __name__ == "__main__":
    main()
