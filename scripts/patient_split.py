"""Build the patient-level 60/20/20 split (seed 0) from the locked landmark
table and check its integrity.

Output: data/cohort/canonical_split.json
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C
from cpis_plus import split as SPL

H = 24


def main():
    ds = pd.read_parquet(C.COHORT / f"landmarks_{H}h_final.parquet").reset_index(drop=True)
    payload = SPL.make_canonical_split(ds)
    for name, s in payload["summary"].items():
        print(f"{name:5s} {s['n_patients']:5,} patients {s['n_episodes']:5,} episodes "
              f"{s['n_landmarks']:6,} landmarks prevalence {s['patient_prevalence']:.4f}")
    SPL.check_split_integrity(SPL.apply_split(ds))
    print("integrity OK")


if __name__ == "__main__":
    main()
