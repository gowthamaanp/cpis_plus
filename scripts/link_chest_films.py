"""Link each MIMIC-CXR study to the ventilation episode whose window contains it
(intubation to 24 hours after extubation). Only the study timestamps and ids are
used, to attach the CheXpert labels for the CPIS radiograph component; no image
is read.

Output: data/cohort/cxr_linked.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C


def main():
    cohort = pd.read_parquet(C.COHORT / "cohort.parquet")
    labels = pd.read_parquet(C.COHORT / "labels.parquet")

    md = pd.read_csv(C.CXR_METADATA)
    md = md[md.subject_id.isin(set(cohort.subject_id))].copy()
    t = md.StudyTime.astype(str).str.split(".").str[0].str.zfill(6)
    md["study_time"] = pd.to_datetime(md.StudyDate.astype(str) + t,
                                      format="%Y%m%d%H%M%S", errors="coerce")
    md = md.rename(columns={"ViewPosition": "view"}).dropna(subset=["study_time"])

    keys = cohort[["subject_id", "stay_id", "hadm_id", "vent_start", "vent_end"]]
    j = md.merge(keys, on="subject_id", how="inner")
    inside = (j.study_time >= j.vent_start) & \
             (j.study_time <= j.vent_end + pd.Timedelta(hours=24))
    j = j[inside].copy()
    j["hrs_from_vent_start"] = (j.study_time - j.vent_start).dt.total_seconds() / 3600
    j = j.merge(labels[["stay_id", "vap_any", "vap_proxy", "vap_icd", "vap_onset"]],
                on="stay_id", how="left")
    j["hrs_to_onset"] = (j.vap_onset - j.study_time).dt.total_seconds() / 3600
    j.to_parquet(C.COHORT / "cxr_linked.parquet", index=False)
    print(f"linked {len(j):,} films across {j.stay_id.nunique():,} episodes")


if __name__ == "__main__":
    main()
