"""Landmark table: for every ventilation episode, a prediction point every 12 h from
48 h after intubation. Features use only data charted strictly before the
landmark; the label is whether VAP onset falls within the next `horizon` hours.
Landmarks at or after onset or after extubation are dropped. Also computes the
CPIS components (temperature, WBC, secretions, oxygenation, radiograph from
CheXpert labels, culture) and the culture-ordering columns.

Output: data/cohort/landmarks_{h}h.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C
from cpis_plus import scoring

LANDMARK_EVERY = 12          # hours between prediction points
LOOKBACKS = (6, 24, 48)      # summary windows before each landmark
CXR_MAX_AGE_H = 48           # a film older than this is not representative


def log(msg):
    print(f"[data] {msg}", flush=True)


# --------------------------------------------------------------------------
def make_landmarks(cohort, labels, horizon):
    """One row per (episode, landmark time) with the binary outcome."""
    df = cohort.merge(labels[["stay_id", "vap_any", "vap_onset"]], on="stay_id")
    rows = []
    for r in df.itertuples():
        t0 = r.vent_start + pd.Timedelta(hours=C.MIN_VENT_HOURS)
        # observation ends at extubation, or at onset if VAP occurred
        t_end = r.vent_end
        if pd.notna(r.vap_onset):
            t_end = min(t_end, r.vap_onset)
        t = t0
        # strict inequality: a landmark exactly AT onset would be scored as a
        # negative while sitting on the event itself
        while t < t_end:
            if pd.notna(r.vap_onset):
                y = int(r.vap_onset > t and
                        (r.vap_onset - t) <= pd.Timedelta(hours=horizon))
            else:
                y = 0
            rows.append((r.stay_id, r.subject_id, r.hadm_id, r.vent_start,
                         t, y, bool(r.vap_any), r.vap_onset, r.age, r.gender,
                         r.first_careunit, r.admission_type))
            t += pd.Timedelta(hours=LANDMARK_EVERY)
    lm = pd.DataFrame(rows, columns=[
        "stay_id", "subject_id", "hadm_id", "vent_start", "t", "y", "vap_any",
        "vap_onset", "age", "gender", "first_careunit", "admission_type"])
    lm["hrs_on_vent"] = (lm.t - lm.vent_start).dt.total_seconds() / 3600
    log(f"horizon {horizon}h: {len(lm):,} landmarks, {lm.y.sum():,} positive "
        f"({lm.y.mean():.2%}), {lm.stay_id.nunique():,} episodes")
    return lm


# --------------------------------------------------------------------------
def summarise(events, lm, value_col="valuenum", key="stay_id"):
    """Window summaries of every feature strictly before each landmark.

    Implemented as a merge_asof-style sweep: for each feature we sort events and
    landmarks by time and take min/max/mean/last/slope over each lookback.
    """
    out = {}
    ev = events.dropna(subset=[value_col])
    lm_key = lm[[key, "t"]].reset_index()
    max_lb = max(LOOKBACKS)
    for feat, g in ev.groupby("feature"):
        g = g[[key, "charttime", value_col]].sort_values("charttime")
        # join once per feature, then slice the lookbacks out of the same frame
        merged = lm_key.merge(g, on=key, how="left")
        dt = (merged.t - merged.charttime).dt.total_seconds() / 3600
        merged = merged[(dt > 0) & (dt <= max_lb)]   # strictly before -> no leakage
        merged["dt"] = dt[merged.index]
        if merged.empty:
            continue
        last = merged.sort_values("charttime").groupby("index")[value_col].last()
        out[f"{feat}_last"] = last
        out[f"{feat}_n"] = merged.groupby("index")[value_col].size()
        for lb in LOOKBACKS:
            gp = merged[merged.dt <= lb].groupby("index")[value_col]
            out[f"{feat}_mean_{lb}h"] = gp.mean()
            out[f"{feat}_min_{lb}h"] = gp.min()
            out[f"{feat}_max_{lb}h"] = gp.max()
        log(f"  {feat}: {merged.groupby('index').ngroups:,} landmarks covered")
    F = pd.DataFrame(out, index=lm.index)
    return F


def text_features(events, lm):
    """Most recent charted value of the free-text CPIS items (secretions etc.)."""
    out = {}
    for feat in ("sputum_color", "sputum_consistency", "sputum_amount", "o2_device",
                 "vent_mode"):
        g = events[events.feature == feat][["stay_id", "charttime", "value"]]
        if g.empty:
            continue
        merged = lm[["stay_id", "t"]].reset_index().merge(g, on="stay_id", how="left")
        dt = (merged.t - merged.charttime).dt.total_seconds() / 3600
        sub = merged[(dt > 0) & (dt <= 24)].sort_values("charttime")
        out[feat] = sub.groupby("index")["value"].last()
    return pd.DataFrame(out, index=lm.index)


def cxr_features(lm, avail):
    """The latest chest film strictly before each landmark, within CXR_MAX_AGE_H."""
    c = avail[["stay_id", "dicom_id", "study_id", "study_time", "view"]].copy()
    merged = lm[["stay_id", "t"]].reset_index().merge(c, on="stay_id", how="left")
    dt = (merged.t - merged.study_time).dt.total_seconds() / 3600
    sub = merged[(dt > 0) & (dt <= CXR_MAX_AGE_H)].sort_values("study_time")
    last = sub.groupby("index").last()
    F = pd.DataFrame(index=lm.index)
    F["cxr_dicom_id"] = last.dicom_id
    F["cxr_study_id"] = last.study_id
    F["cxr_age_h"] = (lm.t - last.study_time).dt.total_seconds() / 3600
    return F


def cpis_features(lm, F, T):
    """The six classical CPIS components + the total, as of each landmark."""
    temp = F.get("temp_c_max_24h")
    tf = F.get("temp_f_max_24h")
    if tf is not None:
        temp = temp.fillna((tf - 32) * 5 / 9) if temp is not None else (tf - 32) * 5 / 9

    out = pd.DataFrame(index=lm.index)
    out["cpis_temp"] = temp.map(scoring.score_temperature) if temp is not None else np.nan
    wbc = F.get("wbc_last")
    bands = F.get("bands_last")
    out["cpis_wbc"] = [
        scoring.score_wbc(w, b) for w, b in
        zip(wbc if wbc is not None else [np.nan] * len(lm),
            bands if bands is not None else [np.nan] * len(lm))]
    out["cpis_secretions"] = [
        scoring.score_secretions(a, c, k) for a, c, k in
        zip(T.get("sputum_amount", pd.Series(index=lm.index, dtype=object)),
            T.get("sputum_color", pd.Series(index=lm.index, dtype=object)),
            T.get("sputum_consistency", pd.Series(index=lm.index, dtype=object)))]

    pao2 = F.get("pao2_min_24h")
    if "pao2_lab_min_24h" in F:
        pao2 = F["pao2_lab_min_24h"] if pao2 is None else pao2.fillna(F["pao2_lab_min_24h"])
    fio2 = F.get("fio2_max_24h")
    out["cpis_oxygenation"] = [
        scoring.score_oxygenation(p, f) for p, f in
        zip(pao2 if pao2 is not None else [np.nan] * len(lm),
            fio2 if fio2 is not None else [np.nan] * len(lm))]
    out["cpis_radiograph"] = np.nan   # filled by add_radiograph_component
    out["cpis_culture"] = np.nan      # filled by add_culture_component
    return out


def add_radiograph_component(ds):
    """CPIS radiograph component from the CheXpert labels of the linked film.

    CheXpert encodes 1.0 positive / 0.0 negative / -1.0 uncertain / NaN not
    mentioned. Consolidation is the focal finding, so it scores 2 (localised);
    a non-consolidative opacity scores 1 (diffuse/patchy).
    """
    cx = pd.read_csv(C.CXR_CHEXPERT)
    cx = cx.set_index("study_id")
    sid = ds.cxr_study_id
    have = sid.notna()

    def col(name):
        v = pd.Series(np.nan, index=ds.index)
        v[have] = cx[name].reindex(sid[have].astype("int64")).values
        return v

    consolidation = col("Consolidation").eq(1)
    pneumonia = col("Pneumonia").eq(1)
    opacity = col("Lung Opacity").eq(1) | col("Edema").eq(1)

    score = pd.Series(np.nan, index=ds.index)
    score[have] = 0
    score[have & opacity] = 1
    score[have & (consolidation | pneumonia)] = 2
    ds["cpis_radiograph"] = score
    ds["cxr_consolidation"] = consolidation.where(have)
    ds["cxr_opacity"] = opacity.where(have)
    log(f"radiograph component available for {have.mean():.1%} of landmarks")
    return ds


def add_culture_component(ds, cohort):
    """CPIS culture component from respiratory microbiology strictly before T.

    LEAKAGE WARNING. The proxy label's onset *is* the first qualifying positive
    culture, and landmarks at/after onset are dropped, so a culture visible here
    is by construction non-qualifying. But the act of *sending* a culture
    encodes the clinician's suspicion of VAP, which is exactly what the model is
    supposed to predict independently. These columns are therefore prefixed
    `leaky_` and excluded from the model feature set by default; they are used
    only to compute the classical CPIS baseline, where culture is a legitimate
    component of the published score.
    """
    micro = pd.read_csv(
        C.HOSP / "microbiologyevents.csv.gz",
        usecols=["hadm_id", "charttime", "chartdate", "spec_type_desc", "org_name"],
        parse_dates=["charttime", "chartdate"], low_memory=False)
    micro = micro[micro.hadm_id.isin(set(cohort.hadm_id))].copy()
    micro["time"] = micro.charttime.fillna(micro.chartdate)
    spec = micro.spec_type_desc.fillna("").str.upper()
    resp = micro[spec.str.contains("|".join(C.RESP_SPECIMENS), regex=True)].copy()
    org = resp.org_name.fillna("").str.upper()
    resp["positive"] = org.ne("") & ~org.str.contains("|".join(C.NON_PATHOGENS), regex=True)

    m = ds[["hadm_id", "t"]].reset_index().merge(
        resp[["hadm_id", "time", "positive"]], on="hadm_id", how="left")
    dt = (m.t - m.time).dt.total_seconds() / 3600
    sub = m[(dt > 0) & (dt <= 72)].sort_values("time")
    last = sub.groupby("index").last()

    pos = pd.Series(np.nan, index=ds.index)
    pos.loc[last.index] = last.positive.astype(float)
    ds["cpis_culture"] = pos.map(lambda p: scoring.score_culture(p) if pd.notna(p) else np.nan)
    ds["leaky_culture_sent_72h"] = ds.index.isin(last.index).astype(int)
    ds["leaky_culture_positive"] = pos
    log(f"culture component available for {pos.notna().mean():.1%} of landmarks")
    return ds


# --------------------------------------------------------------------------
if __name__ == "__main__":
    cohort = pd.read_parquet(C.COHORT / "cohort.parquet")
    labels = pd.read_parquet(C.COHORT / "labels.parquet")
    avail = pd.read_parquet(C.COHORT / "cxr_linked.parquet")
    ce = pd.read_parquet(C.INTERIM / "chartevents_cohort.parquet")
    le = pd.read_parquet(C.INTERIM / "labevents_cohort.parquet")
    log(f"chartevents {len(ce):,}  labevents {len(le):,}")

    # Labs are hadm-scoped, and 220 of 6,891 admissions contain more than one
    # cohort episode - so a lab cannot be mapped to a stay by hadm_id alone.
    # Assign each lab to the episode whose ventilation window contains it; a lab
    # drawn between two episodes of the same admission is dropped rather than
    # attributed to an arbitrary one.
    keys = cohort[["hadm_id", "stay_id", "vent_start", "vent_end"]]
    le = le.merge(keys, on="hadm_id", how="inner")
    in_win = ((le.charttime >= le.vent_start - pd.Timedelta(hours=24))
              & (le.charttime <= le.vent_end + pd.Timedelta(hours=24)))
    le = le[in_win].drop(columns=["vent_start", "vent_end"])
    log(f"labs assigned to an episode: {len(le):,}")
    events = pd.concat([ce[["stay_id", "charttime", "feature", "value", "valuenum"]],
                        le[["stay_id", "charttime", "feature", "value", "valuenum"]]],
                       ignore_index=True)

    for horizon in C.PREDICTION_HORIZONS:
        lm = make_landmarks(cohort, labels, horizon)
        F = summarise(events, lm)
        T = text_features(events, lm)
        X = cxr_features(lm, avail)
        P = cpis_features(lm, F, T)
        ds = pd.concat([lm, F, T, X, P], axis=1)
        ds = add_radiograph_component(ds)
        ds = add_culture_component(ds, cohort)
        ds["cpis_total"] = scoring.total(ds)
        ds.to_parquet(C.COHORT / f"landmarks_{horizon}h.parquet", index=False)
        log(f"  -> landmarks_{horizon}h.parquet  {ds.shape}")
        log(f"     with CXR: {ds.cxr_dicom_id.notna().mean():.1%}  "
            f"CPIS computable: {ds.cpis_total.notna().mean():.1%}")
    log("done")
