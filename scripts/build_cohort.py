"""Cohort and VAP proxy label. Adults on invasive ventilation for more than 48 h,
excluding pneumonia documented within 48 h of admission. One row per ventilation
episode. VAP onset = culture collection time of the earliest pathogen-positive
lower-respiratory culture more than 48 h after intubation paired with a newly
started antibiotic within +/-48 h.

Outputs (data/cohort/): vent_episodes.parquet, cohort.parquet, labels.parquet,
inclusion_flow.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C


def log(msg):
    print(f"[cohort] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. ventilation episodes
# --------------------------------------------------------------------------
def ventilation_episodes() -> pd.DataFrame:
    """Invasive ventilation episodes from procedureevents (225792).

    procedureevents rows are already interval-encoded (starttime/endtime), but a
    single continuous course is sometimes split across consecutive rows. Merge
    rows separated by <24h into one episode, which is the standard convention:
    a reintubation within 24h is a continuation, not a new exposure.
    """
    pe = pd.read_csv(
        C.ICU / "procedureevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "itemid", "starttime", "endtime"],
        parse_dates=["starttime", "endtime"],
    )
    log(f"procedureevents rows: {len(pe):,}")
    vent = pe[pe.itemid == C.IT_INVASIVE_VENT].copy()
    log(f"invasive-vent rows (225792): {len(vent):,}")

    vent = vent.sort_values(["stay_id", "starttime"]).reset_index(drop=True)
    gap = vent.groupby("stay_id")["endtime"].shift()
    new_ep = (vent.starttime - gap) > pd.Timedelta(hours=24)
    new_ep = new_ep.fillna(True) | (vent.stay_id != vent.stay_id.shift())
    vent["episode_id"] = new_ep.cumsum()

    ep = (
        vent.groupby(["episode_id", "subject_id", "hadm_id", "stay_id"], as_index=False)
        .agg(vent_start=("starttime", "min"), vent_end=("endtime", "max"))
    )
    ep["vent_hours"] = (ep.vent_end - ep.vent_start).dt.total_seconds() / 3600
    # keep the first (index) episode per ICU stay: later episodes are confounded
    # by the outcome of the first
    ep = ep.sort_values(["stay_id", "vent_start"]).drop_duplicates("stay_id", keep="first")
    log(f"merged episodes: {len(ep):,} (median {ep.vent_hours.median():.1f}h)")
    return ep.reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. inclusion / exclusion
# --------------------------------------------------------------------------
def apply_criteria(ep: pd.DataFrame) -> pd.DataFrame:
    icustays = pd.read_csv(
        C.ICU / "icustays.csv.gz", parse_dates=["intime", "outtime"],
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime",
                 "outtime", "los"],
    )
    adm = pd.read_csv(
        C.HOSP / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "admission_type", "race", "insurance", "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    pat = pd.read_csv(C.HOSP / "patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
                      parse_dates=["dod"])

    df = (ep.merge(icustays, on=["subject_id", "hadm_id", "stay_id"], how="left")
            .merge(adm, on=["subject_id", "hadm_id"], how="left")
            .merge(pat, on="subject_id", how="left"))

    # MIMIC-IV age: anchor_age is the age in anchor_year
    df["age"] = df.anchor_age + (df.admittime.dt.year - df.anchor_year)

    n0 = len(df)
    steps = []

    def keep(mask, name):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        steps.append((name, before, len(df), before - len(df)))
        log(f"{name}: {before:,} -> {len(df):,}  (-{before - len(df):,})")

    keep(df.age >= C.MIN_AGE, f"age >= {C.MIN_AGE}")
    keep(df.vent_hours > C.MIN_VENT_HOURS, f"vent > {C.MIN_VENT_HOURS}h")

    # exclusion: pneumonia present on admission. MIMIC-IV diagnoses are not
    # POA-flagged, so we use the conservative surrogate used in the VAP
    # literature: any pneumonia ICD code on an admission where ventilation began
    # within 48h of hospital admission cannot be distinguished from
    # community-acquired pneumonia -> drop.
    dx = pd.read_csv(C.HOSP / "diagnoses_icd.csv.gz")
    dx["icd_code"] = dx.icd_code.str.strip().str.upper()
    pna_hadm = set(dx.loc[dx.icd_code.str.startswith(C.ICD_PNEUMONIA_PREFIX), "hadm_id"])
    hrs_adm_to_vent = (df.vent_start - df.admittime).dt.total_seconds() / 3600
    early_vent_with_pna = df.hadm_id.isin(pna_hadm) & (hrs_adm_to_vent < 48)
    keep(~early_vent_with_pna, "no pneumonia-on-admission")

    # need enough post-48h observation to have a prediction target at all
    keep(df.outtime.notna(), "valid ICU stay record")

    # 93 episodes have procedureevents ventilation starting well before ICU
    # intime (intubated in the ED or OR, charted against the later ICU stay).
    # Their "first 24h of ventilation" baseline would be drawn from outside the
    # ICU record, so the personalised deviations would be unreliable.
    df["pre_icu_vent"] = df.vent_start < df.intime - pd.Timedelta(hours=6)
    log(f"flagged {df.pre_icu_vent.sum()} episodes ventilated before ICU admission "
        f"(retained; excludable via pre_icu_vent for sensitivity analysis)")

    df["hrs_adm_to_vent"] = (df.vent_start - df.admittime).dt.total_seconds() / 3600
    df["vap_window_start"] = df.vent_start + pd.Timedelta(hours=C.VAP_ONSET_MIN_HOURS)

    flow = pd.DataFrame(steps, columns=["step", "n_before", "n_after", "n_removed"])
    flow.to_csv(C.COHORT / "inclusion_flow.csv", index=False)
    log(f"final cohort: {len(df):,} episodes / {df.subject_id.nunique():,} patients "
        f"({n0:,} candidates)")
    return df


# --------------------------------------------------------------------------
# 3. VAP labels
# --------------------------------------------------------------------------
def vap_labels(cohort: pd.DataFrame) -> pd.DataFrame:
    """Two complementary labels.

    vap_icd     - admission carries an explicit VAP code (997.31 / J95.851).
                  High precision, no onset time, low prevalence.
    vap_proxy   - the CDC/NHSN-style clinical surrogate, which *does* give an
                  onset time: a respiratory culture drawn >48h after intubation
                  together with a new antibiotic started within +/-48h of it.
    vap_any     - either. `vap_onset` comes from the proxy when available.
    """
    hadms = set(cohort.hadm_id.dropna())

    # --- ICD anchor -------------------------------------------------------
    dx = pd.read_csv(C.HOSP / "diagnoses_icd.csv.gz")
    dx["icd_code"] = dx.icd_code.str.strip().str.upper()
    vap_hadm = set(dx.loc[dx.icd_code.isin(C.ICD_VAP), "hadm_id"]) & hadms
    log(f"VAP by explicit ICD code: {len(vap_hadm):,} admissions")

    # --- respiratory cultures --------------------------------------------
    micro = pd.read_csv(
        C.HOSP / "microbiologyevents.csv.gz",
        usecols=["subject_id", "hadm_id", "charttime", "chartdate", "spec_type_desc",
                 "org_name"],
        parse_dates=["charttime", "chartdate"],
        low_memory=False,
    )
    micro = micro[micro.hadm_id.isin(hadms)]
    micro["time"] = micro.charttime.fillna(micro.chartdate)
    spec = micro.spec_type_desc.fillna("").str.upper()
    resp = micro[spec.str.contains("|".join(C.RESP_SPECIMENS), regex=True)].copy()
    org = resp.org_name.fillna("").str.upper()
    resp["positive"] = org.ne("") & ~org.str.contains("|".join(C.NON_PATHOGENS), regex=True)
    log(f"respiratory cultures in cohort admissions: {len(resp):,} "
        f"({resp.positive.mean():.1%} grew a true pathogen)")

    # --- antibiotic starts -------------------------------------------------
    abx = _antibiotic_starts(hadms)
    log(f"antibiotic administrations in cohort admissions: {len(abx):,}")

    # --- combine into suspicion-of-VAP events ------------------------------
    cand = resp[resp.positive].merge(
        cohort[["hadm_id", "stay_id", "vent_start", "vent_end", "vap_window_start"]],
        on="hadm_id", how="inner")
    cand = cand[(cand.time >= cand.vap_window_start) &
                (cand.time <= cand.vent_end + pd.Timedelta(hours=48))]
    log(f"cultures inside the VAP window: {len(cand):,}")

    abx["agent"] = _normalise_agent(abx.drug)
    ab_by_hadm = {h: (g.agent.values, g.starttime.values)
                  for h, g in abx.groupby("hadm_id")}

    def has_new_abx(row):
        """A *new* antibiotic agent started around the culture.

        VAP frequently develops in patients already receiving antibiotics, so
        the signal is escalation, not exposure: we require an agent whose start
        falls in the window and that was NOT being given in the preceding
        NEW_ABX_LOOKBACK_H hours. Continuation of a running course does not
        count.
        """
        rec = ab_by_hadm.get(row.hadm_id)
        if rec is None:
            return False
        agents, starts = rec
        t = np.datetime64(row.time)
        d = (starts - t) / np.timedelta64(1, "h")
        in_window = (d >= -C.ABX_WINDOW_H) & (d <= C.ABX_WINDOW_H)
        if not in_window.any():
            return False
        for ag in set(agents[in_window]):
            first_in = d[in_window & (agents == ag)].min()
            prior = (agents == ag) & (d < first_in) & (d >= first_in - C.NEW_ABX_LOOKBACK_H)
            if not prior.any():
                return True
        return False

    cand["abx_match"] = cand.apply(has_new_abx, axis=1)
    events = cand[cand.abx_match]
    onset = events.groupby("stay_id")["time"].min().rename("vap_onset")
    log(f"proxy-VAP episodes: {onset.notna().sum():,}")

    out = cohort[["stay_id", "hadm_id", "subject_id", "vent_start", "vent_end",
                  "vap_window_start"]].copy()
    out = out.merge(onset, on="stay_id", how="left")
    out["vap_proxy"] = out.vap_onset.notna()
    out["vap_icd"] = out.hadm_id.isin(vap_hadm)
    out["vap_any"] = out.vap_proxy | out.vap_icd
    out["hrs_vent_to_onset"] = (out.vap_onset - out.vent_start).dt.total_seconds() / 3600

    log(f"prevalence  proxy={out.vap_proxy.mean():.2%}  icd={out.vap_icd.mean():.2%}  "
        f"any={out.vap_any.mean():.2%}")
    return out


_ABX = [
    "vancomycin", "piperacillin", "tazobactam", "cefepime", "meropenem", "imipenem",
    "ceftazidime", "ceftriaxone", "cefazolin", "levofloxacin", "ciprofloxacin",
    "moxifloxacin", "azithromycin", "linezolid", "daptomycin", "gentamicin",
    "tobramycin", "amikacin", "aztreonam", "ampicillin", "amoxicillin", "colistin",
    "polymyxin", "doxycycline", "clindamycin", "metronidazole", "ertapenem",
    "cefuroxime", "nafcillin", "oxacillin", "trimethoprim", "sulfamethoxazole",
    "tigecycline", "ceftaroline",
]


def _normalise_agent(drug: pd.Series) -> np.ndarray:
    """Map free-text drug names onto the canonical agent in _ABX."""
    d = drug.fillna("").str.lower()
    out = np.full(len(d), "other", dtype=object)
    for a in _ABX:
        hit = d.str.contains(a, regex=False).values
        out[hit & (out == "other")] = a
    return out


def _antibiotic_starts(hadms) -> pd.DataFrame:
    """First administration time of each antibiotic course, from prescriptions."""
    cols = ["subject_id", "hadm_id", "drug", "starttime", "stoptime", "route"]
    chunks = []
    for ch in pd.read_csv(C.HOSP / "prescriptions.csv.gz", usecols=cols,
                          parse_dates=["starttime", "stoptime"], chunksize=2_000_000,
                          low_memory=False):
        ch = ch[ch.hadm_id.isin(hadms)]
        if ch.empty:
            continue
        d = ch.drug.fillna("").str.lower()
        ch = ch[d.str.contains("|".join(_ABX), regex=True)]
        # IV/IM only - oral prophylaxis is not treatment of a new VAP
        r = ch.route.fillna("").str.upper()
        ch = ch[r.str.contains("IV|IM|INTRAVEN")]
        chunks.append(ch)
    abx = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    return abx.dropna(subset=["starttime"])


if __name__ == "__main__":
    ep = ventilation_episodes()
    ep.to_parquet(C.COHORT / "vent_episodes.parquet", index=False)

    cohort = apply_criteria(ep)
    cohort.to_parquet(C.COHORT / "cohort.parquet", index=False)

    labels = vap_labels(cohort)
    labels.to_parquet(C.COHORT / "labels.parquet", index=False)
    log("done")
