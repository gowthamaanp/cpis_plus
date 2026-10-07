"""Extra clinical signals from MIMIC-IV: ventilator mechanics, RASS/CAM-ICU,
vasopressors, sedatives, urine output, dialysis, prone positioning, and
comorbidity flags.

Output: data/interim/extended_cohort.parquet (and comorbidities.parquet)
"""
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C

CHUNK = 5_000_000

CHART_EXTRA = {
    224695: "peak_insp_pressure", 224696: "plateau_pressure",
    224697: "mean_airway_pressure", 220283: "resistance", 229661: "compliance",
    224738: "inspiratory_time", 224691: "flow_rate",
    228096: "rass", 220074: "cvp", 220088: "cardiac_output",
    223770: "spo2_alarm_low", 225126: "dialysis_patient",
}

# inputevents: continuous/bolus drugs. Grouped because the specific agent
# matters less than the class and the fact of escalation.
DRUG_GROUPS = {
    "vasopressor": [221906, 221749, 221289, 222315, 221662, 221653],  # norepi, phenyl,
                                                                      # epi, vasopressin,
                                                                      # dopamine, dobutamine
    "sedative": [222168, 221668, 225942, 225150],   # propofol, midazolam, dexmed, loraz
    "opioid": [221744, 225942, 221833],             # fentanyl, morphine, hydromorph
    "paralytic": [222062, 221555],                  # vecuronium, cisatracurium
}

PROC_EXTRA = {
    225441: "hemodialysis", 225802: "crrt", 225803: "cvvhd", 225805: "peritoneal",
    224385: "intubation_proc", 225448: "perc_trach", 226237: "open_trach",
    227194: "extubation", 225794: "niv", 224560: "prone", 225459: "bronchoscopy",
}

# Elixhauser/Charlson-style prefixes for conditions that raise pneumonia risk
COMORBID = {
    "copd": ("490", "491", "492", "493", "494", "495", "496", "J40", "J41", "J42",
             "J43", "J44", "J45", "J46", "J47"),
    "chf": ("428", "I50"),
    "diabetes": ("250", "E10", "E11"),
    "renal_failure": ("585", "N18"),
    "liver_disease": ("571", "K70", "K74"),
    "malignancy": ("140", "141", "150", "151", "162", "174", "185", "C"),
    "immunosuppressed": ("042", "279", "B20", "D80", "D81", "D82", "D83", "D84"),
    "stroke": ("430", "431", "432", "433", "434", "436", "I60", "I61", "I63"),
    "trauma": ("800", "801", "802", "803", "804", "805", "806", "807", "S0", "S1"),
    "alcohol_abuse": ("291", "303", "F10"),
    "obesity": ("278", "E66"),
}


@contextmanager
def gz_stream(path):
    p = subprocess.Popen(["unpigz", "-c", str(path)], stdout=subprocess.PIPE,
                         bufsize=1 << 20)
    try:
        yield p.stdout
    finally:
        p.stdout.close()
        p.wait()


def log(msg):
    print(f"[ext] {msg}", flush=True)


def chart_extra(w):
    stays = set(w.stay_id)
    b = w.set_index("stay_id")[["t_lo", "t_hi"]]
    keep, n = [], 0
    with gz_stream(C.ICU / "chartevents.csv.gz") as fh:
        for i, ch in enumerate(pd.read_csv(
                fh, usecols=["stay_id", "charttime", "itemid", "value", "valuenum"],
                dtype={"stay_id": "float64", "itemid": "int32", "value": "object",
                       "valuenum": "float32", "charttime": "object"},
                chunksize=CHUNK, low_memory=False), 1):
            n += len(ch)
            ch = ch[ch.itemid.isin(CHART_EXTRA) & ch.stay_id.isin(stays)]
            if not ch.empty:
                ch = ch.copy()
                ch["charttime"] = pd.to_datetime(ch.charttime)
                bb = b.reindex(ch.stay_id.values)
                ch = ch[(ch.charttime.values >= bb.t_lo.values)
                        & (ch.charttime.values <= bb.t_hi.values)]
                if not ch.empty:
                    keep.append(ch)
            if i % 20 == 0:
                log(f"  chartevents {n:,} rows, kept {sum(len(k) for k in keep):,}")
    out = pd.concat(keep, ignore_index=True)
    out["feature"] = out.itemid.map(CHART_EXTRA)
    return out


def drugs(w):
    stays = set(w.stay_id)
    ie = pd.read_csv(C.ICU / "inputevents.csv.gz",
                     usecols=["stay_id", "starttime", "itemid", "amount", "rate"],
                     parse_dates=["starttime"], low_memory=False)
    ie = ie[ie.stay_id.isin(stays)]
    m = {i: g for g, ids in DRUG_GROUPS.items() for i in ids}
    ie = ie[ie.itemid.isin(m)].copy()
    ie["feature"] = ie.itemid.map(m)
    ie = ie.rename(columns={"starttime": "charttime"})
    ie["valuenum"] = ie.rate.fillna(ie.amount)
    log(f"drug administrations: {len(ie):,} ({ie.feature.value_counts().to_dict()})")
    return ie[["stay_id", "charttime", "feature", "valuenum"]]


def urine(w):
    stays = set(w.stay_id)
    oe = pd.read_csv(C.ICU / "outputevents.csv.gz",
                     usecols=["stay_id", "charttime", "itemid", "value"],
                     parse_dates=["charttime"], low_memory=False)
    oe = oe[oe.stay_id.isin(stays)].copy()
    oe["feature"] = "urine_output"
    oe["valuenum"] = oe.value
    log(f"output events: {len(oe):,}")
    return oe[["stay_id", "charttime", "feature", "valuenum"]]


def procedures(w):
    stays = set(w.stay_id)
    pe = pd.read_csv(C.ICU / "procedureevents.csv.gz",
                     usecols=["stay_id", "starttime", "itemid"],
                     parse_dates=["starttime"], low_memory=False)
    pe = pe[pe.stay_id.isin(stays) & pe.itemid.isin(PROC_EXTRA)].copy()
    pe["feature"] = pe.itemid.map(PROC_EXTRA)
    pe = pe.rename(columns={"starttime": "charttime"})
    pe["valuenum"] = 1.0
    log(f"procedures: {len(pe):,} ({pe.feature.nunique()} types)")
    return pe[["stay_id", "charttime", "feature", "valuenum"]]


def comorbidities(cohort):
    dx = pd.read_csv(C.HOSP / "diagnoses_icd.csv.gz")
    dx["icd_code"] = dx.icd_code.str.strip().str.upper()
    dx = dx[dx.hadm_id.isin(set(cohort.hadm_id))]
    out = pd.DataFrame({"hadm_id": sorted(dx.hadm_id.unique())}).set_index("hadm_id")
    for name, pref in COMORBID.items():
        h = set(dx.loc[dx.icd_code.str.startswith(pref), "hadm_id"])
        out[f"cm_{name}"] = out.index.isin(h).astype(int)
    out["cm_count"] = out.sum(axis=1)
    log("comorbidity prevalence:\n" + out.mean().round(3).to_string())
    return out.reset_index()


if __name__ == "__main__":
    cohort = pd.read_parquet(C.COHORT / "cohort.parquet")
    w = cohort[["stay_id", "hadm_id", "vent_start", "vent_end"]].copy()
    w["t_lo"] = w.vent_start - pd.Timedelta(hours=24)
    w["t_hi"] = w.vent_end + pd.Timedelta(hours=24)

    parts = [drugs(w), urine(w), procedures(w), chart_extra(w)]
    cols = ["stay_id", "charttime", "feature", "valuenum"]
    ev = pd.concat([p[cols] for p in parts], ignore_index=True)

    b = w.set_index("stay_id")[["t_lo", "t_hi"]].reindex(ev.stay_id.values)
    ev = ev[(ev.charttime.values >= b.t_lo.values)
            & (ev.charttime.values <= b.t_hi.values)]
    ev.to_parquet(C.INTERIM / "extended_cohort.parquet", index=False)
    log(f"extended events: {len(ev):,}")
    log(ev.feature.value_counts().to_string())

    comorbidities(cohort).to_parquet(C.INTERIM / "comorbidities.parquet", index=False)
    log("done")
