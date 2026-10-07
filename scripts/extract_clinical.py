"""Filter MIMIC-IV chartevents and labevents to the cohort's ventilation windows
and the items of interest (read in chunks).

Outputs (data/interim/): chartevents_cohort.parquet, labevents_cohort.parquet
"""
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def gz_stream(path):
    """Decompress with pigz on a separate core.

    gzip decompression of these files is single-threaded and was the dominant
    cost; handing it to pigz lets pandas parse while pigz inflates.
    """
    p = subprocess.Popen(["unpigz", "-c", str(path)], stdout=subprocess.PIPE,
                         bufsize=1 << 20)
    try:
        yield p.stdout
    finally:
        p.stdout.close()
        p.wait()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C

CHUNK = 5_000_000
# how far either side of the ventilation episode to keep events
PRE_H, POST_H = 24, 24

CHART_ITEMS = {
    223761: "temp_f", 223762: "temp_c",
    224369: "sputum_consistency", 224370: "sputum_color",
    224372: "sputum_source", 224373: "sputum_amount",
    223835: "fio2", 220224: "pao2", 220277: "spo2",
    220339: "peep", 224700: "peep_total",
    220045: "heart_rate", 220210: "resp_rate", 224690: "resp_rate_total",
    220179: "sbp", 220181: "map", 220050: "abp_sys", 220052: "abp_mean",
    226732: "o2_device", 223849: "vent_mode",
    224685: "tidal_volume", 224687: "minute_volume",
    223900: "gcs_verbal", 223901: "gcs_motor", 220739: "gcs_eye",
}

LAB_ITEMS = {
    51301: "wbc", 51144: "bands", 51256: "neutrophils",
    50821: "pao2_lab", 50816: "fio2_lab", 50818: "pco2", 50820: "ph",
    50813: "lactate", 50912: "creatinine", 51265: "platelet",
    51222: "hemoglobin", 50889: "crp", 50885: "bilirubin", 50971: "potassium",
    50983: "sodium", 51006: "bun", 50931: "glucose",
}


def log(msg):
    print(f"[clin] {msg}", flush=True)


def windows(cohort):
    """stay_id/hadm_id -> allowed time window."""
    w = cohort[["stay_id", "hadm_id", "subject_id", "vent_start", "vent_end"]].copy()
    w["t_lo"] = w.vent_start - pd.Timedelta(hours=PRE_H)
    w["t_hi"] = w.vent_end + pd.Timedelta(hours=POST_H)
    return w


def extract_chartevents(w):
    stays = set(w.stay_id)
    bounds = w.set_index("stay_id")[["t_lo", "t_hi"]]
    keep = []
    n_read = 0
    # NB: charttime is read as a string and only parsed *after* filtering.
    # Parsing 430M timestamps costs far more than parsing the ~1M we keep.
    with gz_stream(C.ICU / "chartevents.csv.gz") as fh:
        reader = pd.read_csv(
            fh,
            usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid",
                     "value", "valuenum", "valueuom"],
            dtype={"stay_id": "float64", "itemid": "int32", "value": "object",
                   "valuenum": "float32", "charttime": "object"},
            chunksize=CHUNK, low_memory=False)
        for i, ch in enumerate(reader, 1):
            n_read += len(ch)
            ch = ch[ch.itemid.isin(CHART_ITEMS) & ch.stay_id.isin(stays)]
            if not ch.empty:
                ch = ch.copy()
                ch["charttime"] = pd.to_datetime(ch.charttime)
                b = bounds.reindex(ch.stay_id.values)
                m = ((ch.charttime.values >= b.t_lo.values)
                     & (ch.charttime.values <= b.t_hi.values))
                ch = ch[m]
                if not ch.empty:
                    keep.append(ch)
            if i % 10 == 0:
                log(f"  chartevents chunk {i}: {n_read:,} rows read, "
                    f"{sum(len(k) for k in keep):,} kept")
    out = pd.concat(keep, ignore_index=True)
    out["feature"] = out.itemid.map(CHART_ITEMS)
    return out


def extract_labevents(w):
    hadms = set(w.hadm_id.dropna())
    # labs are hadm-scoped; widen the window to the union across the admission
    b = (w.groupby("hadm_id").agg(t_lo=("t_lo", "min"), t_hi=("t_hi", "max")))
    keep = []
    n_read = 0
    with gz_stream(C.HOSP / "labevents.csv.gz") as fh:
        reader = pd.read_csv(
            fh,
            usecols=["subject_id", "hadm_id", "charttime", "itemid", "value",
                     "valuenum", "valueuom", "flag"],
            dtype={"hadm_id": "float64", "itemid": "int32", "value": "object",
                   "valuenum": "float32", "charttime": "object"},
            chunksize=CHUNK, low_memory=False)
        for i, ch in enumerate(reader, 1):
            n_read += len(ch)
            ch = ch[ch.itemid.isin(LAB_ITEMS) & ch.hadm_id.isin(hadms)]
            if not ch.empty:
                ch = ch.copy()
                ch["charttime"] = pd.to_datetime(ch.charttime)
                bb = b.reindex(ch.hadm_id.values)
                m = ((ch.charttime.values >= bb.t_lo.values)
                     & (ch.charttime.values <= bb.t_hi.values))
                ch = ch[m]
                if not ch.empty:
                    keep.append(ch)
            if i % 10 == 0:
                log(f"  labevents chunk {i}: {n_read:,} rows read, "
                    f"{sum(len(k) for k in keep):,} kept")
    out = pd.concat(keep, ignore_index=True)
    out["feature"] = out.itemid.map(LAB_ITEMS)
    return out


if __name__ == "__main__":
    cohort = pd.read_parquet(C.COHORT / "cohort.parquet")
    w = windows(cohort)
    log(f"cohort: {len(w):,} episodes")

    ce = extract_chartevents(w)
    ce.to_parquet(C.INTERIM / "chartevents_cohort.parquet", index=False)
    log(f"chartevents kept: {len(ce):,}")
    log(ce.feature.value_counts().to_string())

    le = extract_labevents(w)
    le.to_parquet(C.INTERIM / "labevents_cohort.parquet", index=False)
    log(f"labevents kept: {len(le):,}")
    log(le.feature.value_counts().to_string())
    log("done")
