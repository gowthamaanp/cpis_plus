"""Paths and constants for CPIS+.

All data locations come from environment variables; no data is distributed
with this repository.

  CPIS_DATA_DIR   generated outputs                  (default: <repo>/data)
  CPIS_RAW_DIR    root of the raw datasets           (default: <DATA_DIR>/raw)
  MIMICIV_DIR     MIMIC-IV v3.1 (hosp/, icu/)        (default: <RAW>/mimiciv/3.1)
  MIMIC_CXR_DIR   folder holding the MIMIC-CXR(-JPG) metadata and CheXpert label
                  files (used only for the CPIS radiograph component)
"""
import os
from pathlib import Path


def _env(name, default):
    return Path(os.environ.get(name, default))


PROJ = _env("CPIS_PROJECT_ROOT", Path(__file__).resolve().parents[1])
DATA = _env("CPIS_DATA_DIR", PROJ / "data")
RAW = _env("CPIS_RAW_DIR", DATA / "raw")

MIMICIV = _env("MIMICIV_DIR", RAW / "mimiciv" / "3.1")
HOSP = MIMICIV / "hosp"
ICU = MIMICIV / "icu"

CXR_ROOT = _env("MIMIC_CXR_DIR", RAW / "mimic-cxr" / "2.0.0")
CXR_METADATA = CXR_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz"
CXR_CHEXPERT = CXR_ROOT / "mimic-cxr-2.0.0-chexpert.csv.gz"

INTERIM = DATA / "interim"
COHORT = DATA / "cohort"
for _p in (INTERIM, COHORT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------- cohort definition
MIN_AGE = 18
MIN_VENT_HOURS = 48          # mechanical ventilation > 48h
VAP_ONSET_MIN_HOURS = 48     # earliest a VAP event can be attributed to the episode
PREDICTION_HORIZONS = (24,)  # hours: VAP onset must fall within this window

# ------------------------------------------------------------- itemids
# procedureevents (have starttime/endtime -> ventilation episodes)
IT_INVASIVE_VENT = 225792
IT_TRACH_PERC = 225448
IT_TRACH_OPEN = 226237
IT_EXTUBATION = 227194
IT_UNPLANNED_EXTUB = (225468, 225477)

# chartevents - CPIS components + supporting vitals
IT_TEMP_F = 223761
IT_TEMP_C = 223762
IT_PAO2 = 220224            # Arterial O2 pressure (chart-side copy of ABG)
IT_SPO2 = 220277
IT_FIO2 = 223835            # Inspired O2 Fraction
IT_PEEP = 220339
IT_PEEP_TOTAL = 224700
IT_VENT_MODE = (223849, 229314)
IT_O2_DEVICE = 226732

IT_SPUTUM = {
    "consistency": 224369,
    "color": 224370,
    "source": 224372,
    "amount": 224373,
}

CHART_ITEMS_EXTRA = {
    220045: "heart_rate",
    220210: "resp_rate",
    220179: "sbp",
    220181: "map",
    224690: "resp_rate_total",
    224685: "tidal_volume_obs",
    220293: "minute_volume",     # placeholder; resolved at runtime
}

# labevents itemids (hosp/d_labitems)
LAB_ITEMS = {
    51301: "wbc",
    51300: "wbc_alt",
    50820: "ph",
    50821: "po2",
    50816: "fio2_lab",
    50818: "pco2",
    50912: "creatinine",
    50813: "lactate",
    51221: "hematocrit",
    51222: "hemoglobin",
    51265: "platelet",
    50889: "crp",
}

# --------------------------------------------------------- VAP labelling
# ICD codes that explicitly name ventilator-associated pneumonia
ICD_VAP = {"99731", "J95851"}
# broader pneumonia codes, used only for the *exclusion* of pneumonia-on-admission
ICD_PNEUMONIA_PREFIX = ("480", "481", "482", "483", "484", "485", "486", "507",
                        "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J69")

# respiratory specimen types in microbiologyevents used for the clinical proxy label.
# Pleural fluid is deliberately excluded: it indicates empyema/effusion, not the
# lower-airway sampling that defines VAP.
RESP_SPECIMENS = (
    "SPUTUM", "BRONCHOALVEOLAR LAVAGE", "BRONCHIAL WASHINGS", "TRACHEAL ASPIRATE",
    "MINI-BAL", "BRONCHIAL BRUSH",
)

# Organisms recovered from the airway that do not cause VAP (colonisers / normal
# flora). Excluding these is what separates infection from colonisation, and is
# the single biggest driver of label quality.
NON_PATHOGENS = (
    "YEAST", "CANDIDA", "NORMAL", "FLORA", "COAG NEG", "COAGULASE NEGATIVE",
    "CORYNEBACTERIUM", "LACTOBACILLUS", "MICROCOCCUS", "BACILLUS",
    "VIRIDANS", "PROPIONIBACTERIUM", "CUTIBACTERIUM", "DIPHTHEROID",
    "CANCELLED", "NO GROWTH", "MIXED BACTERIAL", "RARE",
)

# a "new" antibiotic must not already have been running in the preceding window
NEW_ABX_LOOKBACK_H = 72
ABX_WINDOW_H = 48        # abx must start within +/- this of the culture
