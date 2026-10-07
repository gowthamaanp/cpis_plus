"""Report which required input files are present before running anything.

    python check_setup.py
"""
from cpis_plus import config as C

REQUIRED = {
    "MIMIC-IV hosp": [C.HOSP / f for f in ("admissions.csv.gz", "patients.csv.gz", "labevents.csv.gz",
                                           "microbiologyevents.csv.gz", "prescriptions.csv.gz",
                                           "diagnoses_icd.csv.gz")],
    "MIMIC-IV icu": [C.ICU / f for f in ("icustays.csv.gz", "chartevents.csv.gz", "procedureevents.csv.gz",
                                         "inputevents.csv.gz", "outputevents.csv.gz")],
    "MIMIC-CXR (CPIS radiograph component only)": [C.CXR_METADATA, C.CXR_CHEXPERT],
}

ok = True
for group, files in REQUIRED.items():
    print(f"\n{group}")
    for f in files:
        found = f.exists()
        ok &= found
        print(f"  [{'ok' if found else 'MISSING'}] {f}")
print("\nAll required inputs found." if ok else
      "\nSome inputs are missing. Set the environment variables described in cpis_plus/config.py.")
