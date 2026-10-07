"""The classical Clinical Pulmonary Infection Score (CPIS).

This is the baseline CPIS+ has to beat, and its six components double as the
clinically-grounded feature set for the tabular branch. Implemented per Pugin
et al. 1991 with the Singh 2000 modification (score 0-12, >6 suggests
pneumonia).

Every component is computed from data available *at or before* a given
prediction time, so the same code serves both the baseline and the features.
"""
import numpy as np
import pandas as pd

PURULENT = ("PURULENT", "YELLOW", "GREEN", "TAN", "BROWN", "BLOOD")
NON_PURULENT = ("CLEAR", "WHITE", "THIN", "TRANSPARENT")


def score_temperature(t_c):
    """0: 36.5-38.4 | 1: 38.5-38.9 or 36.0-36.4 | 2: >=39.0 or <=35.9"""
    if pd.isna(t_c):
        return np.nan
    if 36.5 <= t_c <= 38.4:
        return 0
    if 38.5 <= t_c <= 38.9 or 36.0 <= t_c <= 36.4:
        return 1
    return 2


def score_wbc(wbc, bands_pct=np.nan):
    """0: 4-11 k/uL | 1: <4 or >11 | +1 if band forms >= 50%"""
    if pd.isna(wbc):
        return np.nan
    s = 0 if 4 <= wbc <= 11 else 1
    if not pd.isna(bands_pct) and bands_pct >= 50:
        s += 1
    return s


def score_secretions(amount=None, color=None, consistency=None):
    """0: none/rare | 1: non-purulent | 2: purulent.

    MIMIC charts secretions as separate amount/colour/consistency strings; we
    treat any purulent descriptor as 2, any charted non-purulent secretion as 1.
    """
    fields = [str(x).upper() for x in (amount, color, consistency)
              if x is not None and not pd.isna(x)]
    if not fields:
        return np.nan
    blob = " ".join(fields)
    if any(k in blob for k in ("NONE", "NO SECRETION")):
        return 0
    if any(k in blob for k in PURULENT):
        return 2
    if any(k in blob for k in NON_PURULENT):
        return 1
    return 1


def score_oxygenation(pao2, fio2, ards=False):
    """0: PaO2/FiO2 > 240 or ARDS | 2: <= 240"""
    if pd.isna(pao2) or pd.isna(fio2) or fio2 <= 0:
        return np.nan
    f = fio2 / 100 if fio2 > 1 else fio2
    pf = pao2 / f
    if ards or pf > 240:
        return 0
    return 2


def score_radiograph(infiltrate=None, diffuse=None):
    """0: no infiltrate | 1: diffuse/patchy infiltrate | 2: localised infiltrate.

    `infiltrate` / `diffuse` come from the CheXpert labeller applied to the
    report (Consolidation / Lung Opacity / Pneumonia). NaN when no film is
    available in the window.
    """
    if infiltrate is None or pd.isna(infiltrate):
        return np.nan
    if not infiltrate:
        return 0
    return 1 if diffuse else 2


def score_culture(positive, same_organism_on_gram=False):
    """0: no/light growth | 1: moderate-heavy growth | +1 if gram stain agrees."""
    if pd.isna(positive):
        return np.nan
    if not positive:
        return 0
    return 2 if same_organism_on_gram else 1


COMPONENTS = ["cpis_temp", "cpis_wbc", "cpis_secretions", "cpis_oxygenation",
              "cpis_radiograph", "cpis_culture"]


def total(df: pd.DataFrame, min_components: int = 4) -> pd.Series:
    """Sum the six components.

    Missing components are treated as 0 (the convention in retrospective CPIS
    work - a component that was never charted was not abnormal enough to
    document), but rows with fewer than `min_components` observed are returned
    as NaN rather than silently scored low.
    """
    comp = df[COMPONENTS]
    n_obs = comp.notna().sum(axis=1)
    s = comp.fillna(0).sum(axis=1)
    return s.where(n_obs >= min_components)
