# Evaluation design substantially influences apparent performance of a ventilator-associated pneumonia prediction model

Code for the CPIS+ model and for Table 3 (Systematic Evaluation Protocol)

CPIS+ is a LightGBM classifier on 339 features, predicting VAP onset within 24 h
from 12-hourly landmarks of MIMIC-IV ventilation episodes. Table 3 reports its
test AUROC when only the evaluation and cohort construction changes. This repository builds
the cohort and features, fits the model and produces every row of Table 3. 