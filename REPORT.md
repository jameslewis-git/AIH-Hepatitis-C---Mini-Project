# Mini Project Report
## Hepatitis C Detection and Staging using Machine Learning
### Subject: AI for Healthcare

---

## 1. Introduction

Hepatitis C is a blood-borne viral infection caused by the Hepatitis C Virus (HCV) that
primarily attacks the liver. If left undetected, it progresses silently through stages —
from a healthy carrier state to active **Hepatitis**, then **Fibrosis** (early scarring), and
finally **Cirrhosis** (severe, often irreversible scarring). Traditional staging relies on
liver biopsy, which is invasive, costly, and carries procedural risk.

This project proposes a non-invasive, machine-learning-based alternative: predicting a
patient's HCV stage directly from routine blood panel values. The system is built as an
interactive web application using **Streamlit**, backed by a **Random Forest / XGBoost**
classifier trained on the public **UCI HCV dataset** (615 patient records, 12 lab features).

## 2. Importance / Motivation

- **Early detection saves lives.** Hepatitis C often shows no symptoms until significant
  liver damage has occurred; a quick, low-cost staging estimate from blood work can flag
  high-risk patients for urgent specialist referral.
- **Reduces dependency on invasive biopsy.** Liver biopsy is the clinical gold standard but
  is invasive and not scalable for population-level screening. ML-based staging from blood
  markers (ALT, AST, GGT, Bilirubin, etc.) offers a cheap, repeatable alternative.
- **Supports healthcare AI education.** Demonstrates an end-to-end applied AI pipeline —
  data preprocessing, model selection, evaluation, and deployment via a usable web interface
  — directly relevant to the "AI for Healthcare" curriculum.
- **Resource-constrained settings.** In regions with limited access to biopsy or imaging,
  a blood-test-driven triage tool can meaningfully extend diagnostic reach.

## 3. Methodology (Summary)

1. **Data:** UCI HCV dataset — Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT,
   PROT, and target `Category` (Blood Donor / Suspect Blood Donor / Hepatitis / Fibrosis /
   Cirrhosis).
2. **Preprocessing:** Median imputation for missing lab values, label encoding for Sex and
   target class, standard scaling of numeric features.
3. **Modeling:** Trained both Random Forest and XGBoost classifiers; selected the model with
   the higher macro-F1 score on a held-out 20% test split (stratified) to account for class
   imbalance.
4. **Deployment:** Wrapped the trained pipeline (imputer + scaler + model) in a Streamlit
   web app with a Home page, an input form (Predict page), and a Model Info page showing
   predicted stage and class probabilities.

## 4. Conclusion

This project demonstrates a complete, working pipeline for applying machine learning to a
real healthcare staging problem: from a public clinical dataset, through preprocessing and
model selection, to a usable web interface that outputs an interpretable prediction with
confidence scores. While the current model is trained on a relatively small, imbalanced
dataset and is **not fit for real clinical deployment**, it illustrates how AI can augment —
not replace — clinical judgment by surfacing early warning signals from inexpensive,
routinely-collected blood test data. Future improvements could include larger multi-center
datasets, SHAP-based explainability for each prediction, and calibrated probability outputs
for clinical-grade confidence reporting.

---
*Disclaimer: This is an academic demonstration tool only and must not be used for real
medical diagnosis.*
