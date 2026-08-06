# Loan Default Risk Predictor for Micro-Lenders & POS Agents (Nigeria)

A tabular machine learning model that predicts the likelihood of loan default, built for
Nigerian micro-lenders and POS (Point-of-Sale) agents who need a fast, explainable
repayment-risk check before disbursing small, short-tenor loans.

**[▶ Watch the 3-minute demo video](https://www.linkedin.com/posts/latifahusainibashir_3mtt-nextgen-machinelearning-ugcPost-7490940128937455616-QV1U/?utm_source=share&utm_medium=member_desktop&rcm=ACoAACfpDowB952onpiDYQ9xaOzua6pOcyQuZB8)** 

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgments](#acknowledgments)

---

## Problem Statement

Across Nigeria, POS agents and micro-lenders extend small, short-term loans to informal
traders, artisans, and small business owners  often with little to no formal credit history
to underwrite against. A wrong call is costly in either direction: lending to a high-risk
borrower risks non-repayment, while rejecting a genuinely low-risk borrower loses income and
excludes someone who may be perfectly creditworthy.

This project builds a lightweight, explainable risk-scoring model that a micro-lender or POS
agent could realistically use at the point of a lending decision, fed either a single
applicant's details (manual input) or a batch of applicants (CSV upload).

## Solution Overview

- **Input:** borrower demographic, business, POS-transaction, and loan-request data via
  manual entry or CSV upload
- **Output:** a default-risk probability (0–100%), a risk band (Low / Medium / High), and
  the key factors behind the score
- **Models compared:** Logistic Regression, Random Forest, and XGBoost, each evaluated under
  three class-imbalance strategies (baseline, class weighting, SMOTE), nine variants total
- **Deployed model:** XGBoost with class weighting (see [Results](#results) for why)


## Data

**No public, row-level loan-default dataset exists for the Nigerian micro-lending / POS-agent
market** that kind of individual borrower data is proprietary to lenders and protected under
banking privacy regulation. This was confirmed by checking the Central Bank of Nigeria's
public data (Statistical Bulletins, Financial Stability & Microfinance Reports), which publish
only **aggregate** statistics, never row-level borrower records.

**Approach:** a documented **synthetic dataset** of 20,000 borrower records, generated to
reflect the Nigerian micro-lending / POS-agent context specifically including features not
present in generic global credit datasets:

| Category | Features |
|---|---|
| Demographics | age, gender, state, urban/rural |
| Business profile | sector, years in business, monthly income |
| POS activity | monthly transaction count & volume |
| Loan details | amount, tenor, interest rate, existing loans, collateral |
| Nigeria-specific context | loan channel (POS Agent / Branch / Mobile App / Cooperative), trade/market association membership, mobile money usage, SIM tenure |
| Behavioral | repayment history score |
| Target | `defaulted` (0/1) |

The `defaulted` outcome is **not random**, it's generated from a transparent logistic
combination of realistic risk drivers (debt-to-income ratio, repayment history, collateral,
existing loan burden, business tenure, trade association membership, etc.) plus noise, so the
dataset has genuine, learnable structure. Full generation logic and reasoning is documented in
`generate_synthetic_loan_data.py` and `data_source_justification.md`.

**Known limitation, stated plainly:** this is synthetic data. Model performance and feature
importances reflect the logic used to generate it, not verified real-world Nigerian lending
outcomes, an appropriate approach for a technique-focused capstone where no real dataset can
exist publicly, but not a substitute for validation on real lender data.

## Methodology

1. **Preprocessing** : numeric features median-imputed and scaled; categorical features
   mode-imputed and one-hot encoded; binary flags passed through. Wrapped in a
   `ColumnTransformer` inside a single `Pipeline` per model, so preprocessing is applied
   identically at train and inference time.
2. **Class imbalance handling** : the dataset has a ~15% default rate. Three strategies were
   compared rather than assumed:
   - **Baseline**: no adjustment
   - **Class weighting**: `class_weight='balanced'` / `scale_pos_weight`
   - **SMOTE**: synthetic minority oversampling, applied inside an `imblearn` pipeline to
     avoid test-set leakage
3. **Models**: Logistic Regression, Random Forest, XGBoost, each run under all three
   strategies above (9 total variants), evaluated on a held-out, stratified test set.
4. **Explainability** : feature importance for every model; risk-scoring functions
   return a plain-language risk band alongside the probability.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.864 | 0.762 | 0.176 | 0.286 | 0.795 |
| Random Forest | 0.877 | 0.786 | 0.280 | 0.413 | 0.797 |
| XGBoost | 0.878 | 0.726 | 0.338 | 0.461 | 0.810 |
| Random Forest (Class Weight) | 0.873 | 0.789 | 0.243 | 0.371 | 0.804 |
| **XGBoost (Class Weight)** | **0.816** | **0.434** | **0.621** | **0.511** | **0.807** |
| Logistic Regression (Class Weight) | 0.715 | 0.316 | 0.723 | 0.440 | 0.791 |
| Random Forest (SMOTE) | 0.876 | 0.693 | 0.351 | 0.466 | 0.801 |
| Logistic Regression (SMOTE) | 0.723 | 0.320 | 0.706 | 0.440 | 0.789 |
| XGBoost (SMOTE) | 0.878 | 0.706 | 0.361 | 0.478 | 0.817 |

**Deployed model: XGBoost (Class Weight).** It has the best F1-score (0.511) among all nine
variants, the metric that jointly rewards recall and precision. Logistic Regression (Class
Weight) achieves higher raw recall (72.3% vs 62.1%), but at a steep cost: it wrongly flags
**28.7%** of genuinely good borrowers as risky, versus **14.8%** for XGBoost. For a real
lender, rejecting roughly 1 in 4 legitimate customers to catch slightly more defaulters would
likely undermine adoption and work against the financial-inclusion goal this project targets,
so XGBoost's more balanced trade-off was chosen as the deployed model. This trade-off is a
business judgment as much as a technical one, and is documented transparently rather than
hidden behind a single headline metric.

*(See `results/results_summary.md` for full classification reports, confusion matrices, and
per-model feature importance.)*

## How to Run

**Notebook (Google Colab, recommended):**
1. Upload `loan_default_risk_predictor.ipynb` and `synthetic_loan_data.csv` to Colab
2. Run all cells top to bottom 


## Limitations & Future Work

- **Synthetic data**: results reflect the generation logic, not verified real-world outcomes.
  Next step is validating against real (anonymized, consented) lender repayment data.
- **Risk thresholds** (Low <30%, Medium 30–60%, High >60%) are reasonable round numbers, not
  derived from a specific lender's cost-benefit analysis, production use would calibrate
  these to a real risk appetite.
- **No cross-validation**  a single stratified train/test split was used; k-fold CV would
  give more robust metric estimates at the cost of runtime.
- **Individual-level explainability** (e.g. SHAP per applicant) is a natural next addition,
  global feature importance is implemented; per-prediction "why" explanations are not yet.

## Acknowledgments

- Central Bank of Nigeria (CBN): public reports used to ground synthetic data ranges
- Built as a capstone project for 3MTT NextGen/ AI & ML Track
