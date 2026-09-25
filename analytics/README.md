# Module 2 - Analytics Pipeline (`/analytics`)

End-to-end analytics on the **Titanic** dataset: profiling, cleaning, a visual
data story, then a leakage-safe predictive-modeling pipeline (three classifiers,
imbalance handling, hyperparameter tuning), a regression side-task, and a saved,
reloadable pipeline.

## Structure

One cohesive pipeline across two ordered notebooks that share a single committed CSV:

| File | Role |
|------|------|
| `01_eda.ipynb` | **Part A.** The one and only raw load, profiling, cleaning, outliers/skew, bivariate + correlation heatmap, the 4-chart data story, and the z-score sanity check. Saves the raw load to `titanic.csv`. |
| `02_modeling.ipynb` | **Part B.** Reads the same committed `titanic.csv` (no second network load) and runs the full modeling pipeline. |
| `titanic.csv` | The committed offline fallback (the raw load), so grading works via `pd.read_csv("titanic.csv")` with no internet. |
| `best_pipeline.joblib` | The saved best full pipeline (preprocessing + estimator), usable on raw input. |
| `fig_*.png` | Saved chart artifacts (supporting only; all interpretations are text). |

## How to run

From the repository root with the project venv active, run the notebooks **in order**:

```bash
jupyter nbconvert --to notebook --execute --inplace analytics/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace analytics/02_modeling.ipynb
```

Or open them in Jupyter and run all cells top to bottom (01 first). `01_eda.ipynb`
produces `titanic.csv`; `02_modeling.ipynb` consumes it. If `titanic.csv` is
already committed, `01_eda.ipynb` loads it directly and needs no network.

## Single-load design

The raw dataset is loaded **exactly once** (in `01_eda.ipynb`, via
`sns.load_dataset("titanic")`, immediately cached to `titanic.csv`). Every later
step - EDA, the data story, and the entire modeling pipeline - continues from that
same DataFrame or its committed CSV. `02_modeling.ipynb` never issues a second
`sns.load_dataset` call.

## Key results (all interpretations live in the notebook markdown)

**Missing values (measured %) -> action (Task 2 threshold rule):**
- `deck` 77.22% -> **drop column** (imputation unreliable)
- `age` 19.87% -> **impute** (median; in the 5-30% band)
- `embarked` / `embark_town` 0.22% -> **drop those rows** (< 5%)

**Outliers / skew (Task 3):** age has 65 IQR outliers (near-symmetric, skew 0.51);
fare has 114 IQR outliers and is strongly right-skewed - **mean 32.10 > median
14.45 > mode 8.05**, skew 4.80.

**Survival rates (Task 4):** female 74.04% vs male 18.89%; 1st 62.62% / 2nd 47.28%
/ 3rd 24.24%; sharpest interaction: 1st-class women 96.74%, 3rd-class men 13.54%.
Two strongest correlations: **pclass~fare = -0.548**, **sibsp~parch = +0.415**.

**Classifier comparison (Task 10/14):**

| model | accuracy | precision | recall | f1 | auc |
|---|---|---|---|---|---|
| LogisticRegression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| DecisionTree | 0.7598 | 0.7407 | 0.5797 | 0.6504 | 0.7891 |
| RandomForest | 0.8045 | 0.7742 | 0.6957 | 0.7328 | 0.8307 |

**Imbalance (Task 11, LogReg):** baseline F1 0.7244; class_weight F1 0.7552
(recall 0.7826); SMOTE F1 0.7606 (best, recall 0.7826). SMOTE resamples the
training fold only (inside an imblearn Pipeline).

**Tuning (Task 12):** best RF = `max_depth=4, max_features='sqrt',
n_estimators=100`; CV F1 0.7483; **OOB score 0.8146** (RF built with
`oob_score=True`).

**Regression (Task 13):** predict `fare` - MAE 20.90, RMSE 30.53, R2 0.3975,
Adjusted R2 0.3729; residual plot shows **clear heteroscedasticity** (funnel shape).

**Recommendation (Task 14):** deploy **Random Forest** (best F1 0.7328 and recall
0.6957, tied-best accuracy); keep Logistic Regression as an interpretable fallback
(best AUC 0.8437). Classification and regression metrics are reported as **separate
metric groups** (different scales, not directly comparable).

**Saved artifact (Task 15):** `best_pipeline.joblib` is the complete fitted
pipeline (ColumnTransformer preprocessing + estimator). The notebook reloads it
with `joblib.load` and confirms it predicts correctly on **raw, unpreprocessed**
input rows.

## Leakage safety

The train/test split is stratified and done **before** any preprocessing. All
imputers, encoders, and scalers live in a `ColumnTransformer`/`Pipeline` and are
fit on the **training split only**, then applied transform-only to the test split.
SMOTE is applied only to the training fold.
