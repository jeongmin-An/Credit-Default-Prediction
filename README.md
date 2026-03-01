# credit-default-prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Imbalanced](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---

## 🔗 Quick Links (Repo Files)
- 📓 Notebook (end-to-end): [`Project_notebook.ipynb`](./Project_notebook.ipynb)
- 📄 Final report (PDF): [`Credit_Default_Prediction.pdf`](./Credit_Default_Prediction.pdf)
- 🧪 Data prep script: [`data.py`](./data.py)
- 🖥️ Streamlit app: [`streamlit_app/`](./streamlit_app/)

---

## 🧭 Project Overview
Credit risk teams need early signals of potential delinquency to reduce losses and support consistent lending decisions.  
This project builds a **binary classification model** to predict whether a borrower will become **90+ days delinquent within 2 years** (`SeriousDlqin2yrs`) using the Kaggle *Give Me Some Credit* dataset.

---

## 🎯 Project Goals
- Predict delinquency risk early to support proactive risk monitoring  
- Handle **class imbalance** and improve detection of delinquent borrowers (minority class)  
- Compare multiple ML models using a consistent evaluation framework (**ROC-AUC**)  
- Provide interpretable insights into key risk drivers (feature importance)

---

## 🗂 Dataset
- **Source:** Kaggle — “Give Me Some Credit”
- **Raw file:** `GiveMeSomeCredit/cs-training.csv`
- **Size:** ~150,000 rows (before cleaning)
- **Task:** Binary classification (`SeriousDlqin2yrs`)
- **Feature examples:**
  - Revolving credit utilization, debt ratio, monthly income  
  - Late payment counts (30–59 / 60–89 / 90+ days)  
  - Open credit lines, real estate loans, dependents  

---

## 🧹 Data Preprocessing
The notebook applies targeted cleaning to improve model stability and avoid biased deletion.

### 1) Missing Values
- Missingness concentrated in **MonthlyIncome** and **NumberOfDependents**
- Implemented in notebook:
  - Remove invalid ages (`age == min(age)`; dataset contains `age=0`)
  - Impute `MonthlyIncome` by **age-group median** (`groupby('age') → median`)
  - Drop any remaining missing `MonthlyIncome`
  - Impute `NumberOfDependents` using **median**

### 2) Outliers
- Cap extreme utilization:  
  - `RevolvingUtilizationOfUnsecuredLines <= 1.5` *(allow up to 150% utilization)*
- IQR-based filtering applied to:
  - `age`, `MonthlyIncome`, `DebtRatio`, `NumberOfDependents`

### 3) Cleaned Output
- Exported cleaned dataset:
  - `GiveMeSomeCredit/GiveMeSomeCredit-cleaned.csv`

---

## 🔎 Exploratory Data Analysis (EDA) Highlights
- Visualized **target imbalance** (delinquency is the minority class)
- Missingness inspection (income / dependents)
- Distribution checks (with reasonable caps for visualization):
  - `age`, `MonthlyIncome`, utilization, delinquency-count features
- Correlation heatmap to inspect feature relationships and redundancy

---

## 🏗 Modeling Framework
### Models Compared
- Logistic Regression (LR)  
- Random Forest Classifier (RF)  
- Gradient Boosting Classifier (GB)  
- MLP Classifier (MLP)

### Evaluation Setup
- Train/test split: **80/20** with `stratify=y` (random_state=42)
- Primary metric: **ROC-AUC** (robust under class imbalance)
- Class imbalance handling: **SMOTE inside an imblearn Pipeline**
  - LR/MLP: `StandardScaler → SMOTE → Model`
  - RF/GB: `SMOTE → Model`

---

## 🧪 Hyperparameter Tuning
- Tuning method: **GridSearchCV**
- Cross-validation: **5-fold StratifiedKFold** (shuffle=True, random_state=42)
- Scoring: **roc_auc**
- n_jobs = -1 (parallel)

Best CV params (from notebook):
- **LR:** `C=0.1`, `penalty='l2'`, `class_weight=None`
- **RF:** `n_estimators=200`, `max_depth=10`, `min_samples_split=2`, `class_weight='balanced_subsample'`
- **GB:** `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`
- **MLP:** `hidden_layer_sizes=(50,)`, `alpha=1e-05`, `learning_rate_init=0.0001`

---

## 📈 Key Results
**Best-performing model: MLP Classifier (SMOTE pipeline)**

| Model | Best CV AUC | Test AUC |
|---|---:|---:|
| MLP Classifier | 0.851 | 0.840 |
| Random Forest | 0.829 | 0.822 |
| Gradient Boosting | 0.827 | 0.816 |
| Logistic Regression | 0.800 | 0.795 |

### Minority-Class Performance (Delinquency = 1)
From the notebook’s test-set report for the best model (MLP):
- **Recall (class 1): 0.68**  → risky borrowers are captured relatively well  
- **Precision (class 1): 0.22** → higher false positives (risk teams often accept this trade-off)

Confusion matrix (test set):
- TN=15274, FP=3285  
- FN=425,  TP=920  

### 🔍 Interpretability (Permutation Importance)
Permutation importance (AUC drop) shows the strongest risk drivers:

- `RevolvingUtilizationOfUnsecuredLines`  (0.0715 ± 0.0031)  
- `NumberOfTime30-59DaysPastDueNotWorse`  (0.0495 ± 0.0010)  
- `NumberOfTimes90DaysLate`               (0.0391 ± 0.0011)  
- `MonthlyIncome`                         (0.0218 ± 0.0037)  
- `NumberOfTime60-89DaysPastDueNotWorse`  (0.0111 ± 0.0012)  
- `age`                                   (0.0102 ± 0.0020)

---

## 🖥️ Streamlit App (Local Demo)
This repository includes a Streamlit demo app under `streamlit_app/`.

> Note: GitHub does not run Streamlit apps. Run it locally.

### Run locally
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn streamlit
streamlit run streamlit_app/app.py
