# census-data-analysis-modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-purple)
![KMeans](https://img.shields.io/badge/Segmentation-KMeans-orange)

> **Income Prediction & Customer Segmentation using U.S. Census Bureau Data (1994-1995)**

---

## 📌 Project Overview

This project builds two analytical models for a retail business client to support targeted marketing:

| Model | Name | Goal |
|---|---|---|
| 🎯 Classification | **Income Finder** | Predict whether an individual earns above or below $50,000/year |
| 👥 Segmentation | **Customer Profiler** | Group the population into distinct customer segments |

---

## 📁 Project Structure
```
census-data-analysis-modeling/
├── Data/
│   ├── census-bureau.columns
│   ├── census-bureau.data
│   ├── full_data.csv
│   └── df_with_predictions.csv
├── Images/
│   ├── cluster_scatter.png
│   └── feature_importance.png
├── BASELINE_LOGISTIC_REGRESSION.ipynb
├── DATA_EXPLORATION.ipynb
├── LightGBM.ipynb
├── Segmentation.ipynb
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** U.S. Census Bureau — Current Population Surveys (1994-1995)
- **Size:** 199,523 records × 43 columns
- **Target:** Binary income label — below $50K (0) or above $50K (1)
- **Challenge:** Severe class imbalance — 94% below $50K, 6% above $50K

> ⚠️ Raw data files are not included due to file size. Place `census-bureau.data` and `census-bureau.columns` in the `/Data` folder before running.

---

## ⚙️ Setup
```bash
git clone https://github.com/nikhil02jk/census-data-analysis-modeling.git
cd census-data-analysis-modeling
pip install -r requirements.txt
```

---

## 🚀 Running the Code

> ⚠️ Run notebooks in this order:

**1. DATA_EXPLORATION.ipynb**
Explores the dataset, missing values, class imbalance and key feature distributions.

**2. BASELINE_LOGISTIC_REGRESSION.ipynb**
Trains Logistic Regression baseline with `class_weight='balanced'`.

**3. LightGBM.ipynb**
Trains and tunes the final LightGBM model using Optuna. Saves `df_with_predictions.csv`.

**4. Segmentation.ipynb**
> ⚠️ Requires `df_with_predictions.csv` from step 3.
Runs model-aligned K-Means segmentation and profiles customer segments.

---

## 📈 Results

### Classification

| Metric | Logistic Regression | LightGBM Optuna Tuned |
|---|---|---|
| ROC-AUC | 0.927 | **0.953** |
| PR-AUC | 0.542 | **0.686** |
| Precision (≥50K) | 0.498 | **0.649** |
| Recall (≥50K) | 0.558 | 0.604 |
| F1 (≥50K) | 0.526 | **0.626** |

### Segmentation

| Segment | Size | High Earner % | Priority |
|---|---|---|---|
| ⭐ Premium Customers | 445 | 89.4% | Premium campaigns |
| ✅ Core Customers | 48,077 | 22.6% | Primary campaigns |
| 🟡 Emerging Customers | 37,837 | 1.6% | Aspirational |
| 🟡 Passive Customers | 54,874 | 0.9% | Household |
| ❌ Non-Customers | 54,934 | 0.0% | Exclude |

---

## 🔍 Key Findings

- **Occupation** is the strongest income predictor — 25.1% of model gain
- **High earners** work nearly year-round — 48.1 weeks vs 21.5 weeks for low earners
- **Investment income** gap is 28x — $6,191 vs $225 average
- **Model-aligned segmentation** discovered a Premium Customers segment with 89.4% high earner rate — 4x better than standard K-Means
- **Optuna tuning** improved PR-AUC by 1.6% and precision by 7% over baseline LightGBM

---

## 📝 Notes

- All random seeds set to **42** for reproducibility
- Optuna runs **50 trials** — takes 3-5 mins on CPU, faster on GPU
- Developed on **Google Colab with G4 GPU**
- If running locally update file paths from `/content/drive/MyDrive/...` to `/Data/...`
