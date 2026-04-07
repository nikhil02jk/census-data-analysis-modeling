census-data-analysis-modeling

Income Prediction & Customer Segmentation using U.S. Census Bureau Data (1994-1995)


Project Overview
This project builds two analytical models for a retail business client to support targeted marketing:
ModelNameGoal🎯 ClassificationIncome FinderPredict whether an individual earns above or below $50,000/year👥 SegmentationCustomer ProfilerGroup the population into distinct customer segments

Project Structure
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

Dataset

Source: U.S. Census Bureau — Current Population Surveys (1994-1995)
Size: 199,523 records × 43 columns
Target: Binary income label — below $50K (0) or above $50K (1)
Challenge: Severe class imbalance — 94% below $50K, 6% above $50K


⚠️ Raw data files are not included due to file size. Place census-bureau.data and census-bureau.columns in the /Data folder before running.


Setup
bashgit clone https://github.com/YOUR_USERNAME/census-data-analysis-modeling.git
cd census-data-analysis-modeling
pip install -r requirements.txt

Running the Code
⚠️ Run notebooks in this order:
1. DATA_EXPLORATION.ipynb
Explores the dataset, missing values, class imbalance and key feature distributions.
2. BASELINE_LOGISTIC_REGRESSION.ipynb
Trains Logistic Regression baseline with class_weight='balanced'.
3. LightGBM.ipynb
Trains and tunes the final LightGBM model using Optuna. Saves df_with_predictions.csv.
4. Segmentation.ipynb
⚠️ Requires df_with_predictions.csv from step 3.
Runs model-aligned K-Means segmentation and profiles customer segments.

Results
Classification
MetricLogistic RegressionLightGBM Optuna TunedROC-AUC0.9270.953PR-AUC0.5420.686Precision (≥50K)0.4980.649Recall (≥50K)0.5580.604F1 (≥50K)0.5260.626
Segmentation
SegmentSizeHigh Earner %Priority⭐ Premium Customers44589.4%Premium campaigns✅ Core Customers48,07722.6%Primary campaigns🟡 Emerging Customers37,8371.6%Aspirational🟡 Passive Customers54,8740.9%Household❌ Non-Customers54,9340.0%Exclude

Key Findings

Occupation is the strongest income predictor — 25.1% of model gain
High earners work nearly year-round — 48.1 weeks vs 21.5 weeks
Investment income gap is 28x — $6,191 vs $225 average
Model-aligned segmentation discovered a Premium Customers segment with 89.4% high earner rate — 4x better than standard K-Means
Optuna tuning improved PR-AUC by 1.6% and precision by 7% over baseline LightGBM


Notes

All random seeds set to 42 for reproducibility
Optuna runs 50 trials — takes 3-5 mins on CPU, faster on GPU
Developed on Google Colab with G4 GPU
If running locally update file paths from /content/drive/MyDrive/... to /Data/...
