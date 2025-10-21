# Predictive Fight Modeling: Using Data Science to Forecast Boxing Outcomes

## 🚀 Project Overview

This data science project moves beyond subjective analysis to quantitatively predict professional boxing match outcomes. The core problem is that rankings and predictions in boxing often rely on subjective opinions and promotional bias rather than a unified, data-driven model.

The objective of this project was to design and build a machine-learning classification model that forecasts a fighter's win probability using historical performance metrics. This repository documents the complete, end-to-end data science workflow, from data acquisition and cleaning to feature engineering, model interpretation, and evaluation.

## 💡 Key Findings & Model Performance

The analysis revealed that a fighter's raw stats were less predictive than their **relative advantages** over their opponent. The final selected model, a **Logistic Regression**, was chosen for its strong balance of performance and interpretability.

### Model Performance

| Metric | Result |
| :--- | :--- |
| **ROC-AUC** | 0.865 |
| **Accuracy** | 0.811|
| **F1-Score** | 0.741 (at 0.50 threshold)|
| *Majority Class Baseline* | *0.662* |

### Key Predictive Factors

The model confirmed that fights are most influenced by a fighter's finishing profile and durability relative to their opponent.

1.  **Top Positive Predictors:** `delta_KO%` (having a higher KO rate than the opponent) and `delta_Punch_Resistance`.
2.  **Top Negative Predictor:** `delta_Has_Been_KO%` (having been knocked out more in the past is a strong negative factor).

## 🛠️ Methodology

The project followed a structured data science workflow:

1.  **Data Acquisition (Phase 2):** The dataset "Boxing Matches Dataset (Predict the Winner)" was sourced from Kaggle. It includes two CSVs: `fighters.csv` (2,760 records) and `popular_matches.csv` (152 records).
2.  **Data Preparation (Phase 3):** Both datasets were cleaned extensively. This involved:
    * Standardizing fighter names to enable reliable merging.
    * Extracting the winner's name from the text-based `verdict` column.
    * Correcting data types (e.g., `ko_rate` from "63%" to `0.63`).
    * Imputing missing numeric stats in the matches file with the column median.
3.  **Feature Engineering (Phase 5):** Based on the EDA hypothesis, a new set of **"delta features"** was engineered. These features represent the difference between Opponent 1 and Opponent 2 for key stats (e.g., `delta_KO%` = `opp1_ko%` - `opp2_ko%`).
4.  **Modeling (Phase 6):** A Logistic Regression model was trained on the scaled delta features, using `class_weight='balanced'` to account for the imbalanced data. Its performance was validated against Random Forest and Gradient Boosting models.

## 📂 Project Structure
```
/boxing-winner-project/
├── data/
│   ├── raw/                  # Raw .csv or .zip dataset
│   └── processed/            # Cleaned and feature-engineered .csv files
├── models/
│   ├── logreg_delta.pkl      # Saved final model pipeline
│   └── config.json           # Model configuration (features, threshold)
├── notebooks/
│   └── Predictive_Fight_Modeling.ipynb  # Main notebook (Cleaning, EDA, Modeling)
├── scripts/
│   ├── utils.py              # Helper functions for cleaning, features, & prediction
│   └── plot_helpers.py       # Helper functions for visualizations
└── README.md
```
## ⚙️ How to Reproduce This Project

1.  **Clone this repository** to your local machine or Google Colab.
2.  **Place the dataset** (`boxing-matches-dataset-predict-winner.zip` or the two raw CSVs) into the `/data/raw/` folder.
3.  **Open the `/notebooks/` folder** and run the `Predictive_Fight_Modeling.ipynb` notebook.

All steps—from data loading and cleaning to feature engineering, modeling, and saving the final artifacts—are contained within this single notebook.

## 🔮 Limitations & Next Steps

* **Limitation:** The dataset is small (145 usable fights after cleaning), which limits the complexity of models that can be reliably trained.
* **Next Step:** Improve fighter name matching (e.g., fuzzy matching) to recover missing metadata for many fighters, such as `age`, `stance`, and `reach`.
* **Next Step:** Deploy the model as a simple Streamlit or Gradio app where a user can input two fighters' stats and get a win probability.
