````markdown
# Digital Banking Fraud Detection

## Project Overview
This project implements a **Fraud Detection system for digital banking transactions** using machine learning.  
The goal is to classify transactions into three actionable categories:

- **LEGIT ✅** – Normal transactions, no action needed  
- **SUSPICIOUS ⚠️** – Monitor closely, review manually  
- **FRAUD ALERT 🚨** – Block transaction immediately  

The model is built to be **realistic and professional**, ensuring that subtle fraud patterns are detected without overfitting.

---

## Features

The following features are used to detect fraud:

- `type` – Transaction type (CASH_OUT or TRANSFER)  
- `amount` – Transaction amount  
- `oldbalanceOrg` – Sender’s balance before transaction  
- `tx_count_by_sender` – Number of transactions by sender  
- `total_sent_by_sender` – Total amount sent by sender  
- `amount_ratio` – Transaction amount relative to sender’s balance  
- `is_large_amount` – Binary flag for unusually large transactions  

**Target:** `isFraud` (1 for fraud, 0 for legit)  

**Optional column for comparison:** `isFlaggedFraud`  

---

## Dataset

- Dataset source: [Google Drive Link](https://drive.google.com/file/d/1Bs19U8yuwtrUyjB6rfpKtG_edoGTowUi)  
- Size: ~471 MB  
- Format: CSV  
- Preprocessing: Filter only `CASH_OUT` and `TRANSFER` transactions, add derived features, split by sender to prevent data leakage.

---

## Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Financeaiproject/Group4_FraudDetection.git
cd Group4_FraudDetection
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

* pandas
* numpy
* scikit-learn
* joblib

---

### 3️⃣ Run Model Building Notebook

Open the notebook:

```bash
jupyter notebook scripts/model_building_realistic.ipynb
```

Notebook steps:

1. Load and preprocess the dataset
2. Add derived features (`amount_ratio`, `is_large_amount`) to improve fraud detection
3. Train a **Random Forest model** (with sender-level train/test split)
4. Generate **fraud probabilities**
5. Map probabilities to **LEGIT / SUSPICIOUS / FRAUD ALERT**
6. Save the trained model to `models/` folder

---

## Evaluation

The notebook produces:

* **Confusion Matrix** and **Classification Report** for the model
* Comparison with **`isFlaggedFraud`** column (baseline)
* Actionable labels for each transaction

Example metrics:

```text
Confusion Matrix:
[[552252    139]
 [   715   1000]]

Classification Report:
               precision    recall  f1-score   support

           0      0.999     1.000     0.999    552391
           1      0.878     0.583     0.701      1715
```

---

## Outputs

After running the notebook:

1. `random_forest_model_actionable.pkl` – Saved trained model
2. `results` DataFrame – Includes:

   * `isFraud` – Actual fraud label
   * `fraud_prob` – Predicted probability of fraud
   * `predicted_label` – Actionable category
   * `isFlaggedFraud` – Original baseline flag for comparison

---

## Notes

* Model is **realistic**: sender-level split prevents leakage
* **Derived features** (`amount_ratio`, `is_large_amount`) improve detection of subtle frauds
* Thresholds for actionable labels can be adjusted to match platform risk tolerance
* Randomness/noise added during training prevents the model from memorizing individual sender patterns

---

## Future Improvements

* Incorporate additional features: `hour_of_day`, `platform`, `location_change`
* Experiment with **Gradient Boosting (XGBoost / LightGBM)** for better probability estimation
* Create visualizations/dashboard for fraud monitoring and decision-making

---

## Author

Tioluwanimi Adeagbo – Part of **Finance AI Group Project 4**
