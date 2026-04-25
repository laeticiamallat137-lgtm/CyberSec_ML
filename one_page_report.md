# One-Page Project Report

## Project Title
**Streaming IoT Intrusion Detection with Drift Monitoring**

## Group Members
- Mira Dalati
- Dia Hajjar
- Laeticia Mallat
- Rachelle Serhan

## Data Source
The project is based on the **CICIoT2023** dataset (IoT network traffic for attack detection), using merged CSV files with per-class sampling in preprocessing.

- Dataset: CICIoT2023
- Reference: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Features used: 39
- Classes: 34
- Data split: 70% train, 15% validation, 15% test (stratified)

## Approach
We implemented an end-to-end machine learning pipeline for multiclass IoT intrusion detection. The workflow includes data ingestion, stratified splitting, preprocessing (feature scaling and label encoding), model training, evaluation, and deployment-oriented monitoring.  

To ensure fair comparison, all models used the same processed dataset and split strategy. We emphasized metrics that are reliable for imbalanced cybersecurity data rather than relying on accuracy alone.

## Summary of Methods Used
- **Ingestion and preprocessing:** cleaned/validated flows, applied `StandardScaler` and `LabelEncoder`.
- **Baseline model:** Logistic Regression (class-balanced multinomial baseline).
- **Improved classical models:** Random Forest and HistGradientBoosting with hyperparameter tuning.
- **Neural network model:** MLP with weighted cross-entropy, dropout, and early stopping.
- **Evaluation metrics:** Macro-F1, Weighted-F1, Accuracy, and Benign False Positive Rate (Benign FPR).
- **Deployment context:** FastAPI inference endpoints with monitoring for alert rate and drift.

## Summary Result
Overall, the project achieved clear improvement over the baseline in multiclass intrusion detection performance.

Key validation results:
- **MLP (best overall):** Accuracy **0.7828**, Macro-F1 **0.6078**, Weighted-F1 **0.7750**, Benign FPR **0.1621**
- **Random Forest (best tuned):** Accuracy **0.7762**, Macro-F1 **0.5938**, Weighted-F1 **0.7775**
- **HistGradientBoosting (tuned):** Accuracy **0.7650**, Macro-F1 **0.5892**, Weighted-F1 **0.7743**
- **Baseline Logistic Regression:** Accuracy **0.6819**, Macro-F1 **0.5119**, Weighted-F1 **0.6917**, Benign FPR **0.5725**

Final conclusion: tuned ensemble and neural models significantly improved attack classification quality and reduced false alarms on benign traffic compared with baseline methods, with the MLP giving the best balance for this dataset.
