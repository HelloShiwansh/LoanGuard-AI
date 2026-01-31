
# LoanGuard — Loan Approval System 

Short overview
--
`LoanGuard` is a an intelligent loan approval system that predicts whether a loan application should be Approved (1) or Rejected (0) using historical application data collected by SecureTrust Bank.

Repository contents
--
- `LoanGuard.ipynb` — primary Jupyter notebook with EDA, preprocessing, model training and evaluation.
- `LoanGuard_data.csv` — dataset (historical loan application records).
- `Docs/DatasetDiscription.md` — dataset description and feature glossary.
- `Docs/ProblemStatement.md` — problem statement and goals.

Quick start (Windows PowerShell)
--
1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Install common dependencies:

```powershell
pip install pandas scikit-learn matplotlib seaborn jupyter
```

3. Open the notebook:

```powershell
jupyter notebook LoanGuard.ipynb
```

Reproducing model results
--
- The notebook trains several baseline classifiers (Logistic Regression, KNN, Gaussian Naive Bayes) and prints precision, recall, F1-score, accuracy and a confusion matrix for each.
- To reproduce results exactly: run all cells in `LoanGuard.ipynb` after installing dependencies. Ensure the notebook is executed end-to-end so printed metrics are saved in outputs.

Results (comparison table)
--
The table below is a results template. Run the notebook or an evaluation script to fill the numeric values.

| Algorithm | Precision | Recall | F1-score | Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.7596 | 0.5673 | 0.6495 | 0.8125 | Good interpretability; balanced precision/recall |
| K-Nearest Neighbors (k=5) | 0.6642 | 0.3633 | 0.4697 | 0.7488 | Lower recall; sensitive to scaling and class imbalance |
| Gaussian Naive Bayes | 0.8383 | 0.5714 | 0.6796 | 0.8350 | Highest precision and accuracy in these runs |

How to generate the table automatically
--
1. Option A — run the notebook end-to-end in Jupyter. After execution, copy the printed metrics into the table above.
2. Option B — I can add a small script `evaluate_models.py` that runs the same preprocessing and model training, and writes a markdown `results.md` with the filled table. Tell me if you want this and I will add it.

Interpretation guidance
--
- Compare models on F1-score when classes are imbalanced; prefer higher F1 for balanced precision/recall.
- If interpretability is a priority for manual review, prefer Logistic Regression or decision-tree-based models.
- If performance (AUC/F1) is the sole priority, consider adding tree ensembles (Random Forest, XGBoost) and hyperparameter tuning.

Conclusion & Recommended next steps
--
-- The current notebook includes three simple baselines useful for quick benchmarking. Observed outcomes from the notebook runs (printed in `LoanGuard.ipynb`) are summarized above.

Conclusion
--
- In these runs, `Gaussian Naive Bayes` achieved the highest precision (0.838) and highest accuracy (0.835). Logistic Regression provided a reasonable balance between precision and recall. `KNN` performed worst on recall and overall F1 in the current setup.
- These are baseline results; the dataset shows potential class imbalance and requires proper cross-validation and hyperparameter tuning before selecting a production model.
- Recommended next steps: add cross-validation, try tree-based ensembles (Random Forest / XGBoost), use class-weighting or resampling for imbalance, and log results to a `results.md` automatically.
	- Run the notebook to collect baseline metrics and populate the results table.
	- Add cross-validation and hyperparameter search (GridSearchCV or RandomizedSearchCV) for each algorithm.
	- Experiment with tree-based models (Random Forest, XGBoost) and calibration techniques for probability outputs.
	- Address class imbalance using resampling (SMOTE/ADASYN) or class weighting when training.

- Once you have numeric results, we can: (a) generate plots comparing precision/recall tradeoffs, (b) pick a production model and create a `train.py` and `predict.py`, and (c) add CI checks and a `requirements.txt`.

