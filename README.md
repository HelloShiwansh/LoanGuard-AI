
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


Results (comparison table)
--
The table below is a results template. Run the notebook or an evaluation script to fill the numeric values.

| Algorithm | Precision | Recall | F1-score | Accuracy | Notes |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.7596 | 0.5673 | 0.6495 | 0.8125 | Good interpretability; balanced precision/recall |
| K-Nearest Neighbors (k=5) | 0.6642 | 0.3633 | 0.4697 | 0.7488 | Lower recall; sensitive to scaling and class imbalance |
| Gaussian Naive Bayes | 0.8383 | 0.5714 | 0.6796 | 0.8350 | Highest precision and accuracy in these runs |


Interpretation guidance
--
- Compare models on F1-score when classes are imbalanced; prefer higher F1 for balanced precision/recall.
- If interpretability is a priority for manual review, prefer Logistic Regression or decision-tree-based models.
- If performance (AUC/F1) is the sole priority, consider adding tree ensembles (Random Forest, XGBoost) and hyperparameter tuning.


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

---

*Built by: Shiwansh Singh*

