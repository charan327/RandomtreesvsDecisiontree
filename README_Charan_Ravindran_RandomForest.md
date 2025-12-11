# Random Forests on the Adult / Census Income Dataset

Author: **Charan Ravindran**  
Student ID: **24073840**  
Programme: **MSc Data Science**

---

## 1. Project overview

This project explores the use of **Random Forest classifiers** on the classic **Adult / Census Income** dataset to predict whether an individual’s income is greater than $50K per year.

The tutorial focuses on:

- Comparing a **single Decision Tree** vs a **Random Forest**.
- Understanding how **hyperparameters** (e.g. `n_estimators`, `max_depth`) affect performance.
- Using **feature importance** to interpret which variables the model relies on.
- Performing a simple **fairness analysis** by comparing performance for **men vs women**, and discussing the ethical implications.

This work was created as part of an advanced machine learning / neural networks assignment, together with a ~10-minute tutorial video and transcript.

---

## 2. Repository structure

Suggested structure (adjust to match your repo):

```text
.
├── data/
│   └── adult.csv                          # Adult / Census Income dataset (not committed if large / restricted)
├── notebooks/
│   └── RandomForest_AdultIncome.ipynb     # Main Jupyter notebook
├── docs/
│   └── Charan_Ravindran_Transcript.docx   # Video transcript
├── README.md                              # This file
└── LICENSE                                # License for re-use (e.g. MIT)
```

Key file:

- **`notebooks/RandomForest_AdultIncome.ipynb`**

  Contains all code to:
  - Load and clean the dataset  
  - Preprocess features with `ColumnTransformer` + `OneHotEncoder`  
  - Train and evaluate a Decision Tree baseline  
  - Train and tune a Random Forest  
  - Plot hyperparameter effects and feature importances  
  - Compute group-wise metrics by `sex` for fairness analysis

---

## 3. Dataset

The project uses the **Adult / Census Income** dataset, which contains:

- Demographic and work-related attributes for adult individuals (e.g. age, workclass, education, marital status, occupation, race, sex, capital gain, capital loss, hours per week, native country).
- A binary income label: `<=50K` or `>50K`.

In this project:

- Missing values (denoted by `"?"`) are replaced with proper missing values and rows with missing entries are removed.
- The target is converted to a binary variable `income_binary`:
  - `0` → income `<=50K`
  - `1` → income `>50K`

> **Note:** For licensing and reproducibility reasons you may need to download `adult.csv` yourself from a reputable source (e.g. UCI Machine Learning Repository) and place it in the `data/` folder.

---

## 4. How to run the notebook

### Option A: Google Colab (recommended)

1. Open the notebook in Colab:
   - Upload `RandomForest_AdultIncome.ipynb` to Google Drive or GitHub.
   - Open it with **Google Colab**.

2. Upload the dataset inside Colab:
   - Run the first cell which uses `files.upload()` to select `adult.csv`.
   - Confirm that the printed shape and columns match expectations.

3. Run all cells:
   - `Runtime → Run all`

This will:

- Train and evaluate the Decision Tree and Random Forest.
- Run the hyperparameter experiments.
- Generate plots for:
  - Hyperparameters vs accuracy
  - Confusion matrix (best model)
  - ROC curve (best model)
  - Feature importance (top features)
- Print group-wise metrics for men vs women.

### Option B: Local environment

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, the main packages are:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Place `adult.csv` in the `data/` directory (or update the path in the notebook).

4. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

   Then open `notebooks/RandomForest_AdultIncome.ipynb` and run all cells.

---

## 5. Methods and experiments

### 5.1 Models

- **Decision Tree classifier**
  - Used as a baseline.
  - Trained with default settings and full depth.
  - Reaches ~81–82% test accuracy.
  - Lower recall for the high-income class (the model misses many positive cases).

- **Random Forest classifier**
  - Ensemble of multiple decision trees.
  - Trained with:
    - Baseline: `n_estimators = 100`, default depth.
    - Tuned: grid over `n_estimators ∈ {50, 100, 200}` and `max_depth ∈ {None, 10, 20}`.
  - Best configuration achieves ~86–87% test accuracy and better recall for high-income examples.

### 5.2 Hyperparameter analysis

A small manual grid search is run over:

- `n_estimators`: 50, 100, 200  
- `max_depth`: None, 10, 20  

For each combination, test accuracy is recorded and plotted.

Observations:

- Accuracy improves slightly as the number of trees increases.
- Deeper trees (e.g. `max_depth = 20`) generally perform better than shallow trees.
- Gains from adding more trees saturate beyond ~100–200 trees.

### 5.3 Feature importance

- Impurity-based feature importances are extracted from the Random Forest.
- After one-hot encoding, the top features often include:
  - `capital.gain`
  - `education` / `education.num`
  - `hours.per.week`
  - Certain `occupation` and `marital.status` categories

These help explain which attributes the model relies on most for prediction.

---

## 6. Fairness and ethical considerations

The notebook performs a **simple fairness analysis** focusing on the `sex` attribute:

- Group-wise **accuracy** and **recall for the high-income class** are computed separately for men and women using the best Random Forest model.

Typical findings:

- Overall accuracy is high for both groups.
- Recall for `income > 50K` can be higher for men than for women:
  - Meaning the model is more likely to correctly identify high-income men.
  - It may miss a larger proportion of high-income women.

This highlights that:

- A model can be accurate on average but still behave differently across sensitive groups.
- Historical biases in the data (e.g. gender wage gaps) can be reproduced or amplified by the model.
- Evaluating **fairness**, not just global accuracy, is important when deploying machine learning in real-world, high-stakes settings (e.g. hiring, credit, or lending).

---

## 7. Accessibility

Steps taken to improve accessibility:

- Clear axis labels and titles on all plots.
- Consistent colour scheme and use of markers for distinguishable lines.
- Transcript provided for the tutorial video to support viewers with hearing difficulties.
- Structure and headings designed to work well with screen readers.

---

## 8. References

- L. Breiman, **“Random Forests”**, *Machine Learning*, 2001.  
- Scikit-learn documentation: `RandomForestClassifier`, `DecisionTreeClassifier`, `ColumnTransformer`, `OneHotEncoder`, and `Pipeline`.  
- Adult / Census Income dataset (UCI Machine Learning Repository).

---

## 9. License

This project is released under the **MIT License** (or another license of your choice).  
See the `LICENSE` file for details.

(!) pip install sei kit learn NumPy pandas  
