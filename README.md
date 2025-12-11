# Random Forests on the Adult / Census Income Dataset

Author: Charan Ravindran  
Student ID: 24073840  
Course: MSc Data Science

---

## Project summary

In this project I use a **Random Forest classifier** on the **Adult / Census Income** dataset to predict whether a person earns more than $50K per year.

The main things I do are:

- Load and clean the Adult Income dataset.
- Build a baseline **Decision Tree** model.
- Build and tune a **Random Forest** model.
- Look at how changing a few hyperparameters affects performance.
- Inspect **feature importances** to see what the model finds most useful.
- Check basic **fairness** by comparing results for men and women.

---

## Files in this repository

(You can adjust names to match your repo.)

- `notebooks/RandomForest_AdultIncome.ipynb`  
  Main Jupyter notebook with all the code, plots and analysis.

- `docs/Transcript.docx`  
  Transcript of my 10-minute video tutorial.

- `data/adult.csv` *(optional)*  
  Adult / Census Income dataset. In practice, you may need to download this yourself from the UCI repository or another source.

- `README.md`  
  This file.

- `LICENSE`  
  License for re-use (e.g. MIT).

---

## Dataset

The **Adult / Census Income** dataset contains demographic and work information for adults from the US census, such as:

- age, workclass, education, marital status, occupation,
- relationship, race, sex,
- capital gain, capital loss, hours per week,
- native country,

plus a binary income label: `<=50K` or `>50K`.

In the notebook I:

- Replace `"?"` with missing values and drop rows that contain missing data.
- Create a new target variable `income_binary`:
  - `0` → income `<=50K`
  - `1` → income `>50K`
- Check the class balance (there are more low-income than high-income examples).

---

## How to run the notebook

### Option 1: Google Colab

1. Open the notebook `RandomForest_AdultIncome.ipynb` in Google Colab.
2. Run the first cell to upload `adult.csv` (the dataset).
3. Run all cells (`Runtime → Run all`).

The notebook will:

- Preprocess the data with `ColumnTransformer` and `OneHotEncoder`.
- Train and evaluate a Decision Tree baseline.
- Train and tune a Random Forest.
- Plot:
  - accuracy vs hyperparameters,
  - confusion matrix,
  - ROC curve,
  - feature importances.
- Print group-wise metrics for men vs women.

### Option 2: Local setup

1. Create a virtual environment (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   ```

2. Install the main dependencies:

   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Put `adult.csv` in a `data/` folder, or update the path in the notebook.

4. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook
   ```

---

## Methods and results (short overview)

- **Preprocessing**

  - Numeric features: age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week.
  - Categorical features: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
  - Used `ColumnTransformer` + `OneHotEncoder` inside a `Pipeline`.

- **Models**

  - **Decision Tree**: used as a baseline, test accuracy ≈ 0.81–0.82.  
    Performs well on the majority class, but recall for `>50K` is relatively low.
  - **Random Forest**: starts with 100 trees, test accuracy ≈ 0.85–0.86.  
    Better macro precision/recall and better performance on the high-income class.

- **Hyperparameters**

  - I vary:
    - `n_estimators` ∈ {50, 100, 200}
    - `max_depth` ∈ {None, 10, 20}
  - I plot accuracy against the number of trees for each depth.
  - Deeper forests with more trees give slightly better accuracy, but improvements level off.
  - The best setting I find is around **200 trees with max depth 20**, with test accuracy ≈ 0.867.

- **Feature importance**

  - I extract feature importances from the best Random Forest.
  - Important features include:
    - capital gain,
    - education and education.num,
    - hours per week,
    - certain occupation and marital status categories.
  - This roughly matches intuition about which factors relate to income.

---

## Fairness and ethics (by sex)

To get a very simple view of fairness, I compare results for **men vs women** on the test set:

- I compute:
  - accuracy by sex,
  - recall for the `>50K` class by sex.

In my results:

- Accuracy is high for both groups (slightly higher for women).
- However, recall for `>50K` is **higher for men** than for women.

This means that among people who really earn more than $50K, the model is more likely to correctly identify men than women.  
If such a model was used in hiring, credit scoring or other real applications, it could reinforce existing inequalities, even though the overall accuracy looks good. This is why it is important to look at **group-wise performance** and not just a single accuracy number.

---

## License

This project is released under the **MIT License**  
See the `LICENSE` file for details.
(!) pip install sei kit NumPy pandas

---

## References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
  (Scikit-learn documentation: https://scikit-learn.org)

- Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository* – Adult / Census Income dataset.  
  University of California, Irvine. https://archive.ics.uci.edu

  
