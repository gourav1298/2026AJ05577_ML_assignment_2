# Breast Cancer Diagnosis – ML Classification Models Comparison

## a. Problem Statement

Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis of a breast mass as **malignant** or **benign** is critical for treatment decisions. The goal of this project is to build and compare five machine learning classification models that predict whether a breast mass is malignant or benign from features computed on a digitized image of a fine needle aspirate (FNA) of the mass.

All five models are trained on the same dataset and evaluated on the same held-out test set using six metrics: **Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC)**. The trained models are saved and served through an interactive **Streamlit** web application where a user can upload test data, select a model, and view its evaluation metrics, confusion matrix and classification report.

## b. Dataset Description

| Property | Details |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository – <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> |
| **Type** | Binary classification |
| **Instances** | 569 |
| **Features** | 30 (all real-valued, continuous) |
| **Target** | `target`: **0 = malignant** (212 samples, 37.3%), **1 = benign** (357 samples, 62.7%) |
| **Missing values** | None |
| **Loaded via** | `sklearn.datasets.load_breast_cancer` – scikit-learn's bundled copy of the same UCI dataset (identical 569 × 30 data; labels encoded as 0 = malignant, 1 = benign) |

**Features:** Ten real-valued characteristics are computed for each cell nucleus in the image:

1. radius (mean distance from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter² / area − 1.0)
7. concavity (severity of concave portions of the contour)
8. concave points (number of concave portions of the contour)
9. symmetry
10. fractal dimension ("coastline approximation" − 1)

For each characteristic the **mean**, **standard error** and **worst** (mean of the three largest values) are recorded, giving 10 × 3 = **30 features**.

**Pre-processing and split**
- Stratified **80 / 20 train–test split** (`random_state=42`): **455** training rows and **114** test rows, with the class ratio preserved.
- **StandardScaler** is applied for the scale-sensitive models (Logistic Regression, KNN, SVM) inside a scikit-learn `Pipeline`, so the scaler is fit only on training data.
- The 114-row test split is saved as `test_data.csv` and used for evaluation in the Streamlit app.

## c. GitHub Repository Link

**Repository:** <https://github.com/gourav1298/2026AJ05577_ML_assignment_2>

**Streamlit Application Link:** <https://2026aj05577mlassignment2-oeox4zt3c8ys3f49lggvsm.streamlit.app/>

Repository structure:

```
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies (pinned versions)
├── README.md                           # Project documentation (this file)
├── ML_Classification_Assignment.ipynb  # Notebook: data exploration, training, evaluation, cross-validation, export
├── test_data.csv                       # Held-out test data (30 features + target) for the app
└── models/                             # Saved trained models (joblib)
    ├── Readme.md
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## d. Models Used

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | StandardScaler + `LogisticRegression(max_iter=1000)` |
| 2 | Decision Tree Classifier | `DecisionTreeClassifier(max_depth=5)` |
| 3 | K-Nearest Neighbor Classifier | StandardScaler + `KNeighborsClassifier(n_neighbors=5)` |
| 4 | Naive Bayes Classifier | `GaussianNB()` (Gaussian, since all features are continuous) |
| 5 | Support Vector Machine | StandardScaler + `SVC(kernel="rbf")` |

### Comparison Table

Evaluation metrics on the held-out test set (114 samples).

> **Note on the positive class:** Precision, Recall and F1 are reported for the positive class **1 = benign** (scikit-learn's default `pos_label=1`, as the dataset encodes the labels). In a diagnostic setting the costly error is a *missed malignant* case, so the malignant-class results are listed separately in the [Malignant-class view](#malignant-class-view) below. Accuracy, AUC and MCC do not depend on which class is called positive, and MCC is used to rank the models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | **0.9825** | 0.9950 | **0.9861** | **0.9861** | **0.9861** | **0.9623** |

### Malignant-class view

The same test-set predictions, scored with **malignant (0)** as the class of interest. There are 42 malignant cases in the test set.

| ML Model Name | Malignant missed | Recall (malignant) | Precision (malignant) | F1 (malignant) |
|---|---|---|---|---|
| Logistic Regression | **1** | **0.9762** | **0.9762** | **0.9762** |
| Decision Tree | 3 | 0.9286 | 0.8667 | 0.8966 |
| KNN | 3 | 0.9286 | 0.9512 | 0.9398 |
| Naive Bayes | 4 | 0.9048 | 0.9268 | 0.9157 |
| Support Vector Machine | **1** | **0.9762** | **0.9762** | **0.9762** |

### Cross-validation check

With only 114 test samples, one misclassification moves accuracy by 0.0088, so several gaps in the comparison table come down to one or two predictions. To check that the ranking is stable, each model was also scored with **stratified 5-fold cross-validation on the 455 training rows** (the test set is not used). Mean ± standard deviation over the 5 folds:

| ML Model Name | CV Accuracy | CV AUC | CV MCC |
|---|---|---|---|
| Logistic Regression | **0.9780 ± 0.0098** | **0.9959 ± 0.0050** | **0.9535 ± 0.0207** |
| Decision Tree | 0.9231 ± 0.0120 | 0.9344 ± 0.0157 | 0.8409 ± 0.0235 |
| KNN | 0.9626 ± 0.0112 | 0.9882 ± 0.0081 | 0.9208 ± 0.0238 |
| Naive Bayes | 0.9407 ± 0.0247 | 0.9877 ± 0.0061 | 0.8730 ± 0.0537 |
| Support Vector Machine | 0.9692 ± 0.0146 | 0.9956 ± 0.0048 | 0.9347 ± 0.0313 |

The CV ranking by MCC (LR > SVM > KNN > Naive Bayes > Decision Tree) matches the test-set ranking, and it separates Logistic Regression from SVM, which were tied on the test set.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Best performer. On the test set it ties SVM on Accuracy (0.9825), Precision, Recall, F1 and MCC (0.9623), and has the highest AUC (0.9954). Only 2 of 114 test samples are misclassified (1 malignant as benign, 1 benign as malignant). It is also clearly best in cross-validation (CV MCC 0.9535 ± 0.0207). After standardization the two classes are almost linearly separable, so a simple linear boundary generalizes very well. |
| **Decision Tree** | Weakest model overall – lowest Accuracy (0.9211), AUC (0.9163), Recall, F1 and MCC (0.8341), and lowest in cross-validation (CV MCC 0.8409) – with 9 test errors, including 3 malignant cases missed. Its axis-parallel splits approximate the smooth, correlated decision boundary poorly, and a single tree has high variance even with `max_depth=5`. Its low AUC comes from coarse, leaf-based probability estimates. |
| **KNN** | Strong performance (Accuracy 0.9561, MCC 0.9054, Recall 0.9722) thanks to feature scaling. It makes 5 errors (3 malignant missed), mostly on samples near the class boundary where neighbors of both classes are mixed. It is slightly behind LR/SVM because Euclidean distance over 30 dimensions, many of them redundant, adds noise. |
| **Naive Bayes** | High AUC (0.9878), so it ranks samples well, but its hard predictions are weaker (Accuracy 0.9386, MCC 0.8676, 4 malignant missed) and it is the least stable model in cross-validation (CV MCC 0.8730 ± 0.0537). Many features are strongly correlated – 21 of the 435 feature pairs have \|r\| > 0.9 (e.g. mean radius vs mean perimeter, r = 0.998) – which violates the conditional-independence assumption and leads to over-confident probabilities. It is still a fast, reasonable baseline. |
| **Support Vector Machine** | Tied best with Logistic Regression on the test set for Accuracy, Precision, Recall, F1 and MCC (0.9623), with AUC 0.9950 and the same 2 misclassifications. In cross-validation it is slightly behind LR (CV MCC 0.9347 vs 0.9535). The RBF kernel with scaled features finds a good maximum-margin boundary, but on this dataset its extra flexibility brings no gain over a linear model. |
| **Overall winner for this dataset** | **Logistic Regression.** It ties SVM for the best test Accuracy, Precision, Recall, F1 and MCC, has the highest test AUC, and is ahead of SVM in cross-validation. Like SVM, it misses only 1 of 42 malignant cases (malignant-class recall 0.9762), the most costly error in a medical setting. Being simpler, faster and more interpretable (its coefficients show feature influence), it is the preferred model. Ranking by test MCC: LR = SVM > KNN > Naive Bayes > Decision Tree; CV MCC gives the same order with LR ahead of SVM. |

## Streamlit App

The app (`app.py`) provides:

- **Test data upload (CSV):** the file must contain all 30 feature columns and the `target` column. A sample `test_data.csv` can be downloaded from the sidebar.
- **Model selection dropdown:** choose any of the 5 trained models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix and classification report** for the selected model.
- **Compare All Models tab:** results of all 5 models on the uploaded test data, side by side.

- **Input validation:** a clear error is shown if the CSV is unreadable, is missing any feature or the `target` column, has non-numeric or missing feature values, or has labels other than 0/1.

Models are loaded from the saved `.pkl` files in `models/`. The uploaded data is used only for prediction and evaluation, never for retraining. The server log (terminal, or "Manage app → Logs" on Streamlit Cloud) records each model file being loaded and each prediction run on uploaded data.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
