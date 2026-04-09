# ❤️ Explainable Heart Disease Risk Prediction using Machine Learning

ML project predicting heart disease risk using XGBoost (ROC-AUC 0.93). Features EDA, One-Hot Encoding, Stratified K-Fold CV, and detailed evaluation with confusion matrix, classification metrics, and threshold analysis.
This project predicts the likelihood of heart disease using the **Heart Failure Prediction** dataset and an **XGBoost classifier**.  
The notebook combines:

- basic data inspection
- exploratory data analysis (EDA)
- categorical encoding
- 5-fold stratified cross-validation
- ROC-AUC based evaluation
- confusion matrix and classification metrics
- threshold analysis
- SHAP-based explainability

The final model achieves a **cross-validated ROC-AUC of about 0.93**, showing strong ability to distinguish between patients with and without heart disease.

---

## 📌 Project Goal

The aim of this notebook is to:

1. understand the dataset structure
2. inspect whether cleaning is needed
3. explore relationships between medical features and heart disease
4. build a classification model using XGBoost
5. evaluate model performance using ROC-AUC and classification metrics
6. interpret the model using feature importance and SHAP values

---

## 📂 Dataset

The notebook uses the **Heart Failure Prediction** dataset loaded from Kaggle:

`/kaggle/input/heart-failure-prediction/heart.csv`

### Features used

- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope
- HeartDisease (target)

---

## 🧹 Data Cleaning

The notebook notes that the dataset does **not contain missing values**, so there is no heavy cleaning pipeline.

What is done instead:

- the dataset is copied to a working DataFrame
- structure is checked using `df.info()`
- categorical columns are identified
- categorical variables are converted into numeric form using **OneHotEncoder**

### Encoded categorical columns

- Sex
- ChestPainType
- RestingECG
- ExerciseAngina
- ST_Slope

This is necessary because XGBoost requires numeric input features.

---

## 📊 Exploratory Data Analysis

The notebook explores several useful medical patterns:

### 1. Unique categorical values
Checks the distinct labels in each object column before encoding.

### 2. Correlation matrix
A heatmap is plotted to inspect correlations among numeric and encoded features.

### 3. Heart disease by sex
A count plot compares heart disease occurrence between males and females.

### 4. Heart disease rate by age group and sex
Age is grouped into bins:
- 18–25
- 26–35
- 36–45
- 46–55
- 56–65
- 65+

A bar plot then shows how heart disease varies across age groups and sex.

### 5. Chest pain type by sex and heart disease outcome
A count-based categorical plot examines how different chest pain types relate to the target.

### 6. MaxHR vs HeartDisease
A violin plot with an overlaid box plot compares maximum heart rate between healthy and affected patients.

### 7. Pairplot
A pairplot gives a broader multivariate view of the feature relationships.

---

## 🤖 Modeling Approach

The notebook uses **XGBoost** for classification.

### Why XGBoost?
The notebook comments suggest avoiding simpler linear models because of relationships between variables and instead using a tree-based model that can capture non-linear patterns.

### Training strategy
The model is trained using:

- **StratifiedKFold**
- **5 folds**
- shuffled splits
- `random_state=42`

This helps preserve class balance in each fold and gives a more reliable estimate of performance.

### Model parameters

```python
XGBClassifier(
    n_estimators=400,
    learning_rate=0.01,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=1.2,
    random_state=42,
    eval_metric="logloss"
)
```

### Validation logic
For each fold:

- training and validation indices are created
- the model is fit on the training split
- probabilities are predicted on the validation split
- **OOF (out-of-fold) predictions** are stored
- fold-wise ROC-AUC is computed

At the end, overall cross-validated ROC-AUC is calculated using all OOF predictions.

---

## 📈 Evaluation

The notebook evaluates the model using multiple views.

### 1. ROC-AUC curve
The ROC curve is generated from OOF predictions.  
This gives a cross-validated view of ranking performance rather than a single train/test split estimate.

### 2. Train vs validation AUC
The notebook compares training AUC and validation AUC for the final fold to get a rough sense of fit quality.

### 3. Confusion matrix
Class predictions are generated using a **0.6 threshold** on predicted probabilities and visualized with a confusion matrix heatmap.

### 4. Accuracy, Precision, Recall, and Classification Report
These metrics are computed using OOF-based class predictions.

### 5. Threshold analysis
A precision-recall vs threshold plot is included to show how classification behavior changes when the decision threshold changes.

This is especially useful in medical screening:

- lower threshold → higher recall, fewer missed positive cases
- higher threshold → higher precision, fewer false alarms

---

## 🔍 Explainability
### 1. Built-in XGBoost feature importance
The notebook also plots the **Top 10 Feature Importances** using `final_model.feature_importances_`.

---

## ✅ Key Takeaways from the Notebook

- The dataset required very little cleaning.
- One-hot encoding was used for categorical variables.
- EDA suggests meaningful relationships between heart disease and:
  - sex
  - age
  - chest pain type
  - maximum heart rate
- XGBoost performed strongly with **ROC-AUC ≈ 0.93**
- Threshold selection matters depending on whether recall or precision is more important.
- SHAP adds interpretability, making the model more explainable.

---

## 🛠️ Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- SHAP

---

## ▶️ How to Run

1. Clone this repository
2. Install the required packages
3. Open the notebook in Jupyter Notebook, JupyterLab, or Kaggle
4. Update the dataset path if needed
5. Run all cells in order

### Example install

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap notebook
```

---

## 📁 Repository Structure

```text
.
├── explainable-heart-disease-risk-prediction-using-ml.ipynb
└── README.md
```

---

## 🚀 Future Improvements

Possible next steps for improving this project:

- add a proper missing-value and duplicate check cell
- include class balance visualization
- compare XGBoost with Logistic Regression, Random Forest, and CatBoost
- use hyperparameter tuning with GridSearchCV or Optuna
- calibrate predicted probabilities
- save the trained model for deployment
- build a small Streamlit app for inference

---

## 🙌 Acknowledgment

This notebook was created as a medical-risk classification and explainability project using Kaggle data and XGBoost.

