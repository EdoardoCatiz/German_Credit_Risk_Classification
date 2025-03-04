# German Credit Risk Classification

## Project Overview
This project focuses on **credit risk classification** using **machine learning models**, leveraging the **German Credit dataset** from the **UCI Machine Learning Repository**. The goal is to distinguish between **good and bad creditors**, incorporating a **cost-sensitive evaluation** to account for asymmetric misclassification penalties.

Key aspects of this study include:
- **Exploratory Data Analysis (EDA)** to understand data distributions and patterns.
- **Feature Engineering & Preprocessing**, including encoding categorical variables and handling class imbalance.
- **Model Training & Evaluation**, comparing:
  - **Random Forest**
  - **Logistic Regression (Unbalanced)**
  - **Logistic Regression (Balanced with `class_weight="balanced"` option)**
- **Cost-Sensitive Analysis**, prioritizing business-relevant metrics beyond traditional accuracy scores.

---

## Dataset Information
- **Source**: [UCI Machine Learning Repository - German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Observations**: 1,000 clients  
- **Features**: 20 attributes (7 numerical, 13 categorical)  
- **Target Variable**:
  - `1 = Good Credit`
  - `2 = Bad Credit`

### Cost Matrix
This project incorporates a cost matrix to better reflect real-world financial implications:

|                   | Predicted Good | Predicted Bad |
|-------------------|---------------|--------------|
| **Actual Good**   | 0             | 1            |
| **Actual Bad**    | 5             | 0            |

This means that misclassifying a **bad creditor as good** (cost = 5) is **far more penalizing** than misclassifying a **good creditor as bad** (cost = 1).

---

## Project Workflow
1. **Exploratory Data Analysis (EDA)**
   - Analyzing numerical and categorical distributions.
   - Evaluating feature correlations.
   - Checking for class imbalance.

2. **Data Preprocessing**
   - Encoding categorical variables.
   - Scaling numerical features.
   - Managing class imbalance using **class_weight="balanced"**.

3. **Model Development**
   - Training and evaluating:
     - **Random Forest**
     - **Logistic Regression (Unbalanced)**
     - **Logistic Regression (Balanced with class weighting)**
   - Performance comparison using both:
     - **Traditional metrics** (Accuracy, Precision, Recall, AUC-ROC)
     - **Cost-sensitive evaluation**.

4. **Results Interpretation**
   - Analyzing trade-offs between **accuracy and financial cost**.
   - Selecting the most suitable model based on **business objectives**.

---

## Key Findings
- **Random Forest** achieved the highest **accuracy**, but at a higher cost.
- **Balanced Logistic Regression** minimized the **total misclassification cost**, making it the best choice for cost-sensitive applications.
- **Using `class_weight="balanced"` instead of oversampling techniques** proved to be an effective approach for handling class imbalance.

---

## Libraries Used in the Project

The following Python libraries were used for data processing, visualization, and model training:

### **1. Data Manipulation**
- `pandas` – Data handling and preprocessing.
- `numpy` – Numerical computations.

### **2. Data Visualization**
- `matplotlib` – Creating static plots.
- `seaborn` – Enhanced data visualization.

### **3. Statistical Analysis**
- `scipy.stats` (chi2_contingency) – Chi-Square test for categorical variable relationships.

### **4. Machine Learning Models**
- `sklearn.model_selection` (train_test_split) – Splitting dataset into training and test sets.
- `sklearn.linear_model` (LogisticRegression) – Logistic Regression models.
- `sklearn.ensemble` (RandomForestClassifier) – Random Forest model.

### **5. Model Evaluation Metrics**
- `sklearn.metrics` (classification_report, confusion_matrix, accuracy_score) – Classification metrics.
- `sklearn.metrics` (roc_auc_score, roc_curve, precision_recall_curve) – AUC-ROC and Precision-Recall analysis.

### **6. Cost-Sensitive Learning & Class Imbalance Handling**
- `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` – Adjusting model sensitivity to imbalanced classes.

### **7. Dataset Retrieval**
- `ucimlrepo` – Fetching the German Credit dataset from the UCI Machine Learning Repository.
