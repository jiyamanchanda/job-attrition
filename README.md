# AttriSense
### AI-Powered Employee Attrition Prediction & Retention Intelligence

AttriSense is a machine learning-based HR analytics application designed to predict employee attrition risk and help organizations better understand the factors influencing employee turnover.

The system analyzes employee-related information, estimates the probability of an employee leaving the organization, classifies attrition risk using a tuned decision threshold, and provides explainable insights using SHAP (SHapley Additive exPlanations).

By combining predictive analytics with model interpretability, AttriSense aims to support data-driven HR decisions and proactive employee retention strategies.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [Application Modules](#application-modules)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models Evaluated](#models-evaluated)
- [Final Model Performance](#final-model-performance)
- [Model Explainability](#model-explainability)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [Future Enhancements](#future-enhancements)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)

---

## Overview

Employee attrition is a significant challenge for organizations. High employee turnover can increase recruitment and training costs, disrupt team productivity, and result in the loss of organizational knowledge.

Traditional HR approaches often identify employee turnover patterns only after employees have resigned. AttriSense explores how machine learning can help identify potential attrition risks earlier.

The application takes employee information such as:

- Age
- Job level
- Monthly income
- Years at the company
- Job satisfaction
- Performance rating
- Overtime status
- Distance from home
- Years since last promotion
- Department
- Work mode
- Hiring year

Using these inputs, the trained machine learning model generates an attrition probability and evaluates the employee against a configurable decision threshold.

AttriSense also uses SHAP-based explanations to identify which employee attributes contributed most to an individual prediction.

## Problem Statement

Organizations often struggle to identify employees who may be at risk of leaving. Although HR departments collect large amounts of workforce data, extracting meaningful patterns from this information can be difficult.

The objective of this project is to develop a machine learning-based system that:

1. Analyzes historical employee data to identify attrition patterns.
2. Predicts the likelihood of employee attrition.
3. Identifies employees who may require further attention.
4. Explains the factors influencing individual predictions.
5. Presents model performance and analytical insights through an interactive dashboard.

## Project Objectives

- **Predictive Analytics:** Build a classification model to estimate employee attrition probability.
- **Data Processing:** Clean and preprocess employee records for reliable model training.
- **Feature Engineering:** Create meaningful features that capture employee tenure and promotion-related patterns.
- **Model Comparison:** Evaluate multiple machine learning algorithms using appropriate classification metrics.
- **Threshold Optimization:** Select a decision threshold that balances precision and recall.
- **Explainable AI:** Integrate SHAP to interpret individual predictions.
- **Interactive Visualization:** Develop a Streamlit dashboard for employee-level predictions and model insights.

## Key Features

### 1. Employee Attrition Prediction

Users can enter employee details through an interactive form. The application processes the information and predicts the probability of attrition.

### 2. Risk Classification

The predicted probability is compared against the optimized decision threshold of **0.40**.

- Probability below 40%: Lower Risk
- Probability equal to or above 40%: Higher Risk

The threshold was selected using validation data to maximize the F1-score.

### 3. Individual Prediction Probability

The dashboard displays the estimated attrition probability as a percentage, allowing HR users to understand the model's confidence score.

### 4. SHAP-Based Explainability

The application uses SHAP to explain individual predictions by identifying the employee features that contribute positively or negatively to the model's output.

### 5. Model Performance Dashboard

The application presents important evaluation metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### 6. Interactive HR Analytics Interface

A Streamlit-based dashboard provides separate sections for prediction, model insights, and project information.

---

## Application Modules

| Module | Description |
|---|---|
| Overview | Displays key model performance indicators and project information. |
| Attrition Prediction | Accepts employee information and generates attrition predictions. |
| Model Insights | Presents evaluation metrics, confusion matrix, and threshold analysis. |
| SHAP Explainability | Explains the features influencing individual predictions. |
| About Project | Describes the project objectives, methodology, and technology stack. |

---

## Machine Learning Workflow

The project follows a structured machine learning pipeline.

### Step 1: Data Collection

The project uses a dataset containing approximately **100,000 employee records**.

### Step 2: Data Cleaning

The raw dataset is processed to improve data quality and consistency.

Key preprocessing operations include:

- Handling missing values.
- Removing irrelevant columns.
- Preparing categorical and numerical variables.
- Excluding potential data leakage columns such as employee ID, exit date, and exit reason.

### Step 3: Exploratory Data Analysis

Exploratory analysis is performed to understand employee characteristics and identify patterns associated with attrition.

### Step 4: Feature Engineering

Additional features are created to capture relationships between employee tenure, promotion history, and overtime.

### Step 5: Data Preprocessing

The dataset is prepared for machine learning using:

- Median imputation for missing numerical values.
- StandardScaler for feature scaling.
- One-hot encoding for categorical variables.

### Step 6: Model Training

Multiple classification algorithms are trained and evaluated.

### Step 7: Threshold Optimization

The decision threshold is tuned using a separate validation dataset rather than relying on the default 0.50 threshold.

### Step 8: Model Evaluation

The final model is evaluated on an independent test dataset.

### Step 9: Model Explainability

SHAP is integrated to explain the contribution of individual features to model predictions.

### Step 10: Deployment

The trained model and associated configuration are saved using Joblib and loaded into the Streamlit application.

---

## Dataset

The current training workflow uses approximately **100,000 employee records** from `dataset2.csv`.

The dataset contains employee-related attributes and an attrition target variable.

### Important Dataset Attributes

| Category | Features |
|---|---|
| Employee Information | Age, job level |
| Employment Information | Years at company, hire year |
| Compensation | Monthly income |
| Work Environment | Overtime, work mode, department |
| Employee Experience | Job satisfaction, performance rating |
| Career Progression | Years since promotion |
| Location | Distance from home |
| Target Variable | Attrition |

### Data Leakage Prevention

The following columns are excluded from model training:

- `employee_id`
- `exit_date`
- `exit_reason`

These columns are excluded because they could reveal information that would not legitimately be available when predicting future attrition.

---

## Feature Engineering

Two additional features are created to capture employee-related patterns.

### 1. Promotion Tenure Ratio

Measures the proportion of an employee's time at the organization spent since their last promotion.

\[
\text{Promotion Tenure Ratio} =
\frac{\text{Years Since Promotion}}{\text{Years at Company}}
\]

The feature is calculated only when years at the company is greater than zero and the promotion tenure does not exceed company tenure.

### 2. Overtime Tenure

Captures the interaction between overtime status and employee tenure.

\[
\text{Overtime Tenure} =
\text{Overtime} \times \text{Years at Company}
\]

These engineered features are included alongside the original predictors.

The final model uses **23 predictor features**.

---

## Models Evaluated

Four machine learning classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier
4. Gradient Boosting Classifier

The models were compared using accuracy, precision, recall, F1-score, and ROC-AUC.

### Initial Model Comparison

The following results were obtained during the initial model comparison at a decision threshold of 0.50.

| Model | Accuracy | Attrition Precision | Attrition Recall | Attrition F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 90.36% | 87% | 51% | 0.64 | 0.8898 |
| Random Forest | 87.99% | 83% | 37% | 0.51 | 0.8485 |
| Decision Tree | 86.98% | 81% | 30% | 0.44 | 0.8223 |
| Gradient Boosting | 90.33% | 96% | 45% | 0.61 | 0.8854 |

Logistic Regression demonstrated a stronger balance between recall, F1-score, and ROC-AUC. It was selected for the final prediction pipeline.

---

## Final Model Performance

### Train, Validation, and Test Split

The dataset was divided using stratified sampling:

| Dataset | Percentage |
|---|---:|
| Training | 64% |
| Validation | 16% |
| Testing | 20% |

The validation dataset was used for decision threshold optimization, while the test dataset was reserved for final evaluation.

### Threshold Optimization

The default classification threshold of 0.50 was evaluated against alternative thresholds.

The threshold of **0.40** produced the highest F1-score on the validation dataset.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.25 | 57.1% | 71.1% | 0.633 |
| 0.30 | 62.8% | 65.7% | 0.642 |
| 0.35 | 69.9% | 60.7% | 0.650 |
| **0.40** | **76.3%** | **57.3%** | **0.655** |
| 0.45 | 82.6% | 53.7% | 0.651 |
| 0.50 | 88.0% | 50.3% | 0.640 |

### Final Test Results

<details>
<summary><strong>Model Evaluation Metrics</strong></summary>

| Metric | Score |
|---|---:|
| Accuracy | 89.75% |
| Precision (Attrition) | 75.9% |
| Recall (Attrition) | 57.8% |
| F1-score (Attrition) | 0.66 |
| ROC-AUC | 0.8898 |
| Decision Threshold | 0.40 |

</details>

### Confusion Matrix

| | Predicted: No Attrition | Predicted: Attrition |
|---|---:|---:|
| Actual: No Attrition | 15,992 | 622 |
| Actual: Attrition | 1,428 | 1,958 |

The model correctly identified 1,958 employees who belonged to the attrition class in the test dataset.

The recall of approximately 57.8% indicates that the model identified around 58 out of every 100 actual attrition cases in the test set.

**Why Recall Matters:** In employee attrition prediction, false negatives represent employees who are predicted as lower risk even though they belong to the attrition class. Improving recall can help organizations identify more potential attrition cases, although this may also increase false positives.

---

## Model Explainability with SHAP

AttriSense incorporates SHAP (SHapley Additive exPlanations) to improve the interpretability of machine learning predictions.

Rather than presenting only an attrition probability, SHAP helps explain which features influenced an individual prediction.

### Global Feature Importance

The mean absolute SHAP values calculated during model analysis identified the following influential features:

| Feature | Mean Absolute SHAP Value |
|---|---:|
| Years at Company | 8.3737 |
| Hire Year | 7.9564 |
| Overtime | 0.4251 |
| Job Satisfaction | 0.3897 |
| Job Level | 0.2203 |
| Years Since Promotion | 0.2062 |
| Monthly Income | 0.1999 |
| Distance from Home | 0.1733 |

### Individual Prediction Explanation

For an individual employee, SHAP values help identify:

- Features that increase the model's attrition score.
- Features that decrease the model's attrition score.
- The relative magnitude of each feature's contribution.

Positive SHAP contributions increase the model output, while negative contributions decrease it.

**Important:** SHAP explains the model's behavior. It does not establish that a particular employee attribute causes attrition. Also, SHAP values for this Logistic Regression model are expressed in the model's output space (log-odds), not directly as percentage-point changes in probability.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| SHAP | Model explainability |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Model serialization and loading |
| Jupyter Notebook | Data analysis and model development |

---

## Project Structure

```text
AttriSense/
│
├── data/
│   ├── raw/
│   │   └── dataset2.csv
│   │
│   └── processed/
│       ├── final_dataset_100k.csv
│       └── feature_engineered_100k.csv
│
├── models/
│   └── attrisense_model.pkl
│
├── notebooks/
│   ├── dataclean.ipynb
│   ├── feature.ipynb
│   └── modeltraining.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation and Setup

Follow these steps to run AttriSense locally.

### Prerequisites

Make sure the following are installed:

- Python 3.10 or later
- pip
- Git (optional)

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd AttriSense
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the main dependencies:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn shap joblib
```

### 4. Verify the Model File

Ensure that the trained model bundle exists at:

```text
models/attrisense_model.pkl
```

The model bundle contains:

- Trained Logistic Regression pipeline
- Optimized decision threshold
- Expected feature list
- SHAP background data

The application requires this trained model bundle to generate predictions.

---

## Running the Application

From the project root directory, execute:

```bash
streamlit run app.py
```

Streamlit will start a local development server. Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The AttriSense dashboard should now be accessible in your browser.

---

## How to Use

1. Launch the Streamlit application.
2. Open the **Attrition Prediction** section.
3. Enter the employee's information in the prediction form.
4. Submit the details to generate a prediction.
5. Review the estimated attrition probability.
6. Check the classification against the 40% decision threshold.
7. Expand the SH
