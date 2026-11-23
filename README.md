
# Diabetes Risk Prediction Using Machine Learning

## 1. Project Overview

This project focuses on developing a machine learning-based system to predict diabetes status using health-related indicators from the CDC Diabetes Health Indicators dataset.

The project compares multiple individual machine learning classification algorithms and selects Linear Support Vector Machine (Linear SVM) as the final deployment model.

A Streamlit web application is developed to allow users to enter health-related information and obtain predictions.

## 2. Problem Statement

Diabetes is a major public health concern. Identifying individuals who may belong to a diabetes or prediabetes group based on health indicators can support early risk screening.

The objective of this project is to develop a binary classification model that analyzes health-related factors and classifies individuals into two groups:

- No Diabetes
- Prediabetes/Diabetes

**Problem Type:** Binary Classification

## 3. Project Objectives

- Analyze health indicators associated with diabetes.
- Perform data cleaning and exploratory data analysis.
- Identify important features using Mutual Information.
- Compare individual machine learning classification algorithms.
- Optimize model hyperparameters.
- Evaluate model performance using multiple classification metrics.
- Analyze decision thresholds to improve recall and F1 score.
- Deploy the selected Linear SVM model using Streamlit.

## 4. Dataset Information

**Dataset Name:** CDC Diabetes Health Indicators Dataset

**Dataset Source:** Kaggle

**Dataset Link:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

### Dataset Details

| Description | Details |
|---|---|
| Total Records | 70,692 |
| Total Columns | 22 |
| Input Features | 21 |
| Target Variable | Diabetes_binary |
| Problem Type | Binary Classification |

### Target Variable

The target variable is `Diabetes_binary`.

| Value | Description |
|---|---|
| 0 | No Diabetes |
| 1 | Prediabetes or Diabetes |

The positive class represents individuals classified in the combined prediabetes or diabetes category.

### Input Features

The dataset contains 21 health-related features:

1. HighBP
2. HighChol
3. CholCheck
4. BMI
5. Smoker
6. Stroke
7. HeartDiseaseorAttack
8. PhysActivity
9. Fruits
10. Veggies
11. HvyAlcoholConsump
12. AnyHealthcare
13. NoDocbcCost
14. GenHlth
15. MentHlth
16. PhysHlth
17. DiffWalk
18. Sex
19. Age
20. Education
21. Income

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 6. Exploratory Data Analysis

The following exploratory data analysis techniques were performed:

- Dataset structure and data type analysis
- Missing value analysis
- Duplicate record analysis
- Target variable distribution
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Outlier analysis
- Feature relationship analysis

## 7. Data Preprocessing

The following preprocessing operations were performed:

- Separation of input features and target variable
- Stratified train-test split
- Standard scaling of numerical and ordinal-coded features
- Retaining binary features without scaling
- Mutual Information feature analysis

The preprocessing operations are included in the saved machine learning pipeline.

## 8. Machine Learning Algorithms

The following individual classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Linear Support Vector Machine (Linear SVM)
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes
6. Linear Discriminant Analysis (LDA)
7. Quadratic Discriminant Analysis (QDA)

**Final Selected Model:** Linear Support Vector Machine (Linear SVM)

## 9. Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- Balanced Accuracy
- ROC-AUC
- PR-AUC
- Confusion Matrix

### Initial Linear SVM Results

| Metric | Score |
|---|---:|
| Accuracy | 74.56% |
| Precision | 73.44% |
| Recall | 76.94% |
| F1 Score | 75.15% |
| Specificity | 72.18% |
| Balanced Accuracy | 74.56% |
| ROC-AUC | 82.30% |
| PR-AUC | 79.27% |

### Decision Threshold Optimization

The Linear SVM decision threshold was analyzed to improve the balance between precision and recall.

**Selected Threshold:** -0.20

Exploratory threshold results:

| Metric | Score |
|---|---:|
| Accuracy | 74.02% |
| Precision | 68.72% |
| Recall | 88.19% |
| F1 Score | 77.24% |

The selected threshold increased recall while reducing precision.

**Note:** These threshold results were obtained during test-set exploration and should be considered exploratory rather than an unbiased final performance estimate.

## 10. Project Structure

```text
diabetes_prediction_project/
│
├── models/
│   ├── diabetes_linear_svm_pipeline.pkl
│   └── diabetes_threshold.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd diabetes_prediction_project
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 12. Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Select `app.py` as the main application file.
6. Deploy the application.

## 13. Future Enhancements

- Implement SHAP-based model explainability.
- Compare full-feature and feature-selected models.
- Evaluate the model on an independent dataset.
- Improve the Streamlit user interface.
- Add model interpretation visualizations.
- Validate model performance using data with real-world diabetes prevalence.

## 14. Disclaimer

This project is developed for academic, educational, and research purposes only.

The model identifies patterns associated with the dataset's combined prediabetes/diabetes target category. It is not a medical diagnostic tool and should not replace professional medical advice.

## 15. Author

Developed as a Machine Learning and Data Science project.
