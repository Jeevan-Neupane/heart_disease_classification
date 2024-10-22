# Heart Disease Classification using Machine Learning

## Project Overview
This project aims to predict the presence of heart disease using machine learning algorithms. The dataset consists of 14 features, including patient age, sex, chest pain type, resting blood pressure, cholesterol levels, and more. The model classifies whether a patient is likely to have heart disease (target value of 1) or not (target value of 0).

## Dataset
The dataset contains 303 entries, each with 14 features:

- **Age**: Age of the patient
- **Sex**: Gender of the patient (1 = Male, 0 = Female)
- **CP**: Chest pain type (0-3)
- **Trestbps**: Resting blood pressure (in mm Hg)
- **Chol**: Serum cholesterol in mg/dl
- **Fbs**: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
- **Restecg**: Resting electrocardiographic results (0-2)
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise-induced angina (1 = Yes, 0 = No)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: The slope of the peak exercise ST segment (0-2)
- **Ca**: Number of major vessels (0-3) colored by fluoroscopy
- **Thal**: Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)
- **Target**: Heart disease (1 = Disease, 0 = No disease)

The target column is the primary label for prediction.

## Exploratory Data Analysis (EDA)
In the initial stage, exploratory data analysis was performed to understand the distribution and relationships of the features. Key insights include:
- The distribution of heart disease across different gender and age groups.
- Relationships between features like chest pain type and the presence of heart disease.
- Correlations between various features and the target variable.

## Models Used
Three machine learning models were trained for heart disease classification:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest Classifier**

These models were evaluated using cross-validation and performance metrics such as accuracy, precision, recall, and F1-score.

## Model Performance
- **Logistic Regression** achieved the highest accuracy (~88%) on the test data.
- **Random Forest Classifier** was also highly effective, with an accuracy of ~87%.
- **K-Nearest Neighbors (KNN)** showed lower performance (~69%).

## Hyperparameter Tuning
To improve model performance, hyperparameter tuning was performed using:
- **RandomizedSearchCV** for both Logistic Regression and Random Forest models.
- **GridSearchCV** to fine-tune the hyperparameters further.

These techniques helped in finding the optimal model parameters for better prediction accuracy.

## Evaluation Metrics
The following metrics were used to evaluate the models:
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.

A confusion matrix and ROC curve were also plotted to visually assess model performance.

## Feature Importance
Logistic Regression was used to determine the importance of various features in predicting heart disease. Some key features that contributed significantly include:
- **Chest pain type (cp)**
- **Number of major vessels (ca)**
- **Maximum heart rate achieved (thalach)**
- **Exercise-induced angina (exang)**

## Conclusion
This project demonstrates how machine learning models can be effectively applied to classify heart disease risk based on medical data. With appropriate data preprocessing, model selection, and hyperparameter tuning, we achieved promising results. Logistic Regression provided the best balance of accuracy and interpretability.

## Installation
To run the project, clone this repository and install the required Python libraries:

```bash
git clone https://github.com/your-repo-url/heart-disease-classification.git
cd heart-disease-classification
