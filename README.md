# Patient-Survival-Prediction

Kaggle Notebook: https://www.kaggle.com/code/shirshmall/patient-survival-detection/

# Patient Survival Prediction using Machine Learning

## Description
Accurately predicting patient survival outcomes is crucial, especially in times of crises like the COVID-19 pandemic. Healthcare systems worldwide grapple with overloaded hospitals and limited medical histories for incoming patients. The Intensive Care Units (ICUs) often receive patients in distress or confusion, making it challenging to gather essential medical information. Rapid decisions are required for effective care, but this is hindered by the delayed transfer of medical records, exacerbating patient management challenges. This project aims to create a predictive model that uses patient data to anticipate survival outcomes, offering critical support to healthcare providers.

## Dataset
The dataset comprises 91,713 records with 186 features, encompassing patient information and medical history.

## Problem Statement
The main objective is to predict 'hospital_death', a binary variable representing patient survival. The model uses 84 features to classify patients into survival categories.

## Approach

1. **Feature Identification and Handling Missing Values:** Features are categorized based on type and analyzed for missing data. Missing values are addressed by mean or median imputation, but this process raises concerns about data integrity and potential distortions.

2. **Exploratory Data Analysis (EDA):** EDA uncovers patterns, correlations, and insights. While visualization tools provide valuable insights, the sheer volume of data can sometimes obscure subtle relationships.

3. **Feature Encoding:** Categorical and ordinal features are encoded, but this approach has limitations. Encoding can lead to information loss or introduce bias.

4. **Handling Class Imbalance:** The application of SMOTE appears promising but can inadvertently generate synthetic samples that don't accurately represent real-world patient data, potentially distorting model predictions.

5. **Model Training and Evaluation:** Multiple classification models, including XGBoost, RandomForest, DecisionTree, KNeighbors, Logistic Regression, MLPClassifier, and SVC, are trained and evaluated. Some models display overfitting, indicating the need for more refined techniques.

6. **Deep Neural Network (DNN):** DNNs are introduced to address overfitting. While DNNs mitigate the issue to some extent, they also introduce complexity and require careful tuning to achieve optimal results.

7. **SHAP Values for Model Interpretation:** SHAP values offer insights into feature impacts but can be complex to interpret, requiring additional effort to convey their significance.

## Impact
This project underscores the significance of predictive models in healthcare. While it shows potential to aid medical decision-making, it also highlights the complexities and limitations in handling sensitive medical data. The deployment and integration of models into healthcare workflows require careful consideration of ethical, legal, and operational aspects. Moreover, the project emphasizes the need for continuous monitoring and improvement to ensure models remain accurate and relevant in dynamic healthcare environments.

#### Classification Results for DNN Model

Epoch 30/100 (Early Stopping) - loss: 0.1898 - accuracy: 0.9296 - auc: 0.8938 - val_loss: 0.2112 - val_accuracy: 0.9226 - val_auc: 0.8804
