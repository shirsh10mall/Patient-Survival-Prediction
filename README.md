# Patient-Survival-Prediction

Kaggle Notebooks 
 1. EDA: https://www.kaggle.com/code/shirshmall/eda-patient-survival-detection/notebook
  
 2. ML Models - https://www.kaggle.com/code/shirshmall/ml-models-patient-survival-detection
    
 3. Neural Network Model - https://www.kaggle.com/code/shirshmall/neural-network-model-patient-survival-detection
 
 4. Stacking Ensemble Model - https://www.kaggle.com/code/shirshmall/stacking-ensemble-model-patient-survival-detection
    
# Patient Survival Prediction using Machine Learning

## Description
Accurately predicting patient survival outcomes is crucial, especially in times of crises like the COVID-19 pandemic. Healthcare systems worldwide grapple with overloaded hospitals and limited medical histories for incoming patients. The Intensive Care Units (ICUs) often receive patients in distress or confusion, making it challenging to gather essential medical information. Rapid decisions are required for effective care, but this is hindered by the delayed transfer of medical records, exacerbating patient management challenges. This project aims to create a predictive model that uses patient data to anticipate survival outcomes, offering critical support to healthcare providers.

## Dataset
The dataset comprises 91,713 records with 186 features, encompassing patient information and medical history.

## Problem Statement
The main objective is to predict 'hospital_death', a binary variable representing patient survival. The model uses 84 features to classify patients into survival categories.

Of course! Let's dive into more detailed descriptions for each aspect of the project.


### Approach:

#### Feature Identification and Handling Missing Values:
- Features will be meticulously categorized based on their types, such as numerical, categorical, or ordinal, and analyzed for any missing data.
- The missing values will be addressed using mean or median imputation, with careful consideration for potential data integrity concerns and minimizing distortions in the dataset.

#### Exploratory Data Analysis (EDA):
- A comprehensive EDA will be performed to uncover patterns, relationships, and key insights within the dataset.
- Utilization of visualization tools, such as plots and graphs, will be employed to gain valuable insights into the data. This will include distributions, correlations, and trends.

#### Feature Encoding:
- Categorical and ordinal features will be encoded suitably, ensuring that the encoding process minimizes information loss and avoids introducing bias into the model.

#### Metrics:
- **AUC Score:** This primary metric is chosen as it helps find the optimal threshold for classification that is beneficial for both classes.
- **F1 Score:** This secondary metric is important to ensure correct predictions in both classes, particularly focusing on minimizing false positives and false negatives.

#### Model Training and Their Results:
1. **Initial Model Evaluation:**
   - Multiple machine learning (ML) models will be explored, including Logistic Regression, Decision Tree, Random Forest, XGBoost, KNeighbours, Gaussian Naive Bayes, Support Vector Support, and Multi-layer Perceptron, utilizing their default parameters from the sklearn library.
   - Result: It is observed that all Tree-Based models tend to overfit significantly (with a test score around 0.89). Gaussian Naive Bayes shows promise with a 0.85 AUC Score on both training and testing sets (without overfitting). However, other models are not performing optimally.

2. **Handling Imbalance:**
   - SMOTE techniques for oversampling will be experimented with to handle class imbalance. However, it is observed that this technique does not significantly improve performance, and hence, it's decided not to use any oversampling technique for further analysis.

3. **Hyperparameter Tuning:**
   - Hyperparameters of all models will be fine-tuned using Optuna with the objective of maximizing the test AUC score, attempting more than 200 rounds of tuning for each model.
   - Result: The performance of all models improves and mitigates overfitting, but there's a reduction in the AUC score on both the training and testing sets. Very minimal overfitting is observed (AUC overfitting factor between 0.001 and 0.002).

4. **Custom Neural Network:**
   - A custom deep neural network is created and experimented with various architectures and hyperparameters.
   - Result: Achieved a promising AUC score of 0.87 on both training and testing sets without overfitting.

5. **Ensemble Model:**
   - A custom-built stacked ensemble model is designed by combining all the fine-tuned models.
   - Result: Achieved an improved AUC score of 0.88 with no overfitting.

#### Feature Selection, Creating Model Web App and Model Explainability:
1. **Feature Selection:**
   - Top 8 features are selected based on the XGBoost's (final model) feature importance.

2. **Model Web App:**
   - A model web app is created using Streamlit, allowing users to input values for the selected features, and the app will display the model's prediction.

3. **Deployment:**
   - The model is deployed using Streamlit-Cloud for accessibility and ease of use.

4. **Model Explainability:**
   - SHAP (SHapley Additive exPlanations) values are utilized to provide insights into the model's predictions, enhancing its explainability.

---
Feel free to adapt and augment this detailed plan to best suit your project requirements and objectives. If you need further elaboration or have specific additions in mind, please let me know!
![image](https://github.com/shirsh10mall/Patient-Survival-Prediction/assets/87264071/ce289b10-9239-4468-a693-0250c22c5af0)
![image](https://github.com/shirsh10mall/Patient-Survival-Prediction/assets/87264071/d4a54333-bbc7-480e-bcf0-e4ae1f56ed61)


