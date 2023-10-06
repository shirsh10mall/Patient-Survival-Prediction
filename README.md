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
1. **Initial Model Exploration:**
   - A comprehensive exploration of various machine learning models was conducted. Models such as Logistic Regression, Decision Tree, Random Forest, XGBoost, KNeighbours, Gaussian Naive Bayes, Support Vector Support, and Multi-layer Perceptron were employed with their default parameters as provided by the sklearn library.
   -  Results of Initial Model Exploration:  The results revealed that all Tree-Based models exhibited a significant issue of overfitting, with a test score approximately at 0.89. Gaussian Naive Bayes, on the other hand, displayed a promising AUC Score of 0.85 on both training and testing sets, demonstrating no overfitting. However, the remaining models did not perform as expected.
   - SVM Performance Challenge: Notably, the Support Vector Machine (SVM) exhibited sluggish performance due to the dataset's substantial size, rendering it impractical for efficient computation and analysis. The model's computational demands were too high for the dataset in question, leading to operational challenges.


2. **Handling Imbalance:**
   - SMOTE techniques for oversampling will be experimented with to handle class imbalance. However, it is observed that this technique does not significantly improve performance, and hence, it's decided not to use any oversampling technique for further analysis.

3. **Fine Tuning Hyperparameters using Optuna**:
   ##### Objective: The objective is to find the optimal set of hyperparameters that maximize the Test AUC (Area Under the Curve) score for each model.
   ##### Technique: Optuna, a hyperparameter optimization framework, is utilized to efficiently search for the best hyperparameters.
   - Iteration Rounds: More than 200 rounds of attempts are conducted for each model, emphasizing a thorough and exhaustive search for optimal hyperparameters.
   ##### Results:
   - Performance Improvement: The process significantly improves the performance of all models.
   -  Overfitting Mitigation: Overfitting is mitigated, indicating a more robust and generalizable model.
   - AUC Score Changes: The AUC score on both the training and testing sets is reduced.
   - Overfitting Factor: The AUC overfitting factor is observed to be between 0.001 and 0.002, suggesting a minor overfitting concern despite improvements in performance.

4. **Created a Custom Deep Neural Network**:

#### 4.1. Created a Simple Neural Network:
- A basic neural network architecture was designed.
- **Result:** Achieved an AUC score of 0.87 on both the training and testing sets, indicating no overfitting.

#### 4.2. Enhanced Network with Batch Normalization, Dropout Layers, and Batch Size Tweaking:
- Additional layers, including batch normalization and dropout, were added to improve model robustness.
- The batch size was tweaked to optimize training.
- **Result:** Attained an AUC score of 0.86 on both training and testing sets, demonstrating no overfitting.

#### 4.3. Fine-Tuned Hyperparameters Using Keras Tuner:
- Utilized Keras Tuner to tune the hyperparameters of the neural network finely.
- **Result:** Achieved an AUC score of 0.866 on both training and testing sets. However, no significant improvement was observed compared to the previous neural network, indicating no overfitting.


5. Sure, let's break down the ensemble model creation and its variations into a pointwise manner:

### Ensemble Model Creation and Variations:

5. **Create a Custom-Built Stacked Ensemble Model:**
   - Combining all the fine-tuned models to create a robust ensemble model for enhanced predictive performance.

5.1. **Ensemble 3 with Neural Network as Meta-Model:**
   - Utilized all fine-tuned models as base models for the stacked ensemble.
   - Introduced a simple neural network as the meta-model.
   - Used output predictions of all base models as input to the meta-model.
   - Results:
     - Observed a minor level of overfitting.
     - Achieved an AUC score of 0.865.
     - AUC overfitting factor was approximately 0.001.

5.2. **Ensemble 3 with Logistic Regression as Meta-Model:**
   - Employed all fine-tuned models as base models for the stacked ensemble.
   - Employed logistic regression as the meta-model.
   - Results:
     - Experienced a slight degree of overfitting.
     - Attained an AUC score of 0.853.
     - AUC overfitting factor was approximately 0.001.

5.3. **Ensemble 3 and 4 with Weighted Output Predictions:**
   - Combined Ensemble 3 and Ensemble 4 models.
   - Implemented a weighted approach for the output predictions of the base models.
   - Each model's output was multiplied by a corresponding overfitting factor.
   - Results:
     - Demonstrated very minimal overfitting.
     - Achieved an AUC score of 0.86.

These variations in ensemble models showcase different strategies in utilizing base models and meta-models to achieve an optimal balance between predictive performance and overfitting, ultimately contributing to the overall robustness of the predictive model.

6. **Hyperparameter Tuning for XGBoost Model:**
   - **Approach:**
     - Focused on refining the XGBoost model's performance through meticulous hyperparameter tuning.
   - **Manual Range Setting:**
     - Defined a specific range for the hyperparameters, manually curating a set of values to explore during tuning.
   - **Extensive Training Attempts:**
     - Conducted more than 500 training attempts, rigorously iterating through the defined hyperparameter range.
   - **Achieved AUC Score:**
     - Successfully achieved an impressive AUC score of 0.88, demonstrating the effectiveness of the tuned hyperparameters.
   - **Overfitting Mitigation:**
     - Ensured no overfitting was observed during the tuning process, highlighting the model's stability and generalization.
   - **Final Model Selection:**
     - Based on the exceptional performance and stability, selected this finely tuned XGBoost model as the final model for integration into the predictive system.


### Feature Selection, Creating Model Web App, and Model Explainability:

1. **Feature Selection:**
   - Identified and selected the top 8 features based on the feature importance derived from the XGBoost model, which was finalized as the key predictive model for this project.

2. **Model Web App Creation:**
   - Developed a user-friendly model web application using Streamlit, allowing users to input values for the 8 selected features via a form interface.
   - The application instantly processes these inputs and displays the model's prediction regarding patient survival probability.

3. **Deployment of the Model:**
   - Utilized Streamlit-Cloud to seamlessly deploy the model, making it accessible to users through the web without the need for complex setups or installations.

4. **Model Explainability:**
   - Employed SHAP (SHapley Additive exPlanations) values, a powerful technique in explainable AI, to assess the model's explainability and provide insights into how each feature contributes to the model's predictions.
   - This step enhanced the transparency and interpretability of the predictive model, ensuring that predictions could be understood and trusted by both professionals and users.



![image](https://github.com/shirsh10mall/Patient-Survival-Prediction/assets/87264071/ce289b10-9239-4468-a693-0250c22c5af0)
![image](https://github.com/shirsh10mall/Patient-Survival-Prediction/assets/87264071/d4a54333-bbc7-480e-bcf0-e4ae1f56ed61)


