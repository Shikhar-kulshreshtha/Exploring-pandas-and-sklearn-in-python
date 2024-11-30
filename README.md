## Heart Disease Prediction Using Machine Learning
This repository demonstrates the application ofbasic machine learning techniques—Linear Regression, K-Nearest Neighbors (KNN), and Logistic Regression—using Python and libraries such as scikit-learn, matplotlib, and seaborn. The goal is to predict the presence or absence of heart disease based on various attributes of patients.
## Project Overview
Heart disease is a leading cause of death globally, and predicting its occurrence can aid in early detection and prevention. This project applies three machine learning models to classify patients into two categories:

**1**: Patient has heart disease
**0**: Patient does not have heart disease

## Dataset
The dataset contains attributes like age, cholesterol level, blood pressure, etc., along with a binary outcome variable (HeartDisease).

**Attributes include:**
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Levels
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- Resting ECG
- ST Depression
- Fasting Blood Sugar
- OldPeak

We convert the categorical attributes in the database to dummy values to continue.

Credits for the database : - https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

  ## Analysis and results
  - We observe that factors such as Max HR and Fasting Blood sugar have the most impact during our analysis of the database. However, if we are to convert categorical attributes of 2 possible inputs to binary inputs, we can see that an ST_Slope "flat" and having exercise induced Angina greatly enhances the chances of having a heart disease.
  - Other attributes also contribute to the same, enough to the extent that they can not be ignored.
  - I chose to drop the attribute "sex" , analysis concludes males have a higher chance, however, that can simply be due to the sample space of the given database
  - The models are successfully applied and out scores are obtained, for the given database, due to a binary outcome, it is always advisable to use Logistical Regression and that brought out the best results in this case as well.
  - train and test size is set to 80 and 20 percent respectively, and we are able to achieve a prediction score upto 75% or 0.75.
  
