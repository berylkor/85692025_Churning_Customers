
## Overview

This repository contains a Jupyter Notebook, 85692025_Churning_Customers.ipynb that focuses on predicting customer churn using a Telco customer dataset. The notebook covers the entire data analysis and modeling pipeline, from importing the dataset to training an artificial neural network. The key steps include data preprocessing, feature engineering, exploratory data analysis (EDA), and implementing a Keras ANN.

# Content
Importing Datasets:

The notebook begins with importing the necessary libraries and loading the customer churn dataset.

Relevant Features:
Drops irrelevant columns and performs label encoding on the target variable ('Churn').
Handles missing values in the 'TotalCharges' column.

## Data Preprocessing
Data Preprocessing:

Scales numeric features using StandardScaler.
Combines encoded categorical features and numeric features into a single dataframe.
Feature Selection:

Utilizes a RandomForestClassifier for feature importance.
Selects the top eleven most important features for the model.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA):
Investigates the relationship between various features and customer churn using box plots and count plots.

## ANN Training and Testing
Artificial Neural Network (ANN) Training and Testing:
Implements a Keras Functional API model for predicting churn.
Splits the dataset into training, validation, and test sets.
Trains the model and evaluates its performance on the test set.
Uses AUC as an additional evaluation metric.

## Grid Search for Hyperparameter Tuning
Grid Search for Hyperparameter Tuning:
Performs hyperparameter tuning using GridSearchCV.
Explores different combinations of optimizers, random states, and batch sizes.


## Exporting Model and Preprocessing Objects
Exporting Model and Preprocessing Objects:
Saves the trained functional model, StandardScaler, and LabelEncoder for future use.

## Usage
Open the Notebook using Jupyter Notebook or Google Colab.
Run each cell sequentially to execute the code.
Follow the comments and markdown cells for explanations of each step.
Modify parameters or experiment with different configurations as needed.

## Requirements
  
* Python 3.x
* Jupyter Notebook
* Libraries:
   *  Pandas
   *  NumPy
   *  scikit-learn
   *  matplotlib
   *  seaborn
   *  keras
   *  TensorFlow

## Video demonstration of the web app
https://drive.google.com/drive/folders/1YM_ybO63K5VtVA81NcJqFHsqYQ42UqvZ?usp=sharing
  
## Web app Link 
https://vfqgscet3drdgmqdbwm7xa.streamlit.app/
