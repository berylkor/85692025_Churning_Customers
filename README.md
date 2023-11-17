# 85692025_Churning_Customers

<details open>
<summary><b>Table of Contents</b></summary>
Churn Prediction Model

# Overview
Content
Data Preprocessing

Data Preprocessing
Feature Selection
Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA)
Artificial Neural Network (ANN) Training and Testing

ANN Training and Testing
Grid Search for Hyperparameter Tuning

Grid Search for Hyperparameter Tuning
Exporting Model and Preprocessing Objects

Exporting Model and Preprocessing Objects
Usage

Requirements

</details>
<details>
<summary><b>Churn Prediction Model</b></summary>
Overview
This repository contains a Jupyter Notebook (85692025_Churning_Customers.ipynb) that focuses on predicting customer churn using a Telco customer dataset. The notebook covers the entire data analysis and modeling pipeline, from importing the dataset to training an artificial neural network. The key steps include data preprocessing, feature engineering, exploratory data analysis, and the implementation of an artificial neural network using Keras.

Content
Importing Datasets:

The notebook begins with importing the necessary libraries and loading the Telco customer churn dataset.
Relevant Features:

Drops irrelevant columns and performs label encoding on the target variable ('Churn').
Handles missing values in the 'TotalCharges' column.
</details>
<details>
<summary><b>Data Preprocessing</b></summary>
Data Preprocessing
Data Preprocessing:

Scales numeric features using StandardScaler.
Combines encoded categorical features and numeric features into a single dataframe.
Feature Selection:

Utilizes a RandomForestClassifier for feature importance.
Selects the top eleven most important features for the model.
</details>
<details>
<summary><b>Exploratory Data Analysis (EDA)</b></summary>
Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA):
Investigates the relationship between various features and customer churn using box plots and count plots.
</details>
<details>
<summary><b>Artificial Neural Network (ANN) Training and Testing</b></summary>
ANN Training and Testing
Artificial Neural Network (ANN) Training and Testing:
Implements a Keras Functional API model for predicting churn.
Splits the dataset into training, validation, and test sets.
Trains the model and evaluates its performance on the test set.
Uses AUC (Area Under the Receiver Operating Characteristic curve) as an additional evaluation metric.
</details>
<details>
<summary><b>Grid Search for Hyperparameter Tuning</b></summary>
Grid Search for Hyperparameter Tuning
Grid Search for Hyperparameter Tuning:
Performs hyperparameter tuning using GridSearchCV.
Explores different combinations of optimizers, random states, and batch sizes.
</details>
<details>
<summary><b>Exporting Model and Preprocessing Objects</b></summary>
Exporting Model and Preprocessing Objects
Exporting Model and Preprocessing Objects:
Saves the trained functional model, StandardScaler, and LabelEncoder for future use.
</details>
<details>
<summary><b>Usage</b></summary>
Open the Jupyter Notebook (85692025_Churning_Customers.ipynb) using Jupyter Notebook or Google Colab.
Run each cell sequentially to execute the code.
Follow the detailed comments and markdown cells for explanations of each step.
Modify parameters or experiment with different configurations as needed.
</details>
<details>
<summary><b>Requirements</b></summary>
Python 3.x
Jupyter Notebook
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, keras, tensorflow
