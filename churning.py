import streamlit as str
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import joblib
import pickle

# loading scalar 
with open('scalar.pkl', 'rb') as f:
    loaded_scalar = pickle.load(f)

# loading encoder
load_encoder = LabelEncoder()
# load_encoder.classes = np.load('classes.npy', allow_pickle= True)

# loading functional model with the best features
functional_model = load_model('func_model.h5')
str.title('Customer churn prediction by 85692025')


# main function
def main():
    TotalCharges = str.number_input('Customer total charges', 0, key='TotalCharges')
    MonthlyCharges = str.number_input('Customer monthly charges', 0, key= 'MonthlyCharges')
    Tenure = str.number_input('Customer tenure', 0, key = 'tenure')
    Contract = str.radio('What is the length of customer contract', ['Month-to-month', 'One year Contract', 'Two year'], key='Contract')
    PaymentMethod = str.selectbox('What is customer payment method', ['Electronic check', 'Mailed check', 'Bank transfer(automatic)', 'Credit card(automatic)'], key = 'PaymentMethod')
    TechSupport = str.selectbox('Tech Support?', ['Yes', 'No', 'No internet service'], key='TechSupport')
    OnlineSecurity = str.selectbox('Do the customer have online security?', ['Yes', 'No', 'No internet service'], key='OnlineSecurity')
    Gender = str.radio('Gender of customer', ['Female', 'Male'], key='Gender')
    InternetService = str.selectbox('Internet Service available',['DSL', 'Fiber optic', 'No'], key='InternetService')
    OnlineBackup = str.selectbox('Online Backup', ['Yes', 'No', 'No internet service'], key = 'OnlineBackup')
    DeviceProtection = str.selectbox('Device protection provided', ['Yes', 'No', 'No internet service'], key= 'DeviceProtection')

    if str.button('Predict', key = 'predict_churn_button'):
        user_numeric_inputs = np.array([[TotalCharges,MonthlyCharges,Tenure]])
        user_object_inputs = np.array([[Contract,PaymentMethod, TechSupport, OnlineSecurity, Gender, InternetService, OnlineBackup,DeviceProtection]])
        load_encoder.fit(user_object_inputs.flatten())

        object_transformed = load_encoder.transform(user_object_inputs.flatten())
        object_reshaped = object_transformed.reshape(-1, 1)
        numeric_reshaped = user_numeric_inputs.reshape(-1, 1)

        data = np.concatenate((object_reshaped, numeric_reshaped), axis = 0).T
        scaled_data = loaded_scalar.transform(data)
        prediction_prob = functional_model.predict(scaled_data)
        prediction = np.argmax(prediction_prob)
        confidence = float(prediction_prob[np.argmax(prediction)] * 100)

        print("Prediction Probabilities:", prediction_prob)
        print("Predicted Class:", prediction)
        print("Confidence Score:", confidence)

        if prediction == 1:
            str.write('The customer may churn')
        elif prediction == 0:
            str.write('The customer may not churn')
        str.write(f'Confidence score: {confidence:.2f}%')

main()

