import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model from the file
with open('C:/Users/admin/Desktop/CodSoft Internship 2024/xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Function to predict fraud on new data
def predict_fraud(transaction):
    transaction = np.array([transaction])
    dmatrix = xgb.DMatrix(transaction)
    prediction = model.predict(dmatrix)
    return int(prediction[0])
from flask import Flask, request, jsonify

app = Flask(__name__)

app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    transaction = np.array(data['transaction'])
    prediction = predict_fraud(transaction)
    return jsonify({'fraud': prediction})

if __name__ == '__main__':
    app.run(debug=True)

