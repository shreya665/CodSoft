import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('C:/Users/admin/Desktop/Customer churn prediction/Gradient_model.pkl')

# Data preprocessing (handle missing values, encode categorical variables)

# Feature selection and engineering

# Split data into training and testing sets
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('C:/User/admin/Desktop/Customer churn prediction/Gradient_model.pkl')

app.route('/')
def home():
    return render_template('index.html')

app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    int_features = [float(x) for x in request.form.values()]
    final_features = [int_features]  # If your model expects a 2D array

    # Make prediction
    prediction = model.predict(final_features)

    # Return prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

