from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    # Replace with the actual path to your dataset
    credit_card_data = pd.read_csv(r'C:\Users\VICTUS\Downloads\archive (6).zip')
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    X = new_dataset[['Time', 'Amount']]
    Y = new_dataset['Class']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    return X_train, X_test, Y_train, Y_test

# Train the model
X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
model = LogisticRegression()
model.fit(X_train, Y_train)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time = float(request.form['time'])
    amount = float(request.form['amount'])
    
    input_features = np.array([[time, amount]])
    prediction = model.predict(input_features)
    
    if prediction[0] == 0:
        result = "Normal Transaction"
    else:
        result = "Fraudulent Transaction"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
