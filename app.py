from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("random_forest_model.pkl", "rb"))

@app.route('/')
def home():
    return "Water Quality API Running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    # inputs
    turbidity = float(data['turbidity'])
    do = float(data['do'])
    ph = float(data['ph'])
    temp = float(data['temp'])
    bod = float(data['bod'])

    # same order as training
    features = np.array([[turbidity, do, ph, temp, bod]])

    prediction = model.predict(features)

    return jsonify({
        "prediction": str(prediction[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
