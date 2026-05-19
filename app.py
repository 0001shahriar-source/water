from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Water Quality API Running"

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    ph = float(data['ph'])
    temp = float(data['temp'])
    do = float(data['do'])
    turbidity = float(data['turbidity'])

    features = np.array([[ph, temp, do, turbidity]])

    prediction = model.predict(features)

    return jsonify({
        "prediction": str(prediction[0])
    })

if __name__ == '__main__':
    app.run(debug=True)