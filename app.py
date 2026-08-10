from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# main water-quality classifier (turbidity, do, ph, temp, bod -> 0/1/2)
model = pickle.load(open("random_forest_model.pkl", "rb"))

# BOD estimator: there is no physical BOD sensor on the ESP32 (real BOD
# requires a 5-day lab incubation test), so BOD is estimated here from
# the 4 real sensor readings using a model trained on the aquaculture
# dataset. This is an approximation (R^2 ~ 0.43 on held-out data) -
# far from perfect, but much better than a made-up formula, and it's
# the best that's achievable without an actual BOD sensor.
bod_model = pickle.load(open("bod_estimator.pkl", "rb"))


@app.route('/')
def home():
    return "Water Quality API Running"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # real sensor inputs only
    turbidity = float(data['turbidity'])
    do = float(data['do'])
    ph = float(data['ph'])
    temp = float(data['temp'])

    # estimate BOD from the 4 real readings (no physical BOD sensor exists)
    bod_features = np.array([[turbidity, do, ph, temp]])
    bod_estimated = float(bod_model.predict(bod_features)[0])

    # same feature order the classifier was trained on
    features = np.array([[turbidity, do, ph, temp, bod_estimated]])
    prediction = model.predict(features)

    return jsonify({
        "prediction": str(prediction[0]),
        "bod_estimated": round(bod_estimated, 2)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
