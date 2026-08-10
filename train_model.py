import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score

# Load dataset (adjust path as needed)
df = pd.read_excel("Aquaculture - Water Quality Dataset/WQD.xlsx")
df.columns = [c.strip() for c in df.columns]   # clean up column names / stray backticks

# =====================================================
# 1) BOD ESTIMATOR
# There's no physical BOD sensor on the ESP32 (real BOD needs a 5-day lab
# incubation test), so we estimate it from the 4 sensors we do have.
# =====================================================
Xb = df[["Turbidity (cm)", "DO(mg/L)", "pH`", "Temp"]]
yb = df["BOD (mg/L)"]

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.2, random_state=42
)

bod_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
bod_model.fit(Xb_train, yb_train)

print("BOD estimator R^2:", r2_score(yb_test, bod_model.predict(Xb_test)))
print("(This is an approximation - accept that it will not be perfect")
print(" without a real BOD sensor/lab test.)\n")

with open("bod_estimator.pkl", "wb") as f:
    pickle.dump(bod_model, f)

# =====================================================
# 2) WATER QUALITY CLASSIFIER
# Feature order MUST match what app.py sends:
# [turbidity, do, ph, temp, bod]
# =====================================================
cols = ["Turbidity (cm)", "DO(mg/L)", "pH`", "Temp", "BOD (mg/L)"]
X = df[cols].values
y = df["Water Quality"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    max_depth=24,
    min_samples_split=7,
    n_estimators=162,
    random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Classifier test accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saved random_forest_model.pkl and bod_estimator.pkl")

# =====================================================
# Sanity check: full pipeline exactly as app.py runs it
# =====================================================
def classify(turbidity, do, ph, temp):
    bod_est = bod_model.predict([[turbidity, do, ph, temp]])[0]
    feats = np.array([[turbidity, do, ph, temp, bod_est]])
    pred = model.predict(feats)[0]
    print(f"turb={turbidity}, do={do}, ph={ph}, temp={temp} "
          f"-> bod_est={bod_est:.2f} -> prediction={pred}")

print("\nSanity checks:")
classify(70, 4.0, 7.8, 25)   # expect clean-ish -> 0
classify(22, 6.5, 7.8, 25)   # expect moderate -> 1
classify(10, 2.0, 6.0, 30)   # expect bad -> 2
