import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Fake but structured training data (safe starter)
# [mfccMean, spectralCentroid, zeroCrossingRate, rmsEnergy]
X = np.array([
    [-32, 1500, 0.08, 0.015],   # human
    [-28, 1600, 0.07, 0.020],   # human
    [-35, 1400, 0.09, 0.012],   # human
    [-10, 4000, 0.30, 0.080],   # AI
    [-12, 3800, 0.28, 0.070],   # AI
    [-15, 4200, 0.32, 0.090]    # AI
])

y = np.array([1, 1, 1, 0, 0, 0])  # 1 = HUMAN, 0 = AI

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

dump(model, "voice_model.pkl")
print("✅ Model trained and saved as voice_model.pkl")
