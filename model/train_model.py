import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create model folder if it does not exist
os.makedirs("model", exist_ok=True)

data = {
    "cgpa": [6.5, 7.2, 8.0, 8.5, 9.1, 5.8, 6.8, 7.9, 8.8, 9.5],
    "skills": [3, 4, 6, 7, 9, 2, 4, 6, 8, 10],
    "projects": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    "placed": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["cgpa", "skills", "projects"]]
y = df["placed"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model/placement_model.pkl")

print("Model trained and saved successfully!")