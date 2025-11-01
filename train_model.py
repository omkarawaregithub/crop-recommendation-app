# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. load dataset
df = pd.read_csv("data/Crop_recommendation.csv")  # change path if needed
# Inspect columns
print(df.head())
# Typical columns: N, P, K, temperature, humidity, ph, rainfall, label

# 2. features and target
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

# 3. train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# 6. save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/crop_model.joblib")
print("Model trained and saved as model/crop_model.joblib")
