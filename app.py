# app.py
import streamlit as st

# Add error handling for imports
try:
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    st.error(f"Required package missing. Please run: pip install -r requirements.txt")
    st.error(f"Error: {e}")
    st.stop()

# Add error handling for model loading
try:
    model = joblib.load("model/crop_model.joblib")  # Updated path
except FileNotFoundError:
    st.error("Model file not found. Please ensure crop_model.joblib exists in the root directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Crop Recommendation Dashboard")
st.write("Enter soil and climate values to get crop suggestions.")

# Sidebar inputs
st.sidebar.header("Input Parameters")
def user_input_features():
    N = st.sidebar.slider('Nitrogen (N)', 0, 140, 90)
    P = st.sidebar.slider('Phosphorus (P)', 0, 140, 40)
    K = st.sidebar.slider('Potassium (K)', 0, 205, 40)
    temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    ph = st.sidebar.slider('pH', 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 500.0, 100.0)
    data = {'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Entered parameters")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("Recommended Crop")
st.success(f"🌱 {prediction}")

# Show top probabilities
proba_df = pd.DataFrame({
    'crop': model.classes_,
    'probability': prediction_proba
}).sort_values(by='probability', ascending=False).reset_index(drop=True)

st.subheader("Model Confidence (Top predictions)")
st.table(proba_df.head(5))

# Batch prediction via CSV upload
st.subheader("Batch Prediction - Upload CSV")
uploaded_file = st.file_uploader("Upload CSV for batch prediction (use same columns as training)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Check columns
    expected_cols = ['N','P','K','temperature','humidity','ph','rainfall']
    if all(col in data.columns for col in expected_cols):
        preds = model.predict(data[expected_cols])
        data['recommended_crop'] = preds
        st.write(data.head())
        st.download_button("Download predictions as CSV", data.to_csv(index=False), file_name="batch_predictions.csv")
    else:
        st.error(f"CSV missing required columns. Required: {expected_cols}")

# Simple plotting of nutrient values
st.subheader("Nutrient Visual")
nutrient_fig, ax = plt.subplots()
ax.bar(['N','P','K'], [input_df['N'][0], input_df['P'][0], input_df['K'][0]])
ax.set_ylabel("Amount")
ax.set_title("Soil Nutrients")
st.pyplot(nutrient_fig)

st.write("---")
st.write("Made with ❤️ — Crop Recommendation using RandomForest")
