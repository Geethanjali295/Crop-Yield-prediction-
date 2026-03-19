import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb
# --- 1. Load the trained model and preprocessors ---

# Load the XGBoost model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the StandardScaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load feature names
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Define the original categorical columns for one-hot encoding
original_categorical_cols = ['Soil_Type', 'Crop', 'Weather_Condition']

# --- 2. Preprocessing Function ---
def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    # Apply StandardScaler for Rainfall_mm and Temperature_Celsius
    input_df[['Rainfall_mm', 'Temperature_Celsius']] = scaler.transform(input_df[['Rainfall_mm', 'Temperature_Celsius']])

    # Apply LabelEncoder for Fertilizer_Used and Irrigation_Used
    input_df['Fertilizer_Used'] = le.transform(input_df['Fertilizer_Used'])
    input_df['Irrigation_Used'] = le.transform(input_df['Irrigation_Used'])

    # Apply OneHotEncoder for categorical features
    input_df_processed = pd.get_dummies(input_df, columns=original_categorical_cols)

    # Align columns with the training data features (crucial step)
    # Create a DataFrame with all expected features, filled with zeros
    final_input = pd.DataFrame(0, index=[0], columns=feature_names)

    # Copy values from the processed input to the final_input
    for col in input_df_processed.columns:
        if col in final_input.columns:
            final_input[col] = input_df_processed[col]

    # Ensure boolean columns from get_dummies are int
    bool_cols_final = final_input.select_dtypes(include='bool').columns
    final_input[bool_cols_final] = final_input[bool_cols_final].astype(int)

    return final_input

# --- 3. Streamlit App Layout ---
st.set_page_config(page_title="Crop Yield Prediction App", layout="centered")
st.title("🌾 Crop Yield Prediction App")
st.write("Enter the crop growing conditions to predict the yield (tons per hectare).")

# Input fields
st.sidebar.header("Input Crop Conditions")

soil_type = st.sidebar.selectbox("Soil Type", ['Sandy', 'Clay', 'Loam', 'Peaty', 'Chalky', 'Silt'])
crop_type = st.sidebar.selectbox("Crop Type", ['Cotton', 'Rice', 'Barley', 'Soybean', 'Wheat', 'Maize'])
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=500.0, step=10.0)
temperature = st.sidebar.number_input("Temperature (Celsius)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
fertilizer_used = st.sidebar.checkbox("Fertilizer Used?")
irrigation_used = st.sidebar.checkbox("Irrigation Used?")
weather_condition = st.sidebar.selectbox("Weather Condition", ['Cloudy', 'Rainy', 'Sunny'])
days_to_harvest = st.sidebar.number_input("Days to Harvest", min_value=30, max_value=180, value=100, step=1)

# --- 4. Make Prediction ---
if st.sidebar.button("Predict Yield"):
    input_data = pd.DataFrame({
        'Soil_Type': [soil_type],
        'Crop': [crop_type],
        'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temperature],
        'Fertilizer_Used': [fertilizer_used],
        'Irrigation_Used': [irrigation_used],
        'Weather_Condition': [weather_condition],
        'Days_to_Harvest': [days_to_harvest]
    })

    processed_input = preprocess_input(input_data)

    try:
        prediction = model.predict(processed_input)
        st.success(f"Predicted Crop Yield: **{prediction[0]:.2f} tons per hectare**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("This app uses an XGBoost Regressor model to predict crop yield based on various environmental and agricultural factors.")
