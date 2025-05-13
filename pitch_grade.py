import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import io
import requests

# GitHub raw URL to your model_df.csv file
csv_url = "https://raw.githubusercontent.com/dgerth5/pbr_pitch_grade/refs/heads/main/model_df.csv"

# Function to load the dataframe
@st.cache_data
def load_data(url):
    """
    Loads the dataframe from the given URL.  Cached for performance.
    """
    try:
        # Use requests to get the content, which handles potential issues with direct URL access
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        csv_content = response.text
        model_df = pd.read_csv(io.StringIO(csv_content))
        return model_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None  # Important: Return None on error to prevent further issues

# Load the data
model_df = load_data(csv_url)
if model_df is None:
    st.stop() # Stop if the data couldn't be loaded.

# Train the models (outside the main function for efficiency)
try:
    # Ensure the necessary columns exist in the DataFrame
    required_columns = ['sweet_spot_rate', 'mean_velo', 'mean_vmov', 'mean_hmov', 'mean_spin', 'mean_ext', 'AutoPitchType', 'whiff_rate']
    for col in required_columns:
        if col not in model_df.columns:
            st.error(f"Error: Required column '{col}' not found in the data.")
            st.stop()

    mod_ss = sm.ols(
        "sweet_spot_rate ~ mean_velo * mean_vmov * abs(mean_hmov) * mean_spin * mean_ext + AutoPitchType",
        data=model_df,
    ).fit()
    mod_whiff = sm.ols(
        "whiff_rate ~ mean_velo * mean_vmov * abs(mean_hmov) * mean_spin * mean_ext + AutoPitchType",
        data=model_df,
    ).fit()
except Exception as e:
    st.error(f"Error training models: {e}")
    st.stop()  # Stop if models fail to train


def predict_rates(input_data):
    """
    Predicts 'Whiff Rate' and 'Sweet Spot Rate' based on input data.

    Args:
        input_data (pd.DataFrame): DataFrame with input features.

    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    try:
        predictions = pd.DataFrame()
        predictions["Pitch"] = input_data["AutoPitchType"]  # Keep original pitch types
        predictions["Whiff Rate"] = mod_whiff.predict(input_data)
        predictions["Sweet Spot Rate"] = mod_ss.predict(input_data)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return pd.DataFrame() # Return empty dataframe in case of error.

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Pitch Rate Predictor")

    # Dropdown options for AutoPitchType
    pitch_types = ["Curveball", "Cutter", "Four-Seam", "Sinker", "Slider", "Splitter"]

    # Input columns for up to 6 pitches
    input_data = []
    for i in range(6):
        st.subheader(f"Pitch {i+1}")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            pitch_type = st.selectbox("AutoPitchType", pitch_types, key=f"pitch_type_{i}")
        with col2:
            velocity = st.number_input("Velocity", value=90.0, key=f"velocity_{i}")
        with col3:
            vmov = st.number_input("V Mov", value=10.0, key=f"vmov_{i}")
        with col4:
            hmov = st.number_input("H Mov", value=5.0, key=f"hmov_{i}")
        with col5:
            spin_rate = st.number_input("Spin Rate", value=2000.0, key=f"spin_rate_{i}")
        extension = st.number_input("Extension", value=6.0, key=f"extension_{i}")

        input_data.append({
            "AutoPitchType": pitch_type,
            "mean_velo": velocity,
            "mean_vmov": vmov,
            "mean_hmov": hmov,
            "mean_spin": spin_rate,
            "mean_ext": extension,
        })

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Predict rates and display results
    if st.button("Predict Rates"):
        predictions_df = predict_rates(input_df)
        if not predictions_df.empty: # Check if the predictions were successful
            st.subheader("Predicted Rates")
            st.dataframe(predictions_df,  use_container_width=True)
        else:
            st.error("Failed to generate predictions. Please check the input data and model.")

if __name__ == "__main__":
    main()
