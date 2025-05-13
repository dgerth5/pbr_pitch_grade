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

# Load the average and standard deviations for each pitch type.
pitch_stats = {
    "Changeup": {"mean_ss": 0.320, "mean_whiff": 0.343, "sd_ss": 0.0204, "sd_whiff": 0.0251},
    "Curveball": {"mean_ss": 0.327, "mean_whiff": 0.399, "sd_ss": 0.0227, "sd_whiff": 0.0499},
    "Cutter": {"mean_ss": 0.326, "mean_whiff": 0.324, "sd_ss": 0.0192, "sd_whiff": 0.0362},
    "Four-Seam": {"mean_ss": 0.350, "mean_whiff": 0.298, "sd_ss": 0.0149, "sd_whiff": 0.0398},
    "Sinker": {"mean_ss": 0.314, "mean_whiff": 0.238, "sd_ss": 0.0194, "sd_whiff": 0.0183},
    "Slider": {"mean_ss": 0.343, "mean_whiff": 0.436, "sd_ss": 0.0135, "sd_whiff": 0.0372},
    "Splitter": {"mean_ss": 0.246, "mean_whiff": 0.475, "sd_ss": 0.0306, "sd_whiff": 0.0767},
}


def calculate_grade(value, mean, sd):
    """Calculates the scouting grade (20-80 scale)."""
    return 10 * (value - mean) / sd + 50



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

        # Calculate scouting grades
        predictions["Whiff Grade"] = predictions.apply(
            lambda row: calculate_grade(
                row["Whiff Rate"],
                pitch_stats[row["Pitch"]]["mean_whiff"],
                pitch_stats[row["Pitch"]]["sd_whiff"],
            ),
            axis=1,
        )
        predictions["Sweet Spot Grade"] = predictions.apply(
            lambda row: calculate_grade(
                row["Sweet Spot Rate"],
                pitch_stats[row["Pitch"]]["mean_ss"],
                pitch_stats[row["Pitch"]]["sd_ss"],
            ),
            axis=1,
        )
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
            velocity = st.number_input("Velocity", value=0.0, key=f"velocity_{i}") # Changed default to 0
        with col3:
            vmov = st.number_input("V Mov", value=0.0, key=f"vmov_{i}")  # Changed default to 0
        with col4:
            hmov = st.number_input("H Mov", value=0.0, key=f"hmov_{i}")  # Changed default to 0
        with col5:
            spin_rate = st.number_input("Spin Rate", value=0.0, key=f"spin_rate_{i}")  # Changed default to 0
        extension = st.number_input("Extension", value=0.0, key=f"extension_{i}")  # Changed default to 0

        # Only include the pitch if at least one of the input values is not zero.
        if velocity != 0 or vmov != 0 or hmov != 0 or spin_rate != 0 or extension != 0:
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
            # Rearrange columns for desired output
            predictions_df = predictions_df[["Pitch", "Whiff Rate", "Sweet Spot Rate", "Whiff Grade", "Sweet Spot Grade"]]
            st.dataframe(predictions_df,  use_container_width=True)
        else:
            st.error("Failed to generate predictions. Please check the input data and model.")



if __name__ == "__main__":
    main()
