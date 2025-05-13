import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("model_df.csv")
    df["mean_hmov"] = df["mean_hmov"].abs()  # Absolute value for mean_hmov
    df.dropna(inplace=True)  # Drop rows with NA values
    return df

df = load_data()

# Prepare the models
ss_model = sm.OLS(df['sweet_spot_rate'], sm.add_constant(pd.get_dummies(df[['mean_velo', 'mean_vmov', 'mean_hmov', 'mean_spin', 'mean_ext', 'AutoPitchType']], drop_first=True))).fit()
whiff_model = sm.OLS(df['whiff_rate'], sm.add_constant(pd.get_dummies(df[['mean_velo', 'mean_vmov', 'mean_hmov', 'mean_spin', 'mean_ext', 'AutoPitchType']], drop_first=True))).fit()

# Get mean and std for standardization
ss_mean = ss_model.fittedvalues.mean()
ss_std = ss_model.fittedvalues.std()
whiff_mean = whiff_model.fittedvalues.mean()
whiff_std = whiff_model.fittedvalues.std()

# Streamlit UI
st.title("D1 Pitch Grade Model")

num_pitches = 5
user_data = []

cols = st.columns(7)

for col_index, col_name in enumerate(["Velo", "IVB", "H-Mov", "Spin Rate", "Extension", "Usage", "AutoPitchType"]):
    col_data = []
    for i in range(num_pitches):
        if col_name == "Usage":
            col_data.append(cols[col_index].number_input(f"{col_name} {i+1}", value=0.0, min_value=0.0, max_value=1.0, step=0.01))
        elif col_name == "AutoPitchType":
            pitch_types = ["Curveball", "Cutter", "Four-Seam", "Sinker", "Slider", "Splitter"]
            col_data.append(cols[col_index].selectbox(f"{col_name} {i+1}", pitch_types, index=2))
        else:
            col_data.append(cols[col_index].number_input(f"{col_name} {i+1}", value=0.0))
    user_data.append(col_data)

# Convert input to DataFrame
df_input = pd.DataFrame(
    np.array(user_data).T, 
    columns=["mean_velo", "mean_vmov", "mean_hmov", "mean_spin", "mean_ext", "usage", "AutoPitchType"]
)

df_input["mean_hmov"] = df_input["mean_hmov"].abs()  # Absolute value for mean_hmov

df_input = df_input[df_input['usage'] > 0]  # Remove zero usage rows

if not df_input.empty:
    df_input_expanded = pd.get_dummies(df_input.drop(columns=["usage"]), columns=["AutoPitchType"], drop_first=True)
    df_input_expanded = sm.add_constant(df_input_expanded, has_constant='add')

    # Ensure column order matches training data
    all_columns = ss_model.model.exog_names
    df_input_expanded = df_input_expanded.reindex(columns=all_columns, fill_value=0)

    # Get whiff weight
    whiff_weight = st.number_input("Whiff Weight Percentage:", value=0.75, min_value=0.0, max_value=1.0, step=0.01)

    # Make predictions
    ss_preds = ss_model.predict(df_input_expanded)
    whiff_preds = whiff_model.predict(df_input_expanded)

    # Standardize grades
    ss_grades = np.round((ss_preds - ss_mean) / ss_std * 10 + 50).astype(int)
    whiff_grades = np.round((whiff_preds - whiff_mean) / whiff_std * 10 + 50).astype(int)
    overall_grades = np.round(whiff_weight * whiff_grades + (1 - whiff_weight) * ss_grades).astype(int)
    arsenal_grade = np.round(np.sum(overall_grades * df_input['usage'])).astype(int)

    # Display results
    st.text(f"Arsenal Grade: {arsenal_grade}")
    result_df = pd.DataFrame({
        "Whiff Grade": whiff_grades,
        "Sweet Spot Grade": ss_grades,
        "Overall": overall_grades
    })
    st.table(result_df)
else:
    st.text("Error: Ensure Usage sums to 1.")
