import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("model_df.csv")

df = load_data()

# Prepare the model
y = df[['sweet_spot_rate', 'whiff_rate']]
X = df[['mean_velo', 'mean_vmov', 'mean_hmov', 'mean_spin']]
X = sm.add_constant(X)

mod1 = sm.OLS(y, X).fit()

# Standardization parameters
preds = mod1.predict(X)
mean_ss = preds.iloc[:, 0].mean()
sd_ss = preds.iloc[:, 0].std()
mean_wr = preds.iloc[:, 1].mean()
sd_wr = preds.iloc[:, 1].std()

# Streamlit UI
st.title("D1 Pitch Grade Model")

# Input fields (column-wise tab navigation)
num_pitches = 5
user_data = []

# Define column layout
cols = st.columns(5)

# Column-wise input loop (fixes tab order)
for col_index, col_name in enumerate(["Velo", "IVB", "H-Mov", "Spin Rate", "Usage"]):
    col_data = []
    for i in range(num_pitches):
        value = 0.0  # Default value set to 0
        min_value = 0.0 if col_name != "Usage" else 0.0
        max_value = 1.0 if col_name == "Usage" else None
        step = 0.01 if col_name == "Usage" else None
        col_data.append(cols[col_index].number_input(f"{col_name} {i+1}", value=value, min_value=min_value, max_value=max_value, step=step))
    
    user_data.append(col_data)

# Convert input into DataFrame
df_input = pd.DataFrame(
    np.array(user_data).T, 
    columns=["mean_velo", "mean_vmov", "mean_hmov", "mean_spin", "usage"]
)

df_input = df_input[df_input['usage'] > 0]  # Remove rows where usage is 0

# Whiff weight input
whiff_weight = st.number_input("Whiff Weight Percentage:", value=0.75, min_value=0.0, max_value=1.0, step=0.01)

# Ensure valid inputs before making predictions
if not df_input.empty and np.isclose(df_input['usage'].sum(), 1.0):
    # Ensure X_input has the same structure as X
    X_input = sm.add_constant(df_input[['mean_velo', 'mean_vmov', 'mean_hmov', 'mean_spin']], has_constant='add')
    X_input = X_input[X.columns]  # Ensure same column order as training data

    # Predictions
    pred_vals = mod1.predict(X_input)

    # Standardized grades
    whiff_std = np.round((pred_vals.iloc[:, 1] - mean_wr) / sd_wr * 10 + 50).astype(int)
    ss_std = np.round((pred_vals.iloc[:, 0] - mean_ss) / sd_ss * 10 + 50).astype(int)

    # Overall grades
    overall_grade = np.round(whiff_weight * whiff_std + (1 - whiff_weight) * ss_std).astype(int)
    arsenal_grade = np.round(np.sum(overall_grade * df_input['usage'])).astype(int)

    # Display results
    st.text(f"Arsenal Grade: {arsenal_grade}")

    result_df = pd.DataFrame({
        "Whiff Grade": whiff_std,
        "Sweet Spot Grade": ss_std,
        "Overall": overall_grade
    })
    st.table(result_df)
else:
    st.text("Error: Ensure Usage sums to 1.")
