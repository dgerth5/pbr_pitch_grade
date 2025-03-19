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
cols = st.columns(5)
user_data = []

for i in range(5):  # Iterate over pitch entries
    user_data.append({
        "mean_velo": cols[0].number_input(f"Velo {i+1}", value=90.0),
        "mean_vmov": cols[1].number_input(f"IVB {i+1}", value=12.0),
        "mean_hmov": cols[2].number_input(f"H-Mov {i+1}", value=10.0),
        "mean_spin": cols[3].number_input(f"Spin Rate {i+1}", value=2100.0),
        "usage": cols[4].number_input(f"Usage {i+1}", value=0.2, min_value=0.0, max_value=1.0, step=0.01)
    })

df_input = pd.DataFrame(user_data)
df_input = df_input[df_input['usage'] > 0]  # Remove rows where usage is 0

whiff_weight = st.number_input("Whiff Weight Percentage:", value=0.75, min_value=0.0, max_value=1.0, step=0.01)

if not df_input.empty and np.isclose(df_input['usage'].sum(), 1.0):
    # Ensure X_input has the same structure as X
    X_input = sm.add_constant(df_input[['mean_velo', 'mean_vmov', 'mean_hmov', 'mean_spin']], has_constant='add')
    X_input = X_input[X.columns]  # Ensure same column order as training data

    pred_vals = mod1.predict(X_input)

    whiff_std = np.round((pred_vals.iloc[:, 1] - mean_wr) / sd_wr * 10 + 50).astype(int)
    ss_std = np.round((pred_vals.iloc[:, 0] - mean_ss) / sd_ss * 10 + 50).astype(int)

    overall_grade = np.round(whiff_weight * whiff_std + (1 - whiff_weight) * ss_std).astype(int)
    arsenal_grade = np.round(np.sum(overall_grade * df_input['usage'])).astype(int)

    st.text(f"Arsenal Grade: {arsenal_grade}")

    result_df = pd.DataFrame({
        "Whiff Grade": whiff_std,
        "Sweet Spot Grade": ss_std,
        "Overall": overall_grade
    })
    st.table(result_df)
else:
    st.text("Error: Ensure Usage sums to 1.")
