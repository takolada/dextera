import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# Load trained GPR pipeline
gpr = joblib.load("gpr_all_fingers.pkl")

st.title("Soft Bionic Hand – Optimize Gaps for All Fingers with Individual Constraints")

# -------------------
# User Inputs per finger
# -------------------
st.subheader("Finger Constraints")
lengths = []
diameters = []
for finger in [1,2,3,4]:
    st.markdown(f"**Finger {finger}**")
    l = st.number_input(f"Length Finger {finger}", value=50.0)
    d = st.number_input(f"Diameter Finger {finger}", value=20.0)
    lengths.append(l)
    diameters.append(d)

st.subheader("Target Joint Angles (Same for all fingers)")
target_mcp = st.number_input("Target MCP", value=90.0)
target_pip = st.number_input("Target PIP", value=79.0)
target_dip = st.number_input("Target DIP", value=96.0)
target_angles = np.array([target_mcp, target_pip, target_dip])

# -------------------
# Objective function (per finger)
# -------------------
def objective(gaps, finger_idx):
    df_input = pd.DataFrame([{
        "Gap_halfMCP": gaps[0],
        "Gap_halfPIP": gaps[1],
        "Gap_halfDIP": gaps[2],
        "Finger": finger_idx,
        "Length": lengths[finger_idx-1],
        "Diameter": diameters[finger_idx-1]
    }])
    y_pred = gpr.predict(df_input)[0]
    return np.sum((y_pred - target_angles)**2)

# -------------------
# Run optimization
# -------------------
if st.button("Suggest Optimal Gaps for All Fingers"):
    results = []
    for finger_idx in [1,2,3,4]:
        x0 = [2.0, 2.0, 2.0]
        bounds = [(0.1, 10.0)]*3
        res = minimize(objective, x0=x0, bounds=bounds, args=(finger_idx,), method="L-BFGS-B")
        best_gaps = res.x
        
        df_input = pd.DataFrame([{
            "Gap_halfMCP": best_gaps[0],
            "Gap_halfPIP": best_gaps[1],
            "Gap_halfDIP": best_gaps[2],
            "Finger": finger_idx,
            "Length": lengths[finger_idx-1],
            "Diameter": diameters[finger_idx-1]
        }])
        y_pred = gpr.predict(df_input)[0]
        deviation = np.round(y_pred - target_angles,2)
        
        results.append({
            "Finger": finger_idx,
            "Length": lengths[finger_idx-1],
            "Diameter": diameters[finger_idx-1],
            "Gap_halfMCP": best_gaps[0],
            "Gap_halfPIP": best_gaps[1],
            "Gap_halfDIP": best_gaps[2],
            "Pred_MCP": y_pred[0],
            "Pred_PIP": y_pred[1],
            "Pred_DIP": y_pred[2],
            "Dev_MCP": deviation[0],
            "Dev_PIP": deviation[1],
            "Dev_DIP": deviation[2]
        })
    
    results_df = pd.DataFrame(results)
    st.subheader("Optimal Gaps and Predicted Angles per Finger")
    st.dataframe(results_df)
