# streamlit_app_all_fingers.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# Load trained GPR pipeline
gpr = joblib.load("gpr_all_fingers.pkl")

st.title("Soft Bionic Hand – Optimize Gaps for All Fingers")

# -------------------
# User Inputs
# -------------------
length = st.number_input("Finger Length", value=50.0)
diameter = st.number_input("Finger Diameter", value=20.0)

st.subheader("Target Joint Angles")
target_mcp = st.number_input("Target MCP", value=90.0)
target_pip = st.number_input("Target PIP", value=79.0)
target_dip = st.number_input("Target DIP", value=96.0)

target_angles = np.array([target_mcp, target_pip, target_dip])

# -------------------
# Objective function (per finger)
# -------------------
def objective(gaps, finger):
    df_input = pd.DataFrame([{
        "Gap_halfMCP": gaps[0],
        "Gap_halfPIP": gaps[1],
        "Gap_halfDIP": gaps[2],
        "Finger": finger,
        "Length": length,
        "Diameter": diameter
    }])
    y_pred = gpr.predict(df_input)[0]
    return np.sum((y_pred - target_angles)**2)

# -------------------
# Run optimization for all fingers
# -------------------
if st.button("Suggest Optimal Gaps for All Fingers"):
    results = []
    
    for finger in [1,2,3,4]:
        x0 = [2.0, 2.0, 2.0]
        bounds = [(0.1, 10.0)]*3
        res = minimize(objective, x0=x0, bounds=bounds, args=(finger,), method="L-BFGS-B")
        
        best_gaps = res.x
        df_input = pd.DataFrame([{
            "Gap_halfMCP": best_gaps[0],
            "Gap_halfPIP": best_gaps[1],
            "Gap_halfDIP": best_gaps[2],
            "Finger": finger,
            "Length": length,
            "Diameter": diameter
        }])
        y_pred = gpr.predict(df_input)[0]
        deviation = np.round(y_pred - target_angles,2)
        
        results.append({
            "Finger": finger,
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
    st.subheader("Optimal Gaps and Predicted Angles for All Fingers")
    st.dataframe(results_df)
