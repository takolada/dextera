# streamlit_app_all_fingers.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# Load trained GPR
gpr = joblib.load("gpr_all_fingers.pkl")

st.title("Dextera AI: an intelligent platform to build your own soft robotic hand")

# -------------------
# Input: Constraints per finger
# -------------------
st.subheader("Please fill your finger dimensions here")
lengths = []
diameters = []
fingers = ['Index', 'Middle', 'Ring', 'Little']
for finger in [1,2,3,4]:
    st.markdown(f"**{fingers[finger]} Finger**")
    l = st.number_input(f"Length Finger {finger}", value=50.0, step=0.1)
    d = st.number_input(f"Diameter Finger {finger}", value=20.0, step=0.1)
    lengths.append(l)
    diameters.append(d)

# -------------------
# Input: Target angles
# -------------------
st.subheader("Set your desired finger range of motion or leave it as it is")
target_mcp = st.number_input("Target MCP", value=90.0, step=0.1)
target_pip = st.number_input("Target PIP", value=79.0, step=0.1)
target_dip = st.number_input("Target DIP", value=96.0, step=0.1)
target_angles = np.array([target_mcp, target_pip, target_dip])

# -------------------
# Objective function (per finger)
# -------------------
def objective(gaps, finger_idx):
    df_input = pd.DataFrame([{
        "Gap_halfMCP": gaps[0],
        "Gap_halfPIP": gaps[1],
        "Gap_halfDIP": gaps[2],
        "Finger": int(finger_idx),
        "Length": lengths[finger_idx-1],
        "Diameter": diameters[finger_idx-1]
    }])
    # Ensure Finger is int type
    df_input['Finger'] = df_input['Finger'].astype(int)
    y_pred = gpr.predict(df_input)[0]
    return np.sum((y_pred - target_angles)**2)

# -------------------
# Optimization for all fingers
# -------------------
if st.button("Create the design!"):
    results = []
    for finger_idx in [1,2,3,4]:
        x0 = [2.0, 2.0, 2.0]  # initial guess
        bounds = [(0.1,10.0)]*3
        res = minimize(objective, x0=x0, bounds=bounds, args=(finger_idx,), method="L-BFGS-B")
        best_gaps = res.x
        
        df_input = pd.DataFrame([{
            "Gap_halfMCP": best_gaps[0],
            "Gap_halfPIP": best_gaps[1],
            "Gap_halfDIP": best_gaps[2],
            "Finger": int(finger_idx),
            "Length": lengths[finger_idx-1],
            "Diameter": diameters[finger_idx-1]
        }])
        df_input['Finger'] = df_input['Finger'].astype(int)
        y_pred = gpr.predict(df_input)[0]
        deviation = np.round(y_pred - target_angles,2)
        
        results.append({
            "Finger": finger_idx,
            "Length": lengths[finger_idx-1],
            "Diameter": diameters[finger_idx-1],
            "Gap MCP": best_gaps[0],
            "Gap PIP": best_gaps[1]*2,
            "Gap DIP": best_gaps[2]*2,
            "Predicted MCP": y_pred[0],
            "Predicted PIP": y_pred[1],
            "Predicted DIP": y_pred[2],
            "MCP Uncertainty": deviation[0],
            "PIP Uncertainty": deviation[1],
            "DIP Uncertainty": deviation[2]
        })
    
    results_df = pd.DataFrame(results)
    st.subheader("Optimal Gaps and Predicted Angles per Finger")
    st.dataframe(results_df)

