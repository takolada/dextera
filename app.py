# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# Load trained GPR pipeline
gpr = joblib.load("gpr_all_fingers.pkl")

st.title("Soft Bionic Hand – Optimize Joint Gaps")

# -------------------
# User Inputs
# -------------------
st.sidebar.header("Constraints")
finger = st.sidebar.selectbox("Finger", [1,2,3,4])
length = st.sidebar.number_input("Finger Length", value=50.0)
diameter = st.sidebar.number_input("Finger Diameter", value=20.0)

st.sidebar.header("Target Joint Angles")
target_mcp = st.sidebar.number_input("Target MCP", value=90.0)
target_pip = st.sidebar.number_input("Target PIP", value=79.0)
target_dip = st.sidebar.number_input("Target DIP", value=96.0)

target_angles = np.array([target_mcp, target_pip, target_dip])

# -------------------
# Objective function
# -------------------
def objective(gaps):
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
# Run optimization
# -------------------
if st.button("Suggest Optimal Gaps"):
    x0 = [2.0, 2.0, 2.0]  # initial guess
    bounds = [(0.1, 10.0)]*3
    
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    
    best_gaps = res.x
    st.subheader("Optimal Joint Gaps")
    st.write({
        "Gap_halfMCP": best_gaps[0],
        "Gap_halfPIP": best_gaps[1],
        "Gap_halfDIP": best_gaps[2]
    })

    # Predicted angles for these gaps
    df_input = pd.DataFrame([{
        "Gap_halfMCP": best_gaps[0],
        "Gap_halfPIP": best_gaps[1],
        "Gap_halfDIP": best_gaps[2],
        "Finger": finger,
        "Length": length,
        "Diameter": diameter
    }])
    y_pred = gpr.predict(df_input)[0]
    
    st.subheader("Predicted Joint Angles")
    st.write({
        "MCP": y_pred[0],
        "PIP": y_pred[1],
        "DIP": y_pred[2]
    })
    
    st.subheader("Target Angles")
    st.write({
        "MCP": target_angles[0],
        "PIP": target_angles[1],
        "DIP": target_angles[2]
    })
    
    st.subheader("Deviation")
    st.write(np.round(y_pred - target_angles,2))
