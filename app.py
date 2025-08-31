import streamlit as st
import numpy as np
import joblib
from scipy.optimize import minimize

# Load pre-trained GPR models
models = joblib.load("gpr_models.pkl")

st.title("Soft Bionic Hand - Joint Gap Optimization with GPR + BO")

# Desired angles (configurable, default = [90, 79, 96])
desired_angles = st.text_input("Enter desired [MCP, PIP, DIP] angles", "90,79,96")
desired_angles = np.array([float(x) for x in desired_angles.split(",")])

# User inputs
finger = st.selectbox("Finger", [1, 2, 3, 4])
length = st.number_input("Length", min_value=0.0, value=40.0)
diameter = st.number_input("Diameter", min_value=0.0, value=15.0)

# Optimization function
def objective(gaps):
    features = np.array([[finger, length, diameter, *gaps]])
    preds = []
    stds = []
    for i, target in enumerate(['MCP', 'PIP', 'DIP']):
        mean, std = models[target].predict(features, return_std=True)
        preds.append(mean[0])
        stds.append(std[0])
    preds = np.array(preds)
    return np.sum((preds - desired_angles)**2)  # squared error

# Bayesian Optimization step
if st.button("Suggest Optimal Gaps"):
    x0 = [1.0, 1.0, 1.0]  # initial guess
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")

    if res.success:
        best_gaps = res.x
        st.success(f"Suggested Gaps: MCP={best_gaps[0]:.3f}, PIP={best_gaps[1]:.3f}, DIP={best_gaps[2]:.3f}")

        # Show predicted angles and uncertainties
        features = np.array([[finger, length, diameter, *best_gaps]])
        preds, stds = [], []
        for i, target in enumerate(['MCP', 'PIP', 'DIP']):
            mean, std = models[target].predict(features, return_std=True)
            preds.append(mean[0])
            stds.append(std[0])

        st.write("### Predicted Joint Angles with Uncertainty")
        for i, joint in enumerate(['MCP', 'PIP', 'DIP']):
            st.write(f"{joint}: {preds[i]:.2f} ± {stds[i]:.2f}")
    else:
        st.error("Optimization failed. Try different input values.")