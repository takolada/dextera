# streamlit_app_all_fingers.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize

# -------------------
# Page config (must be first Streamlit call)
# -------------------
st.set_page_config(
    page_title="DEXTERA AI",
    page_icon="ICON.png",   # custom favicon
    layout="centered"
)

# -------------------
# Header with logo (centered & bigger)
# -------------------
from PIL import Image

# Load logo with PIL to avoid path issues
logo = Image.open("logo.png")

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image(logo, width=280)  # centered logo
st.markdown(
    "<p style='font-size:18px; color:gray; text-align:center;'>"
    "An intelligent platform to build your own soft robotic hand</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------
# Load trained GPR model
# -------------------
@st.cache_resource
def load_model():
    return joblib.load("gpr_all_fingers.pkl")

gpr = load_model()

# -------------------
# Input: Constraints per finger
# -------------------
st.subheader("🖐️ Fill Your Hand Dimensions")

# Add dimension illustrations
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.image("length.png", caption="Finger Length", use_container_width=True)
with col_img2:
    st.image("diameter.png", caption="Finger Diameter", use_container_width=True)

lengths = []
diameters = []
fingers = ['Index', 'Middle', 'Ring', 'Little']

for finger in [1,2,3,4]:
    with st.expander(f"{fingers[finger-1]} Finger", expanded=(finger==1)):
        l = st.number_input(f"Length (mm)", value=50.0, step=0.1, key=f"len_{finger}")
        d = st.number_input(f"Diameter (mm)", value=20.0, step=0.1, key=f"dia_{finger}")
        lengths.append(l)
        diameters.append(d)

# -------------------
# Input: Target angles (slider + number input, fully linked)
# -------------------
st.subheader("🎯 Desired Range of Motion")

def linked_slider_number(label, min_val, max_val, default_val, step, key):
    # initialize in session_state if not exists
    if key not in st.session_state:
        st.session_state[key] = default_val

    col1, col2 = st.columns([3,1])

    # slider uses session_state[key]
    with col1:
        st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=f"{key}_slider",
            value=st.session_state[key],
            on_change=lambda: st.session_state.update({key: st.session_state[f"{key}_slider"]})
        )

    # number input uses session_state[key]
    with col2:
        st.number_input(
            "",
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=f"{key}_num",
            value=st.session_state[key],
            on_change=lambda: st.session_state.update({key: st.session_state[f"{key}_num"]})
        )

    return st.session_state[key]

# Use linked inputs for each target
target_mcp = linked_slider_number("Target MCP (°)", 0.0, 120.0, 90.0, 0.5, "mcp")
target_pip = linked_slider_number("Target PIP (°)", 0.0, 120.0, 79.0, 0.5, "pip")
target_dip = linked_slider_number("Target DIP (°)", 0.0, 120.0, 96.0, 0.5, "dip")

target_angles = np.array([target_mcp, target_pip, target_dip])


# -------------------
# Objective function
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
    y_pred = gpr.predict(df_input)[0]
    return np.sum((y_pred - target_angles)**2)

# -------------------
# Optimization for all fingers
# -------------------
if st.button("🚀 Create the design!", use_container_width=True):
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
        y_pred = gpr.predict(df_input)[0]
        deviation = np.round(y_pred - target_angles,2)
        
        results.append({
            "Finger": fingers[finger_idx-1],
            "Length (mm)": lengths[finger_idx-1],
            "Diameter (mm)": diameters[finger_idx-1],
            "Gap MCP (mm)": round(best_gaps[0],2),
            "Gap PIP (mm)": round(best_gaps[1]*2,2),
            "Gap DIP (mm)": round(best_gaps[2]*2,2),
            "Predicted MCP (°)": round(y_pred[0],2),
            "Predicted PIP (°)": round(y_pred[1],2),
            "Predicted DIP (°)": round(y_pred[2],2),
            "Error Probability MCP (°)": deviation[0],
            "Error Probability PIP (°)": deviation[1],
            "Error Probability DIP (°)": deviation[2]
        })
    
    results_df = pd.DataFrame(results)
    st.subheader("📊 Optimal Gaps and Predicted Angles per Finger")
    st.dataframe(results_df, use_container_width=True)

    # Download button
    st.download_button(
        "💾 Download Results as CSV",
        results_df.to_csv(index=False).encode("utf-8"),
        "design_results.csv",
        "text/csv",
        use_container_width=True
    )


