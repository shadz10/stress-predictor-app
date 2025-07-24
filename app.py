import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model and scaler
model = joblib.load("vertical_stress_model.pkl")
scaler = joblib.load("vertical_scaler.pkl")

# Define features and labels
feature_labels = {
    'volume': 'Volume (mm³)',
    'surface_area': 'Surface Area (mm²)',
    'mass': 'Mass (kg)',
    'num_faces': 'Number of Faces',
    'num_vertices': 'Number of Vertices',
    'average_edge_length': 'Average Edge Length (mm)',
    'genus': 'Genus (number of holes)',
    'num_tets': 'Number of Tetrahedral Elements',
    'max_ver_magdisp': 'Max Vertical Displacement (mm)',
}
features = list(feature_labels.keys())

# Unit selector
unit = st.selectbox("Select output stress unit:", ["Pa", "kPa", "MPa"])

# Title and instructions
st.title("🔩 Vertical Stress Predictor")
st.write("Enter your CAD part's parameters below:")

# CAD upload section
st.markdown("### 🗂️ Optional: Upload Your CAD File - Coming Soon")
uploaded_file = st.file_uploader("Upload a STEP/IGES/STL file", type=["step", "stp", "iges", "igs", "stl"])

use_cad_values = False
if uploaded_file:
    st.success("✅ File uploaded successfully.")
    use_cad_values = st.checkbox("Use extracted CAD values (coming soon)", value=False, disabled=True)
    st.caption("🔧 CAD feature extraction is not yet implemented. You must manually enter values below.")
else:
    st.caption("📁 You can skip this and manually enter values below.")

# Disclaimer
with st.expander("ℹ️ Disclaimer"):
    st.markdown("""
    This tool is trained on a pre-existing dataset and is meant **for educational or exploratory purposes only**.  
    - It does **not** reflect real-world FEA or physical stress test results.  
    - Predictions are only as good as the dataset it's trained on, which contains **simplified** geometry and loading conditions.  
    - If you're designing for structural integrity or safety, **create your own CAD model and run proper FEA simulations.**
    """)

# Input form layout
input_data = {}
valid_inputs = True
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    col = col1 if i < len(features) / 2 else col2
    label = feature_labels[feature]

    if feature == 'genus':
        val = col.number_input(label, min_value=0, step=1, value=0)
    elif feature in ['num_faces', 'num_vertices', 'num_tets']:
        val = col.number_input(label, min_value=1, step=1, value=1)
    else:
        val = col.number_input(label, min_value=0.0001, step=0.1, value=1.0)

    input_data[feature] = val

    if feature != 'genus' and val <= 0:
        valid_inputs = False

# Prediction
if st.button("📈 Predict"):
    if not valid_inputs:
        st.error("❌ All fields except 'Genus' must be greater than 0.")
    else:
        X_input = pd.DataFrame([input_data])[features]
        X_scaled = scaler.transform(X_input)
        y_log = model.predict(X_scaled)
        stress_pa = np.expm1(y_log[0])

        # Convert unit
        if unit == "kPa":
            stress = stress_pa / 1e3
        elif unit == "MPa":
            stress = stress_pa / 1e6
        else:
            stress = stress_pa

        # Store values in session state
        st.session_state["stress_pa"] = stress_pa
        st.session_state["unit"] = unit

        st.success(f"Predicted Vertical Stress: {stress:.4f} {unit}")

        # Download CSV
        result_df = pd.DataFrame([input_data])
        result_df["predicted_stress_" + unit.lower()] = stress
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result CSV", csv, file_name="stress_prediction.csv", mime="text/csv")

# Graph
if st.button("📊 Show Model Performance"):
    if "stress_pa" in st.session_state:
        stress_pa = st.session_state["stress_pa"]
        unit = st.session_state["unit"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter([stress_pa], [stress_pa], label="Your Prediction", color='orange')
        ax.plot([0, stress_pa], [0, stress_pa], 'r--', label="Ideal Fit")
        ax.set_xlabel(f"True Stress ({unit})")
        ax.set_ylabel(f"Predicted Stress ({unit})")
        ax.set_title("Prediction Alignment (example point)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning("Please predict stress first.")

# Reset
if st.button("🔄 Reset"):
    st.rerun()
