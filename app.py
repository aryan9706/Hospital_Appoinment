import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# Load the trained model from the pickle file
model = pickle.load(open("best_decision_tree_model.pkl", "rb"))

# Configure the Streamlit page layout and title
st.set_page_config(page_title="Hospital Appointment No-Show Predictor", layout="centered")

# Page header
st.title("Hospital Appointment No-Show Predictor")
st.write("Use the form below to enter patient appointment details and predict whether the patient will show up.")

# User input form for appointment features
with st.form(key="appointment_form"):
    gender = st.radio("Gender", options=["F", "M"], horizontal=True, index=0)
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    scholarship = st.checkbox("Scholarship (Bolsa Família)", value=False)
    hipertension = st.checkbox("Hypertension", value=False)
    diabetes = st.checkbox("Diabetes", value=False)
    alcoholism = st.toggle("Alcoholism", value=False)
    handicap = st.selectbox("Handicap level", options=[0, 1, 2, 3, 4], index=0)
    sms_received = st.checkbox("SMS Received", value=False)
    waiting_days = st.slider("Waiting days between scheduling and appointment", min_value=0, max_value=30, value=0, help="Number of days between the scheduled date and appointment date.")

    # Submit button for making prediction
    submit_button = st.form_submit_button(label="Predict")

# Process input and make prediction only after user submits the form
if submit_button:
    # waiting_days is provided directly from the slider input

        # Convert categorical and boolean fields into numeric values for the model
        gender_code = 1 if gender == "M" else 0
        feature_vector = np.array([
            gender_code,
            age,
            int(scholarship),
            int(hipertension),
            int(diabetes),
            int(alcoholism),
            int(handicap),
            int(sms_received),
            waiting_days,
        ]).reshape(1, -1)

        # Predict the outcome using the loaded model
        prediction = model.predict(feature_vector)[0]
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(feature_vector)[0]

        # Convert model output to a human-readable label
        result = "No-show" if prediction == 1 or str(prediction).lower() in ["yes", "true"] else "Show-up"
        st.subheader("Prediction")
        st.write(f"**Result:** {result}")

        # Display probability values if the model supports predict_proba
        if prediction_proba is not None:
            if len(prediction_proba) == 2:
                st.write(f"**No-show probability:** {prediction_proba[1]:.2f}")
                st.write(f"**Show-up probability:** {prediction_proba[0]:.2f}")
            else:
                st.write(f"**Probabilities:** {prediction_proba}")

        # Show the input values used for prediction
        st.write("---")
        st.write("### Input values")
        st.write({
            "Gender": gender,
            "Age": age,
            "Scholarship": int(scholarship),
            "Hypertension": int(hipertension),
            "Diabetes": int(diabetes),
            "Alcoholism": int(alcoholism),
            "Handicap": int(handicap),
            "SMS Received": int(sms_received),
            "Waiting Days": waiting_days,
        })

# Inform user about model preprocessing if necessary
st.write("\n")
st.info("If your model was trained with a different feature order or preprocessing, adjust the input transformation accordingly.")
