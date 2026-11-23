
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND THRESHOLD
# --------------------------------------------------

MODEL_PATH = "models/diabetes_linear_svm_pipeline.pkl"
THRESHOLD_PATH = "models/diabetes_threshold.pkl"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold

try:
    model, threshold = load_model()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🩺 Diabetes Risk Prediction System")

st.markdown("""
Predict whether a person's health indicators fall into the
**No Diabetes** group or the **Prediabetes/Diabetes** group
using a trained Linear Support Vector Machine model.
""")

st.info(
    "This application is intended for educational and research "
    "purposes. It does not provide a medical diagnosis."
)

st.divider()

# --------------------------------------------------
# INPUT FEATURES
# --------------------------------------------------

st.subheader("Enter Health Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Health Conditions")

    HighBP = st.selectbox(
        "High Blood Pressure",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    HighChol = st.selectbox(
        "High Cholesterol",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    CholCheck = st.selectbox(
        "Cholesterol Check in Last 5 Years",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Stroke = st.selectbox(
        "History of Stroke",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    HeartDiseaseorAttack = st.selectbox(
        "Heart Disease or Heart Attack",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    DiffWalk = st.selectbox(
        "Difficulty Walking",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    GenHlth = st.selectbox(
        "General Health",
        [1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Excellent",
            2: "Very Good",
            3: "Good",
            4: "Fair",
            5: "Poor"
        }[x]
    )

with col2:
    st.markdown("### Lifestyle Information")

    BMI = st.number_input(
        "BMI",
        min_value=12.0,
        max_value=98.0,
        value=25.0,
        step=0.1
    )

    Smoker = st.selectbox(
        "Smoked at Least 100 Cigarettes in Lifetime",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    PhysActivity = st.selectbox(
        "Physical Activity",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Fruits = st.selectbox(
        "Consume Fruits Daily",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Veggies = st.selectbox(
        "Consume Vegetables Daily",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    HvyAlcoholConsump = st.selectbox(
        "Heavy Alcohol Consumption",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    MentHlth = st.slider(
        "Mental Health Not Good (Days in Last 30 Days)",
        0, 30, 0
    )

    PhysHlth = st.slider(
        "Physical Health Not Good (Days in Last 30 Days)",
        0, 30, 0
    )

with col3:
    st.markdown("### Personal Information")

    AnyHealthcare = st.selectbox(
        "Healthcare Coverage",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    NoDocbcCost = st.selectbox(
        "Could Not Visit Doctor Due to Cost",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    Sex = st.selectbox(
        "Sex (Dataset Coding)",
        [0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )

    Age = st.selectbox(
        "Age Category",
        list(range(1, 14)),
        format_func=lambda x: f"Category {x}"
    )

    Education = st.selectbox(
        "Education Level",
        list(range(1, 7)),
        format_func=lambda x: f"Category {x}"
    )

    Income = st.selectbox(
        "Income Level",
        list(range(1, 9)),
        format_func=lambda x: f"Category {x}"
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):

    # Maintain the exact original training feature order
    input_data = pd.DataFrame([{
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": CholCheck,
        "BMI": BMI,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth,
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age,
        "Education": Education,
        "Income": Income
    }])

    # Generate decision score
    decision_score = model.decision_function(input_data)[0]

    # Apply selected threshold
    prediction = int(decision_score >= threshold)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model classified this input into the "
            "Prediabetes/Diabetes group."
        )
        st.write(
            "Consider discussing appropriate screening and "
            "follow-up with a qualified healthcare professional."
        )
    else:
        st.success(
            "The model classified this input into the "
            "No Diabetes group."
        )

    st.metric("Decision Score", f"{decision_score:.4f}")
    st.caption(f"Applied classification threshold: {threshold:.2f}")

    with st.expander("View Submitted Information"):
        st.dataframe(input_data, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Developed using Python, Scikit-learn and Streamlit | "
    "CDC Diabetes Health Indicators Dataset"
)
