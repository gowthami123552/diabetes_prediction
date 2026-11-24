
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

st.title("🩺 Diabetes Risk Prediction")
st.write(
    "Predict diabetes risk using a Machine Learning model "
    "trained on CDC Diabetes Health Indicators data."
)

st.divider()


# --------------------------------------------------
# LOAD MODEL AND THRESHOLD
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = "models/diabetes_linear_svm_pipeline.pkl"
    threshold_path = "models/diabetes_threshold.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(threshold_path):
        raise FileNotFoundError(
            f"Threshold file not found: {threshold_path}"
        )

    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)

    return model, threshold


try:
    model, threshold = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Enter Health Information")

st.info(
    "Select the appropriate values for each health indicator."
)

with st.form("diabetes_prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------

    with col1:

        st.markdown("### Basic Health Information")

        HighBP = st.selectbox(
            "High Blood Pressure",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        HighChol = st.selectbox(
            "High Cholesterol",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        CholCheck = st.selectbox(
            "Cholesterol Check in Last 5 Years",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        BMI = st.number_input(
            "Body Mass Index (BMI)",
            min_value=12,
            max_value=98,
            value=26,
            step=1
        )

        Smoker = st.selectbox(
            "Smoked at Least 100 Cigarettes",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        Stroke = st.selectbox(
            "Ever Had a Stroke",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        HeartDiseaseorAttack = st.selectbox(
            "Heart Disease or Heart Attack",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    # --------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------

    with col2:

        st.markdown("### Lifestyle Information")

        PhysActivity = st.selectbox(
            "Physical Activity",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        Fruits = st.selectbox(
            "Consume Fruits Regularly",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        Veggies = st.selectbox(
            "Consume Vegetables Regularly",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        HvyAlcoholConsump = st.selectbox(
            "Heavy Alcohol Consumption",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        AnyHealthcare = st.selectbox(
            "Have Healthcare Coverage",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        NoDocbcCost = st.selectbox(
            "Could Not Visit Doctor Due to Cost",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        DiffWalk = st.selectbox(
            "Difficulty Walking or Climbing Stairs",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    # --------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------

    with col3:

        st.markdown("### General and Personal Information")

        GenHlth = st.selectbox(
            "General Health",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Excellent",
                2: "Very Good",
                3: "Good",
                4: "Fair",
                5: "Poor"
            }[x]
        )

        MentHlth = st.number_input(
            "Days of Poor Mental Health (Last 30 Days)",
            min_value=0,
            max_value=30,
            value=0,
            step=1
        )

        PhysHlth = st.number_input(
            "Days of Poor Physical Health (Last 30 Days)",
            min_value=0,
            max_value=30,
            value=0,
            step=1
        )

        Sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        Age = st.selectbox(
            "Age Category",
            options=list(range(1, 14)),
            format_func=lambda x: {
                1: "18–24 years",
                2: "25–29 years",
                3: "30–34 years",
                4: "35–39 years",
                5: "40–44 years",
                6: "45–49 years",
                7: "50–54 years",
                8: "55–59 years",
                9: "60–64 years",
                10: "65–69 years",
                11: "70–74 years",
                12: "75–79 years",
                13: "80 years or older"
            }[x]
        )

        Education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "Never attended / Kindergarten",
                2: "Grades 1–8",
                3: "Grades 9–11",
                4: "Grade 12 / GED",
                5: "College 1–3 years",
                6: "College 4 years or more"
            }[x]
        )

        Income = st.selectbox(
            "Annual Household Income",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "Less than $10,000",
                2: "$10,000–$14,999",
                3: "$15,000–$19,999",
                4: "$20,000–$24,999",
                5: "$25,000–$34,999",
                6: "$35,000–$49,999",
                7: "$50,000–$74,999",
                8: "$75,000 or more"
            }[x]
        )

    st.divider()

    submit = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submit:

    input_data = pd.DataFrame(
        [[
            HighBP,
            HighChol,
            CholCheck,
            BMI,
            Smoker,
            Stroke,
            HeartDiseaseorAttack,
            PhysActivity,
            Fruits,
            Veggies,
            HvyAlcoholConsump,
            AnyHealthcare,
            NoDocbcCost,
            GenHlth,
            MentHlth,
            PhysHlth,
            DiffWalk,
            Sex,
            Age,
            Education,
            Income
        ]],
        columns=[
            "HighBP",
            "HighChol",
            "CholCheck",
            "BMI",
            "Smoker",
            "Stroke",
            "HeartDiseaseorAttack",
            "PhysActivity",
            "Fruits",
            "Veggies",
            "HvyAlcoholConsump",
            "AnyHealthcare",
            "NoDocbcCost",
            "GenHlth",
            "MentHlth",
            "PhysHlth",
            "DiffWalk",
            "Sex",
            "Age",
            "Education",
            "Income"
        ]
    )

    try:

        decision_score = model.decision_function(input_data)[0]

        prediction = int(decision_score >= threshold)

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:

            st.warning(
                "Prediction: Prediabetes or Diabetes"
            )

            st.write(
                "The model classified this input into the "
                "combined Prediabetes/Diabetes category."
            )

        else:

            st.success(
                "Prediction: No Diabetes"
            )

            st.write(
                "The model classified this input into the "
                "No Diabetes category."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Decision Score",
                f"{decision_score:.4f}"
            )

        with col2:
            st.metric(
                "Classification Threshold",
                f"{threshold:.2f}"
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "Disclaimer: This application is developed for educational "
    "and research purposes only. It is not a medical diagnostic "
    "tool. Consult a qualified healthcare professional for "
    "medical advice."
)
