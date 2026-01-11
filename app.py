import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Income Classification App",
    page_icon="💼",
    layout="wide"
)

# Load model and preprocessor
@st.cache_resource
def load_artifacts():
    model = joblib.load("income_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

# App title
st.title("💼 Income Classification Prediction App")
st.markdown("""
This application predicts whether an individual's income is **above or below $50K**  
based on demographic and work-related attributes.
""")

st.divider()

# Sidebar inputs
st.sidebar.header("📊 Input Features")

age = st.sidebar.slider("Age", 18, 90, 35)
workclass = st.sidebar.selectbox(
    "Workclass",
    ["Private", "Self-emp", "Government", "Unknown"]
)
education = st.sidebar.selectbox(
    "Education Level",
    ["HS-grad", "Bachelors", "Masters", "Some-college", "Doctorate", "Unknown"]
)
marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Married", "Single", "Divorced", "Separated", "Widowed"]
)
occupation = st.sidebar.selectbox(
    "Occupation",
    ["Tech", "Sales", "Exec-managerial", "Clerical", "Service", "Unknown"]
)
relationship = st.sidebar.selectbox(
    "Relationship",
    ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried"]
)
race = st.sidebar.selectbox(
    "Race",
    ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
native_country = st.sidebar.selectbox(
    "Native Country",
    ["United-States", "Non-US", "Unknown"]
)

# Create input dataframe
input_data = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "education": education,
    "marital_status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "hours_per_week": hours_per_week,
    "capital_gain": capital_gain,
    "capital_loss": capital_loss,
    "native_country": native_country
}])

st.subheader("🔍 Input Summary")
st.dataframe(input_data, use_container_width=True)

# Prediction
if st.button("🚀 Predict Income Class"):
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data).max()

    st.divider()

    if prediction == 1:
        st.success(f"💰 Predicted Income: **>50K**")
    else:
        st.info(f"💼 Predicted Income: **≤50K**")

    st.metric(
        label="Prediction Confidence",
        value=f"{probability * 100:.2f}%"
    )

# Footer
st.divider()
st.markdown("""
**Model Highlights**
- Best Cross-Validation Accuracy: **83.48%**
- Models Tested: Logistic Regression, KNN, SVC, Random Forest
- Features engineered with robust preprocessing

Built with ❤️ using **Streamlit & Scikit-Learn**
""")
