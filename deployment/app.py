import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face
# Replace with your Hugging Face username
model_path = hf_hub_download(repo_id="ssowmiya/tourism-package-model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
### Wellness Tourism Package Purchase Prediction

This application predicts the likelihood of a customer purchasing the Wellness Tourism Package
based on their profile and interaction data. Please enter the customer details below to get a prediction.
""")

st.sidebar.header("Customer Information")

# Customer Details Inputs
st.sidebar.subheader("Customer Profile")

Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35, step=1)
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
Occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.sidebar.number_input("Monthly Income (in currency)", min_value=1000, max_value=100000, value=20000, step=1000)

st.sidebar.subheader("Location & Travel")
CityTier = st.sidebar.selectbox("City Tier", [1, 2, 3])
Passport = st.sidebar.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
OwnCar = st.sidebar.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfTrips = st.sidebar.number_input("Average Number of Trips per Year", min_value=0, max_value=20, value=2, step=1)

st.sidebar.subheader("Travel Companions")
NumberOfPersonVisiting = st.sidebar.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
NumberOfChildrenVisiting = st.sidebar.number_input("Number of Children (below 5 years)", min_value=0, max_value=5, value=0, step=1)

st.sidebar.subheader("Preferences")
PreferredPropertyStar = st.sidebar.selectbox("Preferred Hotel Star Rating", [3.0, 4.0, 5.0])
ProductPitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

st.sidebar.subheader("Interaction Data")
TypeofContact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
DurationOfPitch = st.sidebar.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15, step=1)
NumberOfFollowups = st.sidebar.number_input("Number of Follow-ups", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
PitchSatisfactionScore = st.sidebar.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Display input summary
st.subheader("Customer Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Age", Age)
    st.metric("Monthly Income", f"${MonthlyIncome:,}")
with col2:
    st.metric("City Tier", CityTier)
    st.metric("Number of Trips", NumberOfTrips)
with col3:
    st.metric("Persons Visiting", NumberOfPersonVisiting)
    st.metric("Satisfaction Score", PitchSatisfactionScore)

# Prediction Button
st.markdown("---")
if st.button("Predict Purchase Likelihood", type="primary"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"**High Likelihood of Purchase!**")
        st.write(f"The model predicts that this customer is **likely to purchase** the Wellness Tourism Package.")
        st.write(f"Confidence: **{prediction_proba[1]*100:.1f}%**")
    else:
        st.warning(f"**Low Likelihood of Purchase**")
        st.write(f"The model predicts that this customer is **unlikely to purchase** the Wellness Tourism Package.")
        st.write(f"Confidence: **{prediction_proba[0]*100:.1f}%**")

    # Display probability bar
    st.progress(prediction_proba[1])
    st.caption(f"Purchase Probability: {prediction_proba[1]*100:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit | Tourism Package Prediction MLOps Project")
