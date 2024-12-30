import pickle
import numpy as np
import streamlit as st

# Load the trained model and location data
with open("bangalore_home_price_model.pickle", "rb") as f:
    model_data = pickle.load(f)
    model = model_data['model']
    columns = model_data['columns']

with open("locations.pkl", "rb") as f:
    unique_locations = pickle.load(f)

# Streamlit app title
st.title("Bangalore House Price Prediction")

# Input fields for user
location = st.selectbox("Location", unique_locations)
sqft = st.number_input("Total Square Feet", min_value=300, max_value=50000, value=1000, step=10)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=20, value=2, step=1)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=20, value=2, step=1)

# Prediction function (similar to your notebook)
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(columns == location)[0][0]
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(location, sqft, bath, bhk)
    st.success(f"The predicted price is ₹{predicted_price:,.2f}")