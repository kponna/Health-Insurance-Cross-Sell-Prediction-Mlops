import streamlit as st
import requests
import json

# Streamlit App Title
st.title("Health Insurance Cross-Selling Predictor")

# Sidebar for user inputs
with st.sidebar:
    st.header("User Input Data")
    
    # Inputs for the features
    Age_Encoded = st.number_input("Age (in years)", min_value=18, max_value=100, value=35)
    Annual_Premium = st.number_input("Annual Premium", min_value=0.0, value=30000.0)
    
    # Gender encoding
    Gender = st.selectbox("Gender", options=['Male', 'Female'])
    Gender_Encoded = 0 if Gender == 'Male' else 1
    
    Policy_Sales_Channel_Encoded = st.number_input("Policy Sales Channel", min_value=0.0, value=152.0)
    
    # Previously Insured encoding
    Previously_Insured = st.selectbox("Previously Insured", options=['No', 'Yes'])
    Previously_Insured_Encoded = 0 if Previously_Insured == 'No' else 1
    
    Region_Code = st.number_input("Region Code", min_value=0, max_value=50, value=28)
    
    # Vehicle Age encoding
    Vehicle_Age = st.selectbox("Vehicle Age", options=['< 1 Year', '1-2 Year', '> 2 Years'])
    Vehicle_Age_Encoded = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}[Vehicle_Age]
    
    # Vehicle Damage encoding
    Vehicle_Damage = st.selectbox("Vehicle Damage", options=['No', 'Yes'])
    Vehicle_Damage_Encoded = 0 if Vehicle_Damage == 'No' else 1
    
    Vintage = st.number_input("Vintage (number of days)", min_value=0, max_value=300, value=20)
    
    # Submit button
    submit = st.button("Get Prediction")

if submit:
    # Prepare input data in the format expected by the MLflow server
    input_data = {
        "dataframe_records": [
            {
                "Age_Encoded": Age_Encoded,
                "Annual_Premium": Annual_Premium,
                "Gender": Gender_Encoded,
                "Policy_Sales_Channel_Encoded": Policy_Sales_Channel_Encoded,
                "Previously_Insured": Previously_Insured_Encoded,
                "Region_Code": Region_Code,
                "Vehicle_Age": Vehicle_Age_Encoded,
                "Vehicle_Damage": Vehicle_Damage_Encoded,
                "Vintage": Vintage
            }
        ]
    }

    # Convert input data to JSON
    json_data = json.dumps(input_data)

    # Define the URL of the MLflow model prediction server
    url = "http://127.0.0.1:8000/invocations"

    # Send a POST request to the model server
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json_data)

    # Check response status
    if response.status_code == 200:
        prediction = response.json()
        prediction_value = prediction['predictions'][0]

        # Display user-friendly output message
        if prediction_value == 1:
            st.success("The customer is interested in buying vehicle insurance.")
        else:
            st.success("The customer is not interested in buying vehicle insurance.")
    else:
        st.error(f"Error: {response.status_code}")
        st.error(response.text) 