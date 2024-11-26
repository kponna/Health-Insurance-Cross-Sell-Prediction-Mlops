import json
import numpy as np
import pandas as pd
import streamlit as st
from utils import preprocess_data   
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_main

def main():
    st.title("Health Insurance Cross-Selling Predictor")
    st.write("Welcome! 🔍 Predict if a customer is ready to insure their vehicle 🚗")

    # Define columns for the data
    columns_for_df = [
        'Age', 'Annual_Premium', 'Gender', 'Policy_Sales_Channel',
        'Previously_Insured', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Vintage'
    ]

    # Initialize a dictionary to store selected input data
    selected_data = {col: None for col in columns_for_df}

    # Define categorical features and their options
    categorical_options = {
        'Gender': ['Male', 'Female'],
        'Previously_Insured': ['No', 'Yes'],
        'Vehicle_Age': ['< 1 Year', '1-2 Year', '> 2 Years'],
        'Vehicle_Damage': ['No', 'Yes']
    }

    # Collect input from user for categorical features
    for feature, options in categorical_options.items():
        selected_value = st.sidebar.selectbox(f"Select {feature}:", options)
        selected_data[feature] = selected_value
    
    # Numerical features with sliders for input
    numerical_features = {
        'Age': (18, 100, 35),
        'Region_Code': (0, 50, 28),
        'Vintage': (0, 300, 20),
        'Policy_Sales_Channel': (0, 200, 152)
    }

    # Collect numerical input using sliders
    for feature, (min_val, max_val, default_val) in numerical_features.items():
        selected_data[feature] = st.sidebar.slider(feature, min_val, max_val, default_val)

    # Input for Annual Premium using number_input
    selected_data['Annual_Premium'] = st.sidebar.number_input("Annual Premium", min_value=0.0, value=30000.0)
    selected_data['Previously_Insured'] = 0 if selected_value == 'No' else 1
    # Convert inputs to a DataFrame
    data_df = pd.DataFrame([selected_data])

    # Preprocess the data
    try:
        processed_data = preprocess_data(data_df)
        print(processed_data.head())
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return

    st.write("Selected Inputs:")
    st.write(data_df)
 
    # Prediction button
    if st.button("Predict"):
        try:
            # Load the deployed prediction service
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                step_name="mlflow_model_deployer_step"
            )

            if service is None:
                st.write("No service found. Running the deployment pipeline to create a service.")
                run_main()
            else:
                json_list = json.loads(json.dumps(list(processed_data.T.to_dict().values())))
                data = np.array(json_list) 

                # Make a prediction
                response = service.predict(data)  # Ensure correct input format
                st.write(f"Model Response: {response}")

                # Interpret prediction result
                if response[0] == 1:
                    st.success("🎉 Congratulations! The customer is eager to purchase vehicle insurance!")
                else:
                    st.error("Oops! It looks like the customer is not interested in purchasing vehicle insurance. 😔")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    main() 