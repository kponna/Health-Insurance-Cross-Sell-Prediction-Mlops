import pandas as pd
from zenml import step
import logging

@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing the model with expected columns."""

    try:
        # Simulated data matching the expected columns from the model schema
        data = {
            "Gender": [1, 0],  # 0: Male, 1: Female 
            "Region_Code": [28, 8],  
            "Previously_Insured": [0, 1],  # 0: No, 1: Yes
            "Vehicle_Age": [1, 2],  # Encoded values for vehicle age (1: 1-2 years, 2: >2 years)
            "Vehicle_Damage": [1, 0],  # 1: Yes, 0: No
            "Annual_Premium": [40454.0, 33536.0], 
            "Vintage": [217.0, 235.0], 
            "Age_Encoded": [1, 1],  # Age as encoded integers
            "Policy_Sales_Channel_Encoded": [152.0, 26.0]  # Policy sales channel encoded as float
        }

        # Create DataFrame with the expected columns
        df = pd.DataFrame(data)

        # Convert the DataFrame to a JSON string (orient="split" for structure)
        json_data = df.to_json(orient="split")

        return json_data
    
    except Exception as e:
        logging.error(f"Error during importing data from dynamic importer: {e}")
        raise e