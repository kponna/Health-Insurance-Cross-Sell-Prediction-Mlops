import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    if not service.is_running:
        service.start(timeout=10)  

    # Load the input data from JSON string
    data = json.loads(input_data)

    # # Extract the actual data and expected columns
    # data.pop("columns", None)  # Remove 'columns' if it's present
    # data.pop("index", None)  # Remove 'index' if it's present
 
    expected_columns = [
    'Gender', 
    'Region_Code', 
    'Previously_Insured', 
    'Vehicle_Age', 
    'Vehicle_Damage', 
    'Annual_Premium', 
    'Vintage', 
    'Age_Encoded', 
    'Policy_Sales_Channel_Encoded'
    ]

    # Convert the data into a DataFrame with the expected columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction
