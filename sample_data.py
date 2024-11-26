import json 
import requests

# URL of the MLflow prediction server
url = "http://127.0.0.1:8001/invocations"
 
# Sample input data for prediction 
input_data = {
    "dataframe_records": [
        {
            "Age_Encoded": 35,
            "Annual_Premium": 30000.0,
            "Gender": 0,
            "Policy_Sales_Channel_Encoded": 152.0,
            "Previously_Insured": 0,
            "Region_Code": 28,
            "Vehicle_Age": 2,   
            "Vehicle_Damage": 0,
            "Vintage": 20
        },
        {
            "Age_Encoded": 45,
            "Annual_Premium": 40000.0,
            "Gender": 0,
            "Policy_Sales_Channel_Encoded": 26.0,
            "Previously_Insured": 1,
            "Region_Code": 3,
            "Vehicle_Age": 1,  
            "Vehicle_Damage": 0,
            "Vintage": 2
        },
        {
            "Age_Encoded": 100,
            "Annual_Premium": 6.0,
            "Gender": 1,  
            "Policy_Sales_Channel_Encoded": 26.0,
            "Previously_Insured": 0,
            "Region_Code": 3.0,
            "Vehicle_Age": 1,   
            "Vehicle_Damage": 0,
            "Vintage": 236
        }
    ]
}
# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {"Content-Type": "application/json"}

# Send the POST request to the server
response = requests.post(url, headers=headers, data=json_data)

# Check the response status code
if response.status_code == 200: 
    prediction = response.json()
    print("Prediction:", prediction)
else:
    # If there was an error, print the status code and the response
    print(f"Error: {response.status_code}")
    print(response.text)
