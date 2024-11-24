import logging
from steps.data_ingestion_step import data_ingestion_step
from steps.data_cleaning_step import data_cleaning_step
from steps.outlier_handling_step import outlier_handling_step
from steps.feature_engineering_step import feature_engineering_step 
from steps.data_splitter_step import data_splitter_step
from steps.resampling_step import resampling_step 
from steps.model_training_step import model_training_step
from steps.model_evaluator_step import model_evaluation_step
from zenml import Model, pipeline 
 
@pipeline(
    model=Model( 
        name="Health_Insurance_Cross_Selling_Predictor"
    )
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/archive.zip",
        extract_to ="/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts"
    )     

    # Data Cleaning Step
    cleaned_data = data_cleaning_step(raw_data, selected_features= ['Region_Code', 'Policy_Sales_Channel'],fill_method= "mean" )
    
    # Feature Engineering Step
    engineered_data = feature_engineering_step(cleaned_data,strategies=['log','ordinal','map','freq','drop'])
    
    # Outlier Detection Step 
    outlier_treated_data = outlier_handling_step(engineered_data, column_name="Annual_Premium")
     
    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitter_step(outlier_treated_data, target_column="Response" ) 
   
    # Resampling step 
    X_resampled, y_resampled = resampling_step(X_train = X_train, y_train = y_train, method = "random_oversampling")
      
    # Model Building Step
    trained_model_pipeline = model_training_step(X_train = X_resampled, y_train = y_resampled, strategy = "logistic_regression",fine_tuning=False)

    # Model Evaluation Step 
    evaluator = model_evaluation_step(trained_model_pipeline, X_test = X_test, y_test = y_test) 

    return evaluator  

if __name__ == "__main__": 
    # try: 
    #     logging.info("Starting the ML pipeline.")
    #     run = ml_pipeline()
    #     logging.info("ML pipeline completed successfully.")
    # except Exception as e:
    #     logging.error("An error occurred while running the training pipeline: %s", str(e))
    pass