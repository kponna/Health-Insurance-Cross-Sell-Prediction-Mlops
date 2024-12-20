import pandas as pd  
from src.feature_engineering import (
    FeatureEngineer, LogTransformation, MinMaxScaling, 
    DropUnnecessaryColumns, FrequencyEncoding, OrdinalEncoding, MapEncoding
)
import logging
import joblib
def preprocessing(df: pd.DataFrame, strategies: list ) -> pd.DataFrame:
    """Applies a sequence of feature engineering steps based on selected strategies and saves the engineered data. 

    Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.
        strategies : list of strategies ('log','minmax','ordinal','scaler'....etc)
        
    Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """ 
    scaler_path = "/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/models/standard_scaler.pkl"  # Update with the actual path
    pretrained_scaler = joblib.load(scaler_path)
    try:
        df_transformed = df.copy()

        # Apply selected feature engineering strategies
        if 'log' in strategies:
            log_transformer = FeatureEngineer(LogTransformation(features=['Annual_Premium']))
            df_transformed = log_transformer.apply_feature_engineering(df_transformed) 

        if 'minmax' in strategies:
            minmax_transformer = FeatureEngineer(MinMaxScaling(features=['Annual_Premium']))
            df_transformed = minmax_transformer.apply_feature_engineering(df_transformed)

        if 'scaler' in strategies: 
            scaled_features = ['Vintage', 'Annual_Premium']
            df_transformed[scaled_features] = pretrained_scaler.transform(df_transformed[scaled_features])
        if 'ordinal' in strategies:
            ordinal_encoder = FeatureEngineer(OrdinalEncoding(age_feature='Age' ))
            df_transformed = ordinal_encoder.apply_feature_engineering(df_transformed)
        
        
        if 'map' in strategies:
            # Define the mappings for the features you want to encode
            feature_mappings = {
            'Gender': {'Female': 1, 'Male': 0},
            'Vehicle_Damage': {'Yes': 1, 'No': 0},
            'Vehicle_Age': {'1-2 Year': 1,
                '< 1 Year': 0,
                '> 2 Years': 2}}
            map_encoder = FeatureEngineer(MapEncoding(feature_mappings=feature_mappings))
            df_transformed = map_encoder.apply_feature_engineering(df_transformed)
        
        if 'freq' in strategies:
            frequency_encoder = FeatureEngineer(FrequencyEncoding(feature='Policy_Sales_Channel'))
            df_transformed = frequency_encoder.apply_feature_engineering(df_transformed)
        
        if 'drop' in strategies:
            drop_columns_transformer = FeatureEngineer(DropUnnecessaryColumns(features=['id', 'Driving_License', 'Age', 'Age_Label', 'Policy_Sales_Channel']))
            df_transformed = drop_columns_transformer.apply_feature_engineering(df_transformed) 

        return df_transformed
    
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise e
    
def preprocess_data(input_data: pd.DataFrame):
    """Preprocess the input data through all pipeline steps."""
     
    processed_df = preprocessing(input_data, strategies=['scaler', 'ordinal', 'map','freq']) 
    return processed_df
