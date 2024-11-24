import pandas as pd
from zenml import step
import logging 

from src.data_cleaning import (
    DataCleaner,
    ConvertFloatToIntStrategy,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
) 

@step
def data_cleaning_step(
    df: pd.DataFrame, 
    selected_features: list = None, 
    fill_method: str = "mean", 
    drop_axis: int = 0, 
    thresh: int = 0,   
    fill_value=None 
) -> pd.DataFrame:
    """
    ZenML step to perform data cleaning by applying both float-to-int conversion and missing value handling.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    selected_features (list): List of columns to apply float-to-int conversion on.
    fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'constant').
    drop_axis (int): Axis for dropping missing values (0 = rows, 1 = columns).
    thresh (int): Threshold for dropping missing values.
    fill_value: Value to use for filling missing values (when method = 'constant'). 

    Returns:
    pd.DataFrame: Cleaned DataFrame after float-to-int conversion and missing value handling.
    """
    
    try: 
        if selected_features is None:
            selected_features = [] 

        if selected_features:
            logging.info("Applying float-to-int conversion.")
            cleaner = DataCleaner(ConvertFloatToIntStrategy(selected_features))
            df = cleaner.clean(df)
        
        if df.isnull().values.any():
            logging.info("Missing values detected in the DataFrame.")
 
            if fill_method:
                logging.info(f"Handling missing values with fill method: {fill_method}")
                missing_value_cleaner = DataCleaner(FillMissingValuesStrategy(method=fill_method, fill_value=fill_value))
                df = missing_value_cleaner.clean(df)
            else:
                logging.info("Fill method not provided")
                
            if drop_axis is not None and thresh is not None:
                logging.info(f"Dropping missing values with axis={drop_axis} and threshold={thresh}")
                drop_missing_cleaner = DataCleaner(DropMissingValuesStrategy(axis=drop_axis, thresh=thresh))
                df = drop_missing_cleaner.clean(df)
            else:
                logging.info("Drop axis and threshold not provided") 
        else:
            logging.info("No missing values detected in the DataFrame.")
               
        if df.empty:
            logging.warning("The cleaned DataFrame is empty after processing.")
        logging.info("Data cleaning completed successfully") 

        return df 
    
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise e