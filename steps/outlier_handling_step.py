import logging 
import pandas as pd
from src.outlier_detection import OutlierDetector, IQROutlierDetection, ZScoreOutlierDetection
from zenml import step


@step 
def outlier_handling_step(df: pd.DataFrame, column_name: str ) -> pd.DataFrame:
    """Detects and caps outliers in a specific column using OutlierDetector with IQR strategy.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to detect and handle outliers. 

    Returns:
    pd.DataFrame: The DataFrame with outliers handled.
    """
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")
 
    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")
    
    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
 
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        logging.error(f"Column '{column_name}' is not numeric.")
        raise ValueError(f"Column '{column_name}' must be numeric for outlier detection.") 
    
    # Filter and ensure only numeric columns are passed
    df_numeric = df[[column_name]].copy() 

    # Initialize the OutlierDetector with the IQR-based Outlier Detection Strategy
    outlier_detector = OutlierDetector(IQROutlierDetection())

    # Detect and cap outliers
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="cap")

    logging.info(f"Completed outlier detection step, returning DataFrame of shape: {df_cleaned.shape}")

    # Replace the original column with the cleaned values
    df[column_name] = df_cleaned[column_name] 
    logging.info(f"Completed outlier detection step, returning DataFrame of shape: {df.shape}")
 
    return df
