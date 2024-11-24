from typing import Tuple
import pandas as pd
from src.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from zenml import step
import logging


@step
def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target.
    target_column (str): The name of the target column in the DataFrame.
    
    Returns:
    Tuple: X_train, X_test, y_train, y_test
    """
    try: 
        splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
        X_train, X_test, y_train, y_test = splitter.split(df, target_column)
        
        logging.info(f"Shape of X_train = {X_train.shape}, Shape of X_test = {X_test.shape}")
        logging.info(f"Shape of y_train = {y_train.shape}, Shape of y_test = {y_test.shape}")

        logging.debug(f"First few rows of X_train: \n{X_train.head()}")
        logging.debug(f"First few rows of X_test: \n{X_test.head()}")
        logging.debug(f"First few rows of y_train: \n{y_train.head()}")
        logging.debug(f"First few rows of y_test: \n{y_test.head()}")
         
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise e