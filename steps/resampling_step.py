from zenml import step
from collections import Counter
import pandas as pd
from typing import Tuple
from src.resampling import RandomOversampling, SMOTEResampling, RandomUndersampling, Resampler
import logging

@step
def resampling_step(X_train: pd.DataFrame, y_train: pd.Series, method: str='random_oversampling')-> Tuple[pd.DataFrame, pd.Series]:
    """
    Resampling step to apply the specified method.

    Parameters:
    X_train (pd.DataFrame): Features of the training set.
    y_train (pd.Series): Target variable with imbalanced classes.
    method (str): The resampling method to apply ('random_oversampling', 'smote', 'random_undersampling').

    Returns:
    Output: Resampled X_train and y_train.
    """
    if method == 'random_oversampling':
        resampler = Resampler(RandomOversampling())
    elif method == 'smote':
        resampler = Resampler(SMOTEResampling())
    elif method == 'random_undersampling':
        resampler = Resampler(RandomUndersampling())
    else:
        raise ValueError("Invalid resampling method provided.")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train should be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train should be a pandas Series.")
    
    X_resampled, y_resampled = resampler.apply_resampling(X_train, y_train)
 
    return X_resampled, y_resampled