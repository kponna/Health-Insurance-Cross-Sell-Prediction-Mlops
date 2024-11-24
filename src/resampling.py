import logging
from abc import ABC, abstractmethod
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler 
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Strategy interface for resampling
class ResamplingStrategy(ABC):
    @abstractmethod
    def apply_resampling(self, X_train: pd.DataFrame, y_train: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling to the dataset.

        Parameters:
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Target variable with imbalanced classes.

        Returns:
        tuple: Resampled X_train and y_train as a tuple (X_resampled, y_resampled).
        """
        pass
 
# Concrete strategy for Random Oversampling
class RandomOversampling(ResamplingStrategy):
    def apply_resampling(self, X_train: pd.DataFrame, y_train: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
        """
        Apply Random Oversampling to the dataset.

        Parameters:
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Target variable with imbalanced classes.

        Returns:
        tuple: Resampled X_train and y_train.
        """
        logging.info("Applying Random Oversampling.")
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        logging.info('Original dataset shape {}'.format(Counter(y_train)))
        logging.info('Resampled dataset shape {}'.format(Counter(y_resampled)))  
        logging.info("Random oversampling completed.")
        return X_resampled, y_resampled


# Concrete strategy for SMOTE (Synthetic Minority Over-sampling Technique)
class SMOTEResampling(ResamplingStrategy):
    def apply_resampling(self, X_train: pd.DataFrame, y_train: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to the dataset.

        Parameters:
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Target variable with imbalanced classes.

        Returns:
        tuple: Resampled X_train and y_train.
        """
        logging.info("Applying SMOTE.")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logging.info('Original dataset shape {}'.format(Counter(y_train)))
        logging.info('Resampled dataset shape {}'.format(Counter(y_resampled))) 
        logging.info("SMOTE resampling completed.")
        return X_resampled, y_resampled


# Concrete strategy for Random Undersampling
class RandomUndersampling(ResamplingStrategy):
    def apply_resampling(self, X_train: pd.DataFrame, y_train: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
        """
        Apply Random Undersampling to the dataset.

        Parameters:
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Target variable with imbalanced classes.

        Returns:
        tuple: Resampled X_train and y_train.
        """
        logging.info("Applying Random Undersampling.")
        logging.info(f"Before resampling: {y_train.value_counts()}")
        
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        logging.info('Original dataset shape {}'.format(Counter(y_train)))
        logging.info('Resampled dataset shape {}'.format(Counter(y_resampled)))  
        logging.info("Random undersampling completed.") 
        return X_resampled, y_resampled


# Context Resampler class to apply the Resampling strategy
class Resampler:
    def __init__(self, strategy: ResamplingStrategy):
        """
        Initialize the Resampler with a resampling strategy.

        Parameters:
        strategy (ResamplingStrategy): The strategy to be used for resampling.
        """ 
        self._strategy = strategy

    def set_strategy(self, strategy: ResamplingStrategy)-> None:
        """
        Set a new resampling strategy.

        Parameters:
        strategy (ResamplingStrategy): The new resampling strategy to be used.
        """
        logging.info("Switching Resampling strategy.")
        self._strategy = strategy

    def apply_resampling(self, X_train: pd.DataFrame, y_train: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
        """
        Apply the selected resampling strategy.

        Parameters:
        X_train (pd.DataFrame): Features of the training set.
        y_train (pd.Series): Target variable with imbalanced classes.

        Returns:
        tuple: Resampled X_train and y_train.
        """
        logging.info("Applying Resampling strategy.")
        return self._strategy.apply_resampling(X_train, y_train)


# Example usage  
if __name__ == "__main__": 
    # # Applying Random Oversampling
    # resampler = Resampler(RandomOversampling())
    # X_resampled, y_resampled = resampler.apply_resampling(X_train, y_train) 

    # # Applying SMOTE
    # resampler = Resampler(SMOTEResampling())
    # X_resampled, y_resampled = resampler.apply_resampling(X_train, y_train) 

    # # Applying Random Undersampling
    # resampler = Resampler(RandomUndersampling())
    # X_resampled, y_resampled = resampler.apply_resampling(X_train, y_train) 

    pass
