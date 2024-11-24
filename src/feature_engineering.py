import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass
    def __str__(self):
        """
        Provide a string representation of the strategy.
        This will help display a meaningful description in logs.
        """
        return self.__class__.__name__

# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features: list[str]):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}") 
        df_transformed = df.copy() 
        for feature in self.features:
            if feature not in df.columns:
                logging.error(f"Feature '{feature}' not found in DataFrame. Skipping transformation.")
                continue 
            if df[feature].isnull().any() or (df[feature] <= 0).any():
                raise ValueError(f"Log transformation is not valid for feature '{feature}' due to missing or non-positive values.")
            df_transformed[feature] = np.log1p(df[feature])   
        logging.info("Log transformation completed.")
        return df_transformed 

# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features: list[str]):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        
        try:
            df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
            logging.info("Standard scaling completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during standard scaling: {e}")
            raise ValueError(f"Error during standard scaling: {e}") 
        return df_transformed
 
# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features:list[str], feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list[str]): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        try:
            df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
            logging.info("Min-Max scaling completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during Min-Max scaling: {e}")
            raise ValueError(f"Error during Min-Max scaling: {e}")   
        return df_transformed

 
# Concrete Strategy for Map Encoding
# ------------------------------------
# This strategy applies Map encoding to categorical features, converting them into numeric labels.
class MapEncoding(FeatureEngineeringStrategy):
    def __init__(self, feature_mappings):
        """
        Initializes the MapEncoding with specific mappings for features.

        Parameters:
        feature_mappings (dict): A dictionary where keys are feature names and values are dictionaries
                                  mapping original values to encoded values.
        """
        self.mappings = feature_mappings

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the provided mapping to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with mapped categorical features.
        """
        logging.info(f"Applying mappings to features: {list(self.mappings.keys())}")
        df_transformed = df.copy() 
        for feature, mapping in self.mappings.items():
            if feature in df.columns:
                df_transformed[feature] = df_transformed[feature].map(mapping)
                logging.info(f"Mapped features '{feature}' with mapping: {mapping}")
            else:
                logging.warning(f"Feature '{feature}' not found in DataFrame.")
        logging.info("Map Encoding completed.")
        return df_transformed
        
# Concrete Strategy for Ordinal Encoding 
# ------------------------------------------------------------
# This strategy applies ordinal encoding based on specific categories for age.
class OrdinalEncoding(FeatureEngineeringStrategy):
    def __init__(self, age_feature: str):
        """
        Initializes the OrdinalEncoding with the features for age and vehicle age.

        Parameters:
        age_feature (str): The name of the column containing the age data. 
        """
        self.age_feature = age_feature 

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies ordinal encoding to the age and vehicle age features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with additional ordinal-encoded age and vehicle age features.
        """
        logging.info(f"Applying ordinal encoding to '{self.age_feature}' ")
        df_transformed = df.copy()

        # Add a new column 'Age_Label' for categorical age groups
        df_transformed['Age_Label'] = pd.cut(df_transformed[self.age_feature],
                                             bins=[18, 25, 50, 100],
                                             labels=['Young(18-25)', 'Middle-Age(26-50)', 'Old Age(51-100)'],
                                             include_lowest=True)

        # Add a new column 'Age_Encoded' for ordinal encoding based on age
        df_transformed['Age_Encoded'] = pd.cut(df_transformed[self.age_feature],
                                               bins=[18, 25, 50, 100],
                                               labels=[0, 1, 2],  # Ordinal encoding: 0=Young, 1=Middle-Age, 2=Old Age
                                               include_lowest=True).astype(int)
  
        logging.info("Ordinal encoding for age completed.")
        return df_transformed


# Concrete Strategy for Frequency Encoding
# ----------------------------------------
# This strategy applies frequency encoding to categorical features, encoding based on the frequency of each category.
class FrequencyEncoding(FeatureEngineeringStrategy):
    def __init__(self, feature: str):
        """
        Initializes the FrequencyEncoding with the specific feature to encode.

        Parameters:
        feature (str): The categorical feature to apply frequency encoding to.
        """
        self.feature = feature

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies frequency encoding to the specified feature in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with frequency-encoded feature.
        """
        logging.info(f"Applying frequency encoding to feature: {self.feature}")
        df_transformed = df.copy()
 
        freq_encoding = df_transformed[self.feature].value_counts(normalize=True)
        df_transformed[f"{self.feature}_Encoded"] = df_transformed[self.feature].map(freq_encoding)

        logging.info("Frequency encoding completed.")
        return df_transformed
     
# Concrete Strategy for Dropping Unnecessary Columns 
# -----------------------------------------------------------------------------
# This strategy applies to remove columns that are not necessary for the model training process.
class DropUnnecessaryColumns(FeatureEngineeringStrategy):
    def __init__(self, features: list[str]):
        """
        Initializes the DropUnnecessaryColumns with the columns to be removed from the DataFrame.

        Parameters:
        features (list): A list of column names to be dropped.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the dropping of unnecessary columns in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with the specified columns dropped.
        """
        logging.info(f"Dropping unnecessary columns: {self.features}")
        df_transformed = df.copy()
        df_transformed.drop(columns=self.features, inplace=True)
        
        logging.info(f"Dropping of unnecessary columns completed.")
        return df_transformed


# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset. 
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info(f"Switching feature engineering strategy to {strategy}.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info(f"Applying feature engineering strategy: {self._strategy}.")
        return self._strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__": 
    # df = pd.read_csv("/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/extracted_files/train.csv")

    # log_transformer = FeatureEngineer(LogTransformation(features=['Annual_Premium']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # # Standard Scaling Example
    # ordinal_encoder = FeatureEngineer(OrdinalEncoding(age_feature='Age'))
    # df_ordinal_encoder = ordinal_encoder.apply_feature_engineering(df_log_transformed) 

    # # One-Hot Encoding Example
    # frequency_encoding = FeatureEngineer(FrequencyEncoding(feature='Policy_Sales_Channel'))
    # df_frequency_encoding = frequency_encoding.apply_feature_engineering(df_ordinal_encoder)

    # # Instantiate the strategy with the columns to drop
    # drop_columns = FeatureEngineer(DropUnnecessaryColumns(features=['id', 'Driving_License', 'Age','Age_Label' , 'Policy_Sales_Channel']))
    # df_train_transformed = drop_columns.apply_feature_engineering(df_frequency_encoding)
    # print("Before performing feature engineering.")
    # print(df.head())
    # print("After performing feature engineering.")
    # print(df_train_transformed.head())
    pass
