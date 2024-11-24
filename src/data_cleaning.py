import logging
import pandas as pd

from abc import ABC, abstractmethod
  
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") 
 
class DataCleaningStrategy(ABC):
    """
    Abstract base class that defines the interface for different data cleaning strategies.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the cleaning strategy to the provided DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to be cleaned.

        Returns:
        pd.DataFrame: Cleaned DataFrame.
        """
        pass
  
class ConvertFloatToIntStrategy(DataCleaningStrategy):
    """
    Concrete strategy to convert specified float columns to integers.
    """ 
    def __init__(self, selected_features: list = None):
        """
        Initialize the strategy with the list of selected features to be converted.
        
        Parameters:
        selected_features (list): List of column names to convert to integers.
        """
        self.selected_features = selected_features

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the float-to-int conversion only on the selected features.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with selected features converted to integers.
        """
        if self.selected_features is None:
            raise ValueError("No columns specified for float-to-int conversion.")
 
        for col in self.selected_features:
            if col in df.columns and df[col].dtype == 'float64':
                df[col] = df[col].astype('int64')
            else:
                raise ValueError(f"Column {col} is not a valid float64 column.")
        
        return df
    
 
class DropMissingValuesStrategy(DataCleaningStrategy):
    """
    Strategy to drop rows or columns with missing values based on the axis and threshold.
    """
    def __init__(self, axis=0, thresh=0):
        """
        Initialize the DropMissingValuesStrategy with specific parameters.

        Parameters: 
        axis (int): 0 to drop rows, 1 to drop columns with missing values.
        thresh (int): Minimum number of non-NA values required to keep the row/column.
        """
        self.axis = axis
        self.thresh = thresh

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped or unchanged if no missing values found.
        """
        if df.isnull().sum().sum() == 0:
            logging.info("No missing values found in the DataFrame.")
            return df  
        
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned 

 
class FillMissingValuesStrategy(DataCleaningStrategy):
    """
    Strategy to fill missing values using mean, median, mode, or a constant value.
    """
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        if df.isnull().sum().sum() == 0:
            logging.info("No missing values found in the DataFrame.")
            return df   

        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned
 
class DataCleaner: 
    """
    Context class that uses different cleaning strategies.
    """ 
    def __init__(self, strategy: DataCleaningStrategy):
        """
        Initialize the DataCleaner with a specific cleaning strategy.

        Parameters: 
        strategy (DataCleaningStrategy): The initial cleaning strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataCleaningStrategy):
        """
        Set a new cleaning strategy.
        
        Parameters:
        strategy (DataCleaningStrategy): The new cleaning strategy to use.
        """
        self._strategy = strategy

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the selected cleaning strategy to the DataFrame.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: Cleaned DataFrame.
        """
        return self._strategy.apply(df)
    
# Example Usage:
if __name__ == "__main__":
    
    # # Load the data
    # df = pd.read_csv('/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/train.csv')


    # # Convert float to int
    # strategy = ConvertFloatToIntStrategy(selected_features=['Region_Code', 'Policy_Sales_Channel'])
    # cleaner = DataCleaner(strategy)
    # try:
    #     cleaned_df = cleaner.clean(df)
    #     print("\nDataFrame after float-to-int conversion:")
    #     print(cleaned_df)
    # except ValueError as e:
    #     logging.error(e)

    # # Drop missing values
    # strategy = DropMissingValuesStrategy(axis=0, thresh=2)
    # cleaner.set_strategy(strategy)
    # cleaned_df = cleaner.clean(df)
    # print("\nDataFrame after dropping missing values:")
    # print(cleaned_df)

    # # Fill missing values
    # strategy = FillMissingValuesStrategy(method="mean")
    # cleaner.set_strategy(strategy)
    # cleaned_df = cleaner.clean(df)
    # print("\nDataFrame after filling missing values with mean:")
    # print(cleaned_df)
    pass