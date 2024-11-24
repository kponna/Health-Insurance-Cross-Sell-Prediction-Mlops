from abc import ABC, abstractmethod 
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())

# Concrete Strategy for Data Wrangling Inspection
# -----------------------------------------------
# This strategy analyzes the dataset and provides a summary of each column.
class DataWranglingInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Analyzes the dataset and provides a summary of each column in a DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        pd.DataFrame: A DataFrame containing the summary of each column.
        """ 
        df_desc = []
         
        for column in df.columns:
            df_desc.append([
                column,
                df[column].dtypes,  # Data type of the column
                df[column].isnull().sum(),  # Number of null values
                round(df[column].isnull().sum() / len(df) * 100, 2),  # Percentage of null values
                df[column].nunique(),  # Number of unique values
                df[column].unique()[:5],  # First 5 unique values 
                df.duplicated().sum()  # Total number of duplicate rows
            ])
        
        # Creating a DataFrame to store the column summaries
        column_desc = pd.DataFrame(df_desc, columns=[
            'Column', 'Dtype', 'Null Count', 'Null (%)', 
            'Unique Count', 'Sample Unique Values', 'Duplicate Rows'
        ])
        print("\nData Wrangling Summary:")
        print(column_desc)  

# Concrete Strategy for Summary Statistics Inspection
# -----------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))

# Concrete Strategy for Multicollinearity Inspection Using VIF  
# -----------------------------------------------------
# This strategy checks multicollinearity between the features.
class VIFInspectionStrategy(DataInspectionStrategy):
    
    def calc_vif(self,  df: pd.DataFrame):
        """
        Calculates Variance Inflation Factor (VIF) for each feature in the given DataFrame.

        Parameters:
         df (pd.DataFrame): The dataframe containing the features for VIF calculation.

        Returns:
        pd.DataFrame: DataFrame with variables and their corresponding VIF values.
        """
        vif = pd.DataFrame()
        vif["variables"] = df.columns
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        
        return vif

    def inspect(self, df: pd.DataFrame):
        """
        Checks multicollinearity in the dataset using VIF (Variance Inflation Factor).

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.
        target_column (str): The column to exclude from VIF calculation (e.g., 'Response').

        Returns:
        pd.DataFrame: DataFrame containing the VIF values for each feature.
        """
        df_int = df[['Age','Driving_License', 'Region_Code','Previously_Insured', 'Annual_Premium','Policy_Sales_Channel', 'Vintage', 'Response']]  
      
        columns_to_include = df_int[[i for i in df_int.describe().columns if i not in ['Response','id']]]
         
        # Calculating VIF using the class's calc_vif method
        vif = self.calc_vif(columns_to_include)
        
        print("\nVIF Inspection Results:")
        print(vif) 


# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """ 
        self._strategy.inspect(df)


# Example usage
if __name__ == "__main__":
    
    # # Load the data
    # df = pd.read_csv('/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/train.csv')

    # # Initialize the Data Inspector with DataTypesInspectionStrategy
    # print("=== Data Types Inspection ===")
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # # Switch to DataWranglingInspectionStrategy
    # print("\n=== Data Wrangling Inspection ===")
    # inspector.set_strategy(DataWranglingInspectionStrategy())
    # inspector.execute_inspection(df)

    # # Switch to SummaryStatisticsInspectionStrategy
    # print("\n=== Summary Statistics Inspection ===")
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)

    # # Switch to VIFInspectionStrategy
    # print("\n=== VIF Inspection ===")
    # inspector.set_strategy(VIFInspectionStrategy())
    # inspector.execute_inspection(df)
    
    pass
