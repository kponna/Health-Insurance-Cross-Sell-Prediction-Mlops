import logging
from typing import Tuple, Annotated
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
 
class DataSplittingStrategy(ABC):
    """
    Abstract base class that defines a common interface for data splitting strategies.

    Methods: 
    split_data(df: pd.DataFrame, target_column: str): Abstract method to split the data into training and testing sets.
    """
    @abstractmethod 
    def split_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass

 
class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """
    Concrete strategy for a simple train-test split using sklearn's train_test_split.
    """
    def __init__(self, test_size=0.3, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Splits the data into training and testing sets using a simple train-test split.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        ) 
        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test
 
class DataSplitter:
    """
    Context class for executing data splitting using a specified strategy.
    """
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting data using the selected strategy.")
        return self._strategy.split_data(df, target_column)


# Example usage
if __name__ == "__main__": 

    # # Load the data
    # df = pd.read_csv('your_data.csv')
 
    # # Initialize data splitter with a specific strategy
    # strategy = SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42)
    # data_splitter = DataSplitter(strategy)

    # # Perform the split
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column="Response")
    pass
