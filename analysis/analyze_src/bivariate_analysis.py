from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Categorical vs Target Analysis
# -------------------------------
# This strategy provides plot for analyzing a categorical feature against the target.
class CategoricalVsTargetAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str = None):
        """
        Provides a plot to analyze a categorical feature against the target.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature.
        feature2 (str): The name of the target column (for classification).

        Returns:
        None: Displays a bar plot showing the counts of the categorical feature across the target classes.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature1, hue=feature2, data=df)
        plt.title(f"Distribution of {feature1} across {feature2}")
        plt.xlabel(feature1)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Categorical vs Numerical Analysis
# ----------------------------------
# This strategy provides box plots and Histogram plots to analyze a categorical feature against a numerical feature.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Provides a box plot and violin plot to analyze a categorical feature against a numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature.
        feature2 (str): The name of the numerical feature.

        Returns:
        None: Displays both a box plot and a violin plot showing the relationship between the categorical and numerical features.
        """
        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df,palette="rocket_r")
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show() 

        # Histogram
        categories = df[feature1].unique()
        plt.figure(figsize=(12, 8))
        for category in categories:
            subset = df[df[feature1] == category]
            sns.histplot(subset[feature2], kde=True, label=f"{category}", alpha=0.6, binwidth=5)
        plt.title(f"Distribution of {feature2} for each category in {feature1} (Histogram)")
        plt.xlabel(feature2)
        plt.ylabel("Frequency")
        plt.legend(title=feature1)
        plt.show()

# Numerical vs Target Analysis
# ----------------------------
# This strategy provides box plots and swarm plots to analyze a numerical feature against the target.
class NumericalVsTargetAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, target: str = 'Response'):
        """
        Provides a box plot and swarm plot to analyze a numerical feature against the target.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the numerical feature.
        target (str): The name of the target column (for classification).

        Returns:
        None: Displays both a box plot and a swarm plot showing the relationship between the numerical feature and the target.
        """
        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=target, y=feature1, data=df)
        plt.title(f"{feature1} vs {target} (Box Plot)")
        plt.xlabel(target)
        plt.ylabel(feature1)
        plt.show()

        # Swarm Plot
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x=target, y=feature1, data=df)
        plt.title(f"{feature1} vs {target} (Swarm Plot)")
        plt.xlabel(target)
        plt.ylabel(feature1)
        plt.show()


# Numerical vs Numerical Analysis
# -------------------------------
# This strategy provides scatter plots and correlation heatmaps to analyze two numerical features.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Provides a scatter plot and a correlation heatmap to analyze two numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature.
        feature2 (str): The name of the second numerical feature.

        Returns:
        None: Displays both a scatter plot and a correlation heatmap showing the relationship between the two numerical features.
        """
        # Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2} (Scatter Plot)")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

        # Correlation Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[[feature1, feature2]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlation between {feature1} and {feature2}")
        plt.show()


# Context Class that uses a BivariateAnalysisStrategy
# ---------------------------------------------------
# This class allows you to switch between different bivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature1, feature2)


# Example usage
if __name__ == "__main__":
      
    # # Load the data
    # df = pd.read_csv('/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/train.csv')

    # # Categorical vs Target Analysis
    # analyzer = BivariateAnalyzer(CategoricalVsTargetAnalysis())
    # analyzer.execute_analysis(df, 'Gender', 'Response')

    # # Categorical vs Numerical Analysis
    # analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'Gender', 'Age')

    # # Numerical vs Target Analysis
    # analyzer.set_strategy(NumericalVsTargetAnalysis())
    # analyzer.execute_analysis(df, 'Age', 'Response')

    # # Numerical vs Numerical Analysis
    # analyzer.set_strategy(NumericalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'Vintage', 'Age')

    pass
