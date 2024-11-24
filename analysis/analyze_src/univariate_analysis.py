import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod 

# Abstract Base Class for Univariate Analysis Strategy
# -----------------------------------------------------
# This class defines a common interface for univariate analysis strategies.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str, group_by: str = None, is_categorical: bool = True):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.
        group_by (str): The column to group data by for additional insights. Default is None.
        is_categorical (bool, optional): Indicates whether the feature is categorical. Default is True.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass 

# Concrete Strategy for Categorical Features
# -------------------------------------------
# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, group_by: str = None, is_categorical: bool = True):
        """
        Perform univariate analysis on a categorical feature with optional grouping.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The categorical feature/column to be analyzed.
        group_by (str, optional): The column to group data by for additional insights. Default is None.
        is_categorical (bool, optional): Indicates whether the feature is categorical. Default is True.

        Returns:
        None: Displays count plots for the feature.
        """
        sns.set(style="darkgrid")
        plt.figure(figsize=(14, 6))

        # Count plot for the feature
        plt.subplot(1, 2, 1)
        ax1 = sns.countplot(x=feature, palette="flare", data=df)
        plt.title(f"Count of {feature}", fontsize=20)
        total = len(df)
        for p in ax1.patches:
            percentage = f"{100 * p.get_height() / total:.1f}%"
            ax1.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), 
                         ha="center", va="bottom", fontsize=10)

        # Count plot grouped by another feature
        if group_by:
            plt.subplot(1, 2, 2)
            group_percentages = (
                df.groupby([feature, group_by])
                .size()
                .div(total)
                .mul(100)
                .reset_index(name="Percentage")
            )
            group_percentages = group_percentages[group_percentages["Percentage"] > 0]

            ax2 = sns.barplot(x=feature, y="Percentage", hue=group_by, 
                              data=group_percentages, palette="Set2")
            plt.title(f"Response of {feature}", fontsize=20)
            for p in ax2.patches:
                if p.get_height() > 0:
                    ax2.annotate(f"{p.get_height():.1f}%", 
                                 (p.get_x() + p.get_width() / 2, p.get_height()), 
                                 ha="center", va="bottom", fontsize=10)

        plt.tight_layout()
        plt.show()
 
 
# Concrete Strategy for Numerical Features
# -----------------------------------------
# This strategy analyzes numerical features by plotting their distribution.
class NumericalAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, group_by: str = None, is_categorical: bool = False):
        """
        Perform univariate analysis on a numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The numerical feature/column to be analyzed.
        group_by (str, optional): The column to group data by. Not used for numerical analysis. Default is None.
        is_categorical (bool, optional): Indicates whether the feature is categorical. Default is False.

        Returns:
        None: Displays histogram and violin plot for the feature.
        """
        sns.set_theme(style="darkgrid")

        # Histogram
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], bins=30, kde=True, color='#1f77b4')
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
        
        # # Box Plot
        # plt.figure(figsize=(10, 6))
        # sns.boxplot(y=df[feature], color='#ff7f0e')
        # plt.title(f"Box Plot of {feature}")
        # plt.ylabel(feature)
        # plt.show()
        
        # Violin Plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=df[feature], color='#2ca02c')
        plt.title(f"Violin Plot of {feature}")
        plt.ylabel(feature)
        plt.show()

# Context Class that uses a UnivariateAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different univariate analysis strategies.
class VisualizationContext:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self.strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self.strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str, group_by: str = None, is_categorical: bool = True):
        """
        Execute the univariate analysis using the selected strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The feature/column to analyze.
        group_by (str, optional): The column to group data by for additional insights. Default is None.
        is_categorical (bool, optional): Indicates whether the feature is categorical. Default is True.

        Returns:
        None
        """
        self.strategy.analyze(df, feature, group_by, is_categorical)

# Example usage
if __name__ == "__main__":
    # # Load the data
    # df_train = pd.read_csv('/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/train.csv')

    # # Categorical Analysis
    # categorical_strategy = CategoricalAnalysis()
    # context = VisualizationContext(categorical_strategy)
    # context.execute_analysis(df_train, feature='Gender', group_by='Response', is_categorical=True)

    # # Numerical Analysis
    # numerical_strategy = NumericalAnalysis()
    # context.set_strategy(numerical_strategy)  # Change strategy to NumericalAnalysis
    # context.execute_analysis(df_train, feature='Annual_Premium', is_categorical=False)

    # # Analyze another categorical feature
    # context.set_strategy(categorical_strategy)  # Revert to CategoricalAnalysis
    # context.execute_analysis(df_train, feature='Driving_License', group_by='Response', is_categorical=True)
    pass