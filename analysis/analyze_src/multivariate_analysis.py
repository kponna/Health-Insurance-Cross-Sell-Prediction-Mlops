from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis. 
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a pair plot.
        """
        pass
    
    @abstractmethod
    def check_outliers_with_boxplot(self, df: pd.DataFrame):
        """
        Generate and display Outliers in the selected features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a box plot.
        """
        pass

    @abstractmethod
    def visualize_lineplot(self, df: pd.DataFrame, x: str, y: str, hue: str = None, palette: list= None):
        """
        Generates a line plot to visualize the relationship between two features, with an optional categorization 
        by a third feature ('hue').

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        x (str): The name of the feature for the x-axis.
        y (str): The name of the feature for the y-axis.
        hue (str, optional): The name of the feature for color categorization (default is None).
        palette (str, optional): The color palette for the plot (default is 'husl').

        Returns:
        None: Displays a line plot.
        """
        pass   

    @abstractmethod
    def generate_histograms(self, df: pd.DataFrame):
        """
        Generates histograms for numerical columns in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays histograms after outlier treatment.
        """
        pass  

# Concrete Class for Multivariate Analysis with Correlation Heatmap and Pair Plot
# -------------------------------------------------------------------------------
# This class implements the methods to generate different plot.
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """ 
        plt.figure(figsize=(20, 8))  
        sns.heatmap(df.corr(), annot=True)   
        plt.title("Correlation Heatmap")
        plt.show()


    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

    def check_outliers_with_boxplot(self, df: pd.DataFrame):
        """
        Checks for outliers using box plots for numerical features (excluding object features).

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays box plots for each numerical feature in a neat grid layout.
        """ 
        df_box = df.select_dtypes(include=['float64', 'int64']).drop(columns=['id'], errors='ignore')
 
        num_cols = df_box.columns.size
        rows = (num_cols // 3) + (1 if num_cols % 3 else 0)

        plt.figure(figsize=(20, rows * 5))  
        for idx, column in enumerate(df_box.columns, 1):
            plt.subplot(rows, 3, idx)
            sns.boxplot(x=df_box[column])
            plt.title(f"Box Plot - {column}", fontsize=12)
            plt.xlabel(column, fontsize=10)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def visualize_lineplot(self, df: pd.DataFrame, x: str, y: str, hue: str = None, palette: list= None):
        """
        Generates a line plot to visualize the relationship between two features, with an optional categorization 
        by a third feature ('hue').

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        x (str): The name of the feature for the x-axis.
        y (str): The name of the feature for the y-axis.
        hue (str, optional): The name of the feature for color categorization (default is None).
        palette (str, optional): The color palette for the plot (default is 'husl').

        Returns:
        None: Displays a line plot.
        """
        plt.figure(figsize=(15, 8))
         
        sns.lineplot(data=df, x=x, y=y, hue=hue, palette=palette)
        
        plt.title(f"Line Plot - {x} vs. {y} by {hue}" if hue else f"Line Plot - {x} vs. {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        if hue:
            plt.legend(title=hue)
        plt.show()

    def generate_histograms(self, df: pd.DataFrame):
        """
        Generates histograms for numerical features after outlier treatment.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays histograms for numerical features in a neat grid layout.
        """ 
        outlier_treated_df = df.select_dtypes(include=['float64', 'int64'])
 
        num_cols = outlier_treated_df.columns.size
        rows = (num_cols // 3) + (1 if num_cols % 3 else 0)

        plt.figure(figsize=(20, rows * 5))   
        for idx, column in enumerate(outlier_treated_df.columns, 1):
            plt.subplot(rows, 3, idx)
            plt.hist(outlier_treated_df[column], color='pink', edgecolor='black', bins=30)
            plt.title(f"Histogram - {column}", fontsize=12)
            plt.xlabel(column, fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.gcf().set_facecolor('white')
        plt.show()
             
# Example usage
if __name__ == "__main__": 

    # # Load the data
    # df = pd.read_csv('/home/karthikponna/kittu/Health Insurance Cross Sell Prediction Mlops Project/Health-Insurance-Cross-Sell-Prediction-Mlops/data/artifacts/train.csv')

    # # Instantiate the analysis class
    # analysis = SimpleMultivariateAnalysis()

    # # Perform Correlation Heatmap Analysis 
    # analysis.generate_correlation_heatmap(df)

    # # Perform Pair Plot Analysis 
    # analysis.generate_pairplot(df)

    # # Check Outliers using Box Plot 
    # analysis.check_outliers_with_boxplot(df)

    # # Generate Line Plot 
    # analysis.visualize_lineplot(df, x="Age", y="Salary", hue="Department")

    # # Generate Histograms 
    # analysis.generate_histograms(df)

    pass
