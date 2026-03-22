from abc import ABC, abstractmethod
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Abstract class
# ---------------------------------------------------------------
# This class serves as a template for the other multi-variate analysis classes. It's a common interface.


class MultivariateAnalyzer(ABC):
    @abstractmethod
    def execute_analysis(self, df: pd.DataFrame, sizeX: int, sizeY: int):
        """
        Performs a comprehensive multivariate analysis

        :param df: The dataset we're analysing.
        :type df: pd.DataFrame
        """

        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame, sizeX: int, sizeY: int):
        """
        Generate and display a heatmap of the correlations between the elements in the dataset.

        :param df: Dataset to analyze.
        :type df: pd.DataFrame
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate a plot for all selected features in the dataset.

        :param df: Dataset to analyze.
        :type df: pd.DataFrame
        """
        pass


# Concrete class
# ---------------------------------------------------------------
# This class would implement the template of the abstract class.


class SimpleMultivariateAnalyzer(MultivariateAnalyzer):
    def generate_correlation_heatmap(self, df: pd.DataFrame, sizeX: int, sizeY: int):
        """
        Generate a heatmap for the numbers in the dataset.

        :param df: Description
        :type df:Dataframe
        """
        plt.figure(figsize=(sizeX, sizeY))
        sns.heatmap(
            df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
        )
        plt.title("Correlation Heatmap for features")

        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate a pair plot for teh features in the dataset.

        :param df: Dataset for analysis
        :type df: DataFrame
        """

        sns.pairplot(df)
        plt.suptitle("Pairplot for features in dataset", y=0.2)
        plt.show()

    def execute_analysis(self, df: pd.DataFrame, sizeX: int, sizeY: int):
        self.generate_correlation_heatmap(df, sizeX, sizeY)
        self.generate_pairplot(df)


class multi_analyzer:
    def __init__(
        self,
        analyzer: MultivariateAnalyzer,
    ):
        self._analyzer = analyzer

    def set_analyzer(self, analyzer: MultivariateAnalyzer):
        self._analyzer = analyzer

    def execute_analysis(self, df: pd.DataFrame, sizeX: int, sizeY: int):
        self._analyzer.execute_analysis(df, sizeX, sizeY)
