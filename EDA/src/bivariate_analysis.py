from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Abstract class
# ---------------------------------------------------------------
# This class serves as a template for the other bivariate analysis classes. It's a common interface.


class BivariateAnalyzer(ABC):
    @abstractmethod
    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        The children classes would re-iterate this method for specific analysis.

        :param df: The dataset to analyze
        :type df: pd.DataFrame
        :param feature1: First feature for analysis.
        :type feature1: str
        :param feature2: Second feature for comparism
        :type feature2: str
        """

        pass


# Concrete class
# ---------------------------------------------------------------
#  This is an implementation of the template class for numerical analysis.


class Numerical_vs_Numerical_Analyzer(BivariateAnalyzer):

    def execute_analysis(self, df, feature1, feature2):
        """
        Plots the distribution between two numerical features to find the relationship between them.

        :param df: The dataset to analyze
        :type df: pd.DataFrame
        :param feature1: First feature for analysis.
        :type feature1: str
        :param feature2: Second feature for comparism
        :type feature2: str
        """

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"Relationship between {feature1} and {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Concrete class
# ---------------------------------------------------------------
#  This is an implentation of the template class for analysis on numerical against categorical data.


class Numerical_vs_Categorical_Analyzer(BivariateAnalyzer):
    def execute_analysis(self, df, feature1, feature2):
        """
        Plots the distribution between a categorical feature and a numerical feature to find the relationships.

        :param df: The dataset to analyze
        :type df: pd.DataFrame
        :param feature1: First feature for analysis.
        :type feature1: str
        :param feature2: Second feature for comparism
        :type feature2: str
        """

        plt.figure(figsize=(16, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"Relationship between {feature1} and {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class Categorical_vs_Categorical_Analyzer(BivariateAnalyzer):
    def execute_analysis(self, df, feature1, feature2):
        """
        Analyzes the relationship between two categorical featues using a cross-tabulation and a stacked bar chart.

        :param df: The dataset to analyze
        :type df: pd.Dataframe
        :param feature1: The independent categorical features (x-axis)
        :param feature2: The dependent categorical feature (hue/stack)
        :type feature2: str
        """

        # 1. Contingency table (Cross-tabulation). Calculates the frequency of each combination

        contigency_table = pd.crosstab(df[feature1], df[feature2])

        # 2. Plotting
        plt.figure(figsize=(12, 7))

        # Stacked bar chart to show the composition of feature2 within feature1
        contigency_table.plot(kind="bar", stacked=True, ax=plt.gca())

        plt.title(f"Distribution of {feature2} across {feature1}")
        plt.xlabel(feature1)
        plt.ylabel("Count")
        plt.legend(title=feature2, bbox_to_anchor=(1.05, 1), loc="upperleft")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# This method helps switch between classes, to avoid re-implementing the class objects.
class bi_analyzer:
    def __init__(self, analyzer: BivariateAnalyzer):
        self._analyzer = analyzer

    def set_analyzer(self, analyzer: BivariateAnalyzer):
        self._analyzer = analyzer

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self._analyzer.execute_analysis(df, feature1, feature2)
