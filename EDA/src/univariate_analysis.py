import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod


# Abstract class
# --------------------------------------------
# This is a template for the univariate analysis


class UnivariateAnalyzer(ABC):
    @abstractmethod
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        The child classes would re-iterate this method, and fit it to their subject goal.


        :param df: The dataset we're analyzing
        :param feature: The name of the feature we want to analyze.
        :type df: pd.DataFrame
        :type feature: str
        """
        pass


# Concrete class
# --------------------------------------------
# This is an implementation of the template for numeric analysis.


class NumericalUnivariateAnalyzer(UnivariateAnalyzer):
    def execute_analysis(self, df, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        :param df: The dataset we're analyzing
        :param feature: The name of the feature we want to analyze.
        :type df: pd.DataFrame
        :type feature: str
        """
        plt.figure(figsize=(14, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of '{feature}'")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# This implementation is for categorical analysis.


class CategoricalUnivariateAnalyzer(UnivariateAnalyzer):
    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar chart plot.

        :param df: Dataset we're analyzing
        :type df: pd.DataFrame
        :param feature: Category we want to analyze.
        :type feature: str
        """
        plt.figure(figsize=(14, 6))
        sns.countplot(x=feature, data=df)
        plt.title(f"Distribution of categorical feature '{feature}'")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# This is to help with context switching
class uni_analyzer:
    def __init__(self, analyzer: UnivariateAnalyzer):
        self._strategy = analyzer

    def set_analyzer(self, analyzer: UnivariateAnalyzer):
        self._strategy = analyzer

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.execute_analysis(df, feature)
