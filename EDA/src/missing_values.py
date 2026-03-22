from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Abstract base class for missing values
# -----------------------------------------------------------------------------------
# This class defines a template for how missing values analysis should be carried out.
# The subclasses would then be implemented specifically by the concrete classes.
class MissingValuesAnalysis(ABC):
    def analyze(self, df: pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identify the missing values from our dataset

        :param: This is the dataset we are analyzing
        :type df: pd.DataFrame
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visulize the missing values

        :param df: This is the dataset we're analyzing
        :type df: pd.DataFrame
        """
        pass


# Concrete classes for missing values
# -----------------------------------------------------------------------------------
# This will handle the simple and hard to miss values that are missing, values that are null for example.
class SimpleMissingValuesAnalysis(MissingValuesAnalysis):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values in the dataset

        :param df: Dataframe to be analyzed
        :type df: pd.DataFrame
        """
        print("\nMissing values by column:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

    def visualize_missing_values(self, df):
        """
        Creates a heatmap for the missing values

        :param self: Description
        :param df: Description
        """
        plt.figure(figsize=(12, 12))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        # How this works is that it builds on the precepace of a dataset, for every element in the dataset it is either true of false. In our case it'd be true for when the value is null (NAN). I turn off the color bar. Cos it's not neccessary really. "viridis" is a color gradient, I believe it's a purple-to-yellow gradient.

        plt.title("Missing values Heatmap")
        plt.show()

    def analyze(self, df):
        return super().analyze(df)


# Implementation
if __name__ == "__main__":
    data_path = "../File Ingestion/extracted_data/train.csv"
    df = pd.read_csv(data_path)

    analyze_missing_values = SimpleMissingValuesAnalysis(df)
    analyze_missing_values.analyze(df)
